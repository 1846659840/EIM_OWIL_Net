"""EIM-OWILNet end-to-end pipeline.

Multi-source Feature Extraction -> Concept Bottleneck -> TD-EAIM ->
{ VAD, Open-World, Incremental, Cross-Dataset } heads.

Adds wrt earlier revision:
  * EMA teacher copy of the model + concept-intervention loss (L_civ),
    paper Eq.~3.
  * Real frozen-teacher forward inside the incremental branch so
    L_int_distill / L_coef / L_logit see the same input on both
    teacher and student.
  * Algorithm 2 helper `seed_new_prototypes_from_clustering` that uses
    HDBSCAN on the open-world unknowns to initialise the prototypes of
    new classes for the next incremental task.
  * Deterministic gate is auto-enabled in eval() via TD-EAIM.train().
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import MultiSourceFeatureExtractor
from models.concept_bottleneck import ConceptBottleneckEncoder
from models.td_eaim import TDEAIM
from models.vad_head import WeaklySupervisedVADHead
from models.open_world_head import OpenWorldDetectionHead
from models.incremental_learner import IncrementalLearningModule
from models.cross_dataset_invariance import CrossDatasetInvariance


class EMA:
    """Exponential moving average of a module's parameters & buffers."""

    def __init__(self, module: nn.Module, decay: float = 0.99):
        self.decay = float(decay)
        self.module = copy.deepcopy(module)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module):
        d = self.decay
        for ema_p, src_p in zip(self.module.parameters(), source.parameters()):
            ema_p.data.mul_(d).add_(src_p.data, alpha=1 - d)
        for ema_b, src_b in zip(self.module.buffers(), source.buffers()):
            if ema_b.dtype.is_floating_point:
                ema_b.data.mul_(d).add_(src_b.data, alpha=1 - d)
            else:
                ema_b.data.copy_(src_b.data)

    def to(self, device):
        self.module.to(device)
        return self


class EIMOWILNet(nn.Module):
    """Top-level network."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = MultiSourceFeatureExtractor(cfg)
        self.concept_bottleneck = ConceptBottleneckEncoder(cfg)
        self._load_concept_text_embeddings()
        self.td_eaim = TDEAIM(cfg)
        self.vad_head = WeaklySupervisedVADHead(cfg)
        self.open_world_head = OpenWorldDetectionHead(cfg)
        self.incremental_module = IncrementalLearningModule(cfg)
        self.cross_dataset_invariance = CrossDatasetInvariance(cfg)
        self.ema_teacher: EMA = None  # populated lazily once the model is on device

    # ---------------- I/O helpers ----------------

    def _load_concept_text_embeddings(self):
        path = self.cfg.concept_bottleneck.get("text_embeddings_path")
        if path and not os.path.exists(path):
            package_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            if os.path.exists(package_path):
                path = package_path
        if path and os.path.exists(path):
            embeddings = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(embeddings, dict):
                embeddings = embeddings.get("text_embeddings", embeddings.get("embeddings"))
            self.concept_bottleneck.prompt_bank.set_text_embeddings(embeddings)
        elif self.cfg.concept_bottleneck.get("require_text_embeddings", False):
            raise FileNotFoundError(
                f"Missing CLIP concept text embeddings at {path}. "
                "The paper protocol requires five CLIP text prompts per concept."
            )

    # ---------------- Core pipeline ----------------

    def encoder_pipeline(self, x):
        """f_t, f_v, f_c, f_o, A_t."""
        f_t, f_v, f_c, f_o = self.feature_extractor(x, return_individual=True)
        A_t, similarities = self.concept_bottleneck(f_t, F_C_t=f_c)
        return f_t, f_v, f_c, f_o, A_t, similarities

    def score_from_concepts(self, A_t):
        return self.td_eaim(A_t)

    def forward(self, x, labels=None, anomaly_classes=None, domain_ids=None,
                task_id=None, compute_open_world=True, compute_incremental=False,
                compute_invariance=False, compute_civ: bool = False):
        f_t, f_v, f_c, f_o, A_t, similarities = self.encoder_pipeline(x)
        s_t_raw, interaction_dict, h_int, sparsity_loss = self.td_eaim(A_t)
        s_t, vad_loss_dict = self.vad_head(s_t_raw, interaction_dict, labels)

        outputs = {
            "f_t": f_t,
            "A_t": A_t,
            "s_t_raw": s_t_raw,
            "s_t": s_t,
            "interaction_dict": interaction_dict,
            "h_int": h_int,
            "sparsity_loss": sparsity_loss,
            "vad_losses": vad_loss_dict,
            "dynamic_k": vad_loss_dict.get("dynamic_k") if isinstance(vad_loss_dict, dict) else None,
            "similarities": similarities,
        }

        # ----- Open-world -----
        # Note: the open-world CE loss is computed inside CompositeLoss
        # (paper L_open + nu*r_t) -- we deliberately do NOT compute it
        # here a second time to avoid duplicate gradient graphs.
        if compute_open_world:
            ow_results = self.open_world_head(s_t, h_int, interaction_dict)
            outputs["open_world"] = ow_results
            if labels is not None and anomaly_classes is not None:
                self.open_world_head.known_classifier.update_prototypes(
                    h_int.detach(), anomaly_classes, labels,
                )

        # ----- Concept-intervention loss (paper Eq. 3) -----
        if compute_civ and self.ema_teacher is not None and self.training:
            civ_pack = self._compute_civ_pack(x, A_t)
            outputs["civ"] = civ_pack

        # ----- Incremental -----
        if compute_incremental and task_id is not None:
            teacher_outputs = None
            if self.incremental_module.has_teacher():
                teacher_outputs = self.incremental_module.teacher_forward(x)
            if labels is not None and anomaly_classes is not None:
                self.incremental_module.update_memory(
                    h_int.detach(), labels, anomaly_classes,
                    interaction_dict, task_id,
                )
            inc_loss, inc_loss_dict = self.incremental_module.compute_incremental_loss(
                self, interaction_dict, h_int, anomaly_classes,
                teacher_outputs=teacher_outputs,
            )
            outputs["inc_loss"] = inc_loss
            outputs["inc_loss_dict"] = inc_loss_dict

        # ----- Cross-dataset invariance -----
        if compute_invariance:
            inv_loss_dict = self.cross_dataset_invariance(h_int, domain_ids)
            inv_loss = self.cross_dataset_invariance.compute_invariance_loss(
                interaction_dict, domain_ids,
            )
            inv_loss_dict["invariance_loss"] = inv_loss
            outputs["inv_loss_dict"] = inv_loss_dict

            # Cache the *current* batch's weights for *future* invariance comparison
            if domain_ids is not None:
                for d in domain_ids.detach().unique():
                    self.cross_dataset_invariance.update_domain_weights(int(d.item()), interaction_dict)

        return outputs

    # ---------------- L_civ helpers ----------------

    def _compute_civ_pack(self, x, A_t):
        """Compute student/teacher scores under a random concept clamp.

        Picks one concept index per video and clamps it to {0, 1} with
        equal probability, then re-runs the (student, teacher) forwards
        on the perturbed concept activation. Teacher uses the EMA copy
        and is detached.
        """
        B, T, K = A_t.shape
        device = A_t.device
        k_idx = torch.randint(0, K, (B,), device=device)
        values = torch.randint(0, 2, (B,), device=device).float()
        A_intv = self.concept_bottleneck.intervene_concepts(A_t, k_idx, values)

        student_s, _, _, _ = self.td_eaim(A_intv)

        with torch.no_grad():
            teacher = self.ema_teacher.module
            t_f_t, _, t_f_c, _ = teacher.feature_extractor(x, return_individual=True)
            t_A_t, _ = teacher.concept_bottleneck(t_f_t, F_C_t=t_f_c)
            t_A_intv = teacher.concept_bottleneck.intervene_concepts(t_A_t, k_idx, values)
            teacher_s, _, _, _ = teacher.td_eaim(t_A_intv)

        return {"student": student_s, "teacher": teacher_s}

    def attach_ema_teacher(self, decay: float = 0.99):
        if self.ema_teacher is None:
            self.ema_teacher = EMA(self, decay=decay).to(next(self.parameters()).device)
        return self.ema_teacher

    def update_ema_teacher(self):
        if self.ema_teacher is not None:
            self.ema_teacher.update(self)

    # ---------------- Algorithm 2: cluster seeding ----------------

    @torch.no_grad()
    def seed_new_prototypes_from_clustering(self, loader, device,
                                            existing_class_ids: list,
                                            new_class_ids: list,
                                            max_samples: int = 4096):
        """Run open-world clustering and seed prototypes of NEW classes.

        Implements paper Algorithm 2 step 2:
          1. Forward unlabeled stream, collect h_t marked unknown.
          2. HDBSCAN on collected h_t to discover new clusters.
          3. Use the largest len(new_class_ids) cluster centroids as
             initial prototypes for the new tasks's classes.
        """
        if not new_class_ids:
            return
        self.eval()
        unknowns = []
        total = 0
        for batch in loader:
            if total >= max_samples:
                break
            snippets = batch[0]
            if isinstance(snippets, dict):
                snippets = {k: v.to(device) for k, v in snippets.items()}
            else:
                snippets = snippets.to(device)
            outputs = self.forward(
                snippets, compute_open_world=True, compute_incremental=False,
                compute_invariance=False,
            )
            ow = outputs["open_world"]
            mask = ow["is_unknown"]
            h_int = outputs["h_int"]
            if mask.any():
                unknowns.append(h_int[mask].detach().cpu())
                total += int(mask.sum().item())
        if not unknowns:
            return
        feats = torch.cat(unknowns, dim=0).numpy()
        labels = self.open_world_head.cluster_unknown_features(feats)
        if labels.size == 0:
            return
        # Pick top-len(new_class_ids) clusters by size
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        order = np.argsort(-counts)
        chosen = unique[order[: len(new_class_ids)]]
        for new_cls, cluster_id in zip(new_class_ids, chosen):
            members = feats[labels == cluster_id]
            if len(members) == 0:
                continue
            centroid = torch.from_numpy(members.mean(axis=0)).float().to(
                self.open_world_head.known_classifier.prototypes.device,
            )
            self.open_world_head.known_classifier.prototypes[new_cls].copy_(centroid)
            self.open_world_head.known_classifier.prototype_count[new_cls] = len(members)

    # ---------------- Explanations / faithfulness ----------------

    def get_explanation(self, outputs, top_k: int = 5, concept_names=None):
        interaction_dict = outputs["interaction_dict"]
        topk_idx, topk_vals = self.td_eaim.get_top_interactions(interaction_dict, top_k)

        if concept_names is None:
            concept_names = self.concept_bottleneck.get_concept_names()

        explanations = []
        n_first = self.td_eaim.num_concepts
        n_and = self.td_eaim.num_and_pairs
        n_or = self.td_eaim.num_or_pairs

        for idx, val in zip(topk_idx, topk_vals):
            idx = idx.item()
            val = val.item()
            if idx < n_first:
                ci = concept_names.get(idx, f"concept_{idx}")
                exp = {
                    "type": "first_order", "concepts": [ci],
                    "description": f"{ci} -> individual concept contribution",
                    "value": val,
                }
            elif idx < n_first + n_and:
                pair_idx = idx - n_first
                i = self.td_eaim.and_i[pair_idx].item()
                j = self.td_eaim.and_j[pair_idx].item()
                ci = concept_names.get(i, f"concept_{i}")
                cj = concept_names.get(j, f"concept_{j}")
                exp = {
                    "type": "AND", "concepts": [ci, cj],
                    "description": f"{ci} AND {cj} -> co-occurrence anomaly",
                    "value": val,
                }
            elif idx < n_first + n_and + n_or:
                pair_idx = idx - n_first - n_and
                i = self.td_eaim.or_i[pair_idx].item()
                j = self.td_eaim.or_j[pair_idx].item()
                ci = concept_names.get(i, f"concept_{i}")
                cj = concept_names.get(j, f"concept_{j}")
                exp = {
                    "type": "OR", "concepts": [ci, cj],
                    "description": f"{ci} OR {cj} -> alternative anomaly",
                    "value": val,
                }
            else:
                temp_idx = idx - n_first - n_and - n_or
                offset_idx = temp_idx // self.td_eaim.num_temporal_pairs
                pair_idx = temp_idx % self.td_eaim.num_temporal_pairs
                i = self.td_eaim.temp_i[pair_idx].item()
                j = self.td_eaim.temp_j[pair_idx].item()
                offset = (
                    self.td_eaim.temporal_offsets[offset_idx]
                    if offset_idx < len(self.td_eaim.temporal_offsets) else 0
                )
                ci = concept_names.get(i, f"concept_{i}")
                cj = concept_names.get(j, f"concept_{j}")
                exp = {
                    "type": "Temporal", "concepts": [ci, cj],
                    "description": f"{ci}_t THEN {cj}_t+{offset} -> temporal anomaly",
                    "value": val,
                }
            explanations.append(exp)
        return explanations

    def _score_from_higher(self, weighted_higher):
        return torch.sigmoid(self.td_eaim.bias + weighted_higher.sum(dim=-1))

    def compute_drop_at_k(self, x, labels=None, k: int = 5):
        with torch.no_grad():
            outputs_full = self.forward(x, labels=labels)
            s_full = outputs_full["s_t"]
            interaction_dict = outputs_full["interaction_dict"]
            topk_idx, _ = self.td_eaim.get_top_interactions(interaction_dict, k)
            # Only zero the higher-order portion of the union index.
            n_first = self.td_eaim.num_concepts
            higher_only = topk_idx - n_first
            higher_only = higher_only[higher_only >= 0]
            gated = interaction_dict["gated_interactions"].clone()
            if higher_only.numel() > 0:
                gated[..., higher_only] = 0.0
            linear_part = interaction_dict["weighted_first"].sum(dim=-1)
            s_dropped = torch.sigmoid(self.td_eaim.bias + linear_part + gated.sum(dim=-1))
        return s_full - s_dropped

    def compute_interaction_perturbation_scores(self, x, labels=None, top_k: int = 5):
        with torch.no_grad():
            outputs = self.forward(x, labels=labels)
            interaction_dict = outputs["interaction_dict"]
            full_gated = interaction_dict["gated_interactions"]
            linear_part = interaction_dict["weighted_first"].sum(dim=-1)
            top_idx, _ = self.td_eaim.get_top_interactions(interaction_dict, top_k)
            n_first = self.td_eaim.num_concepts
            higher_top = top_idx - n_first
            higher_top = higher_top[higher_top >= 0]

            zero = torch.zeros_like(full_gated)

            def _score(higher_part):
                return torch.sigmoid(self.td_eaim.bias + linear_part + higher_part.sum(dim=-1))

            deletion_scores = [outputs["s_t"]]
            insertion_scores = [_score(zero)]
            for k in range(1, min(top_k, higher_top.numel()) + 1):
                idx_k = higher_top[:k]
                deleted = full_gated.clone()
                deleted[..., idx_k] = 0.0
                deletion_scores.append(_score(deleted))
                inserted = zero.clone()
                inserted[..., idx_k] = full_gated[..., idx_k]
                insertion_scores.append(_score(inserted))

            only_top = insertion_scores[-1]
            without_top = deletion_scores[-1]
            return {
                "full": outputs["s_t"],
                "without_top": without_top,
                "only_top": only_top,
                "deletion": deletion_scores,
                "insertion": insertion_scores,
                "top_idx": top_idx,
            }

    # ---------------- Train-time helpers ----------------

    def freeze_backbones(self):
        if self.cfg.feature_extractor.freeze_video:
            for p in self.feature_extractor.video_backbone.parameters():
                p.requires_grad = False
        if self.cfg.feature_extractor.freeze_clip:
            for p in self.feature_extractor.clip_extractor.parameters():
                p.requires_grad = False
        if self.cfg.feature_extractor.get("freeze_motion", True):
            for p in self.feature_extractor.motion_encoder.parameters():
                p.requires_grad = False

    def get_trainable_params(self):
        backbone_params, new_module_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "feature_extractor" in name:
                backbone_params.append(param)
            else:
                new_module_params.append(param)
        return [
            {"params": backbone_params, "lr": self.cfg.train.backbone_lr},
            {"params": new_module_params, "lr": self.cfg.train.lr},
        ]
