"""Incremental Learning Module (paper Sec. III-E).

L_inc = L_logit + eta1 * L_int_distill + eta2 * L_coef + eta3 * L_proto

Three memories carry old knowledge:
  M_ex     = 20 exemplars per old class
  M_proto  = class prototypes
  M_int    = (beta^old, g^old, I^old) snapshots from a frozen teacher

Fixes wrt earlier revision:
  * `save_old_model` now keeps a real *frozen-eval* deep copy of the
    full network at the end of each task. Distillation runs the frozen
    teacher on the *current* batch so I_S^old / logits^old are computed
    with the same input as the student (no cross-sample mismatch).
  * `LogitDistillationLoss` is now actually invoked as the L_logit term
    when the old model is available; standard CE is used only as the
    fallback before any task transition.
  * `update_memory` is intended to be called at *task boundaries* by
    the trainer (after the task finishes) to snapshot M_int / M_proto;
    inside-batch update remains exposed for exemplar replenishment.
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    """Three-memory bank for incremental learning."""

    def __init__(self, memory_size: int, proto_dim: int, num_classes: int):
        self.memory_size = memory_size
        self.proto_dim = proto_dim
        self.num_classes = num_classes
        self.exemplars = defaultdict(list)
        self.prototypes = {}
        self.interactions = {}
        self.interaction_weights = {}

    def update_exemplars(self, class_id: int, features: torch.Tensor, max_per_class: int = 20):
        self.exemplars[class_id].append(features.detach().cpu())
        if len(self.exemplars[class_id]) > max_per_class:
            self.exemplars[class_id] = self.exemplars[class_id][-max_per_class:]

    def update_prototypes(self, class_id: int, prototype: torch.Tensor):
        self.prototypes[class_id] = prototype.detach().cpu()

    def update_interactions(self, task_id: int, interaction_dict: dict):
        self.interactions[task_id] = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in interaction_dict.items()
        }

    def update_interaction_weights(self, task_id: int, beta_and, beta_or, gamma_temp):
        self.interaction_weights[task_id] = {
            "beta_and": beta_and.detach().cpu(),
            "beta_or": beta_or.detach().cpu(),
            "gamma_temp": gamma_temp.detach().cpu(),
        }

    def get_exemplar_features(self, device: str = "cpu"):
        feats, lbls = [], []
        for class_id, feat_list in self.exemplars.items():
            for feat in feat_list:
                if feat.shape[0] > 0:
                    feats.append(feat.to(device))
                    lbls.append(torch.full((feat.shape[0],), class_id, device=device))
        if not feats:
            return None, None
        return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

    def get_old_interactions(self, task_id: int, device: str = "cpu"):
        if task_id not in self.interactions:
            return None
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.interactions[task_id].items()
        }

    def get_old_weights(self, task_id: int, device: str = "cpu"):
        if task_id not in self.interaction_weights:
            return None
        return {k: v.to(device) for k, v in self.interaction_weights[task_id].items()}

    def estimated_bytes(self) -> int:
        total = 0
        for feat_list in self.exemplars.values():
            for feat in feat_list:
                total += feat.numel() * feat.element_size()
        for proto in self.prototypes.values():
            total += proto.numel() * proto.element_size()
        for weights in self.interaction_weights.values():
            for value in weights.values():
                total += value.numel() * value.element_size()
        for ints in self.interactions.values():
            for value in ints.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total


class LogitDistillationLoss(nn.Module):
    """Soft KL divergence between teacher and student logits at temperature T."""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, new_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
        new_soft = F.log_softmax(new_logits / self.temperature, dim=-1)
        old_soft = F.softmax(old_logits / self.temperature, dim=-1)
        return F.kl_div(new_soft, old_soft, reduction="batchmean") * (self.temperature ** 2)


class InteractionDistillationLoss(nn.Module):
    """L_int_distill = sum_S || I_S^new(x) - I_S^old(x) ||_2^2 (per-snippet sum)."""

    def forward(self, new_interactions: dict, old_interactions: dict) -> torch.Tensor:
        device_ref = next(iter(new_interactions.values()))
        loss = torch.tensor(0.0, device=device_ref.device)
        for key in ("I_and", "I_or", "I_temp"):
            if key in new_interactions and key in old_interactions:
                new_val = new_interactions[key]
                old_val = old_interactions[key].to(device_ref.device)
                min_pairs = min(new_val.shape[-1], old_val.shape[-1])
                if min_pairs == 0:
                    continue
                diff = new_val[..., :min_pairs] - old_val[..., :min_pairs]
                loss = loss + (diff ** 2).sum(dim=-1).mean()
        return loss


class CoefficientStabilityLoss(nn.Module):
    """L_coef = sum_S ( beta_S^new - beta_S^old )^2."""

    def forward(self, new_weights: dict, old_weights: dict) -> torch.Tensor:
        device = new_weights["beta_and"].device
        loss = torch.tensor(0.0, device=device)
        for key in ("beta_and", "beta_or", "gamma_temp"):
            if key in new_weights and key in old_weights:
                new_w = new_weights[key]
                old_w = old_weights[key].to(device)
                min_len = min(new_w.shape[0], old_w.shape[0])
                if min_len == 0:
                    continue
                diff = new_w[:min_len] - old_w[:min_len]
                loss = loss + (diff ** 2).sum()
        return loss


class IncrementalLearningModule(nn.Module):
    """Implements the four-term L_inc with a real frozen teacher."""

    def __init__(self, cfg):
        super().__init__()
        inc_cfg = cfg.incremental
        eaim_cfg = cfg.td_eaim
        self.memory_size = inc_cfg.memory_size
        self.exemplar_per_class = inc_cfg.exemplar_per_class
        self.eta_distill = inc_cfg.eta_distill
        self.eta_coef = inc_cfg.eta_coef
        self.eta_proto = inc_cfg.eta_proto
        self.eta_logit = inc_cfg.get("eta_logit", 1.0)
        distill_temp = inc_cfg.get("distill_temperature", 4.0)

        proto_dim = eaim_cfg.hidden_dim
        num_classes = cfg.data.num_classes

        self.memory_bank = MemoryBank(inc_cfg.memory_size, proto_dim, num_classes)
        self.logit_distill_loss = LogitDistillationLoss(temperature=distill_temp)
        self.int_distill_loss = InteractionDistillationLoss()
        self.coef_stability_loss = CoefficientStabilityLoss()
        self.logit_classifier = nn.Linear(proto_dim, num_classes)

        self.current_task = 0
        # Frozen full-model teacher (deep-copied parent net).
        self._teacher_module: nn.Module = None

    def has_teacher(self) -> bool:
        return self._teacher_module is not None

    @torch.no_grad()
    def save_old_model(self, model: nn.Module):
        """Snapshot a frozen eval-mode copy of the current student network."""
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Avoid recursive teacher chain
        if hasattr(teacher, "incremental_module"):
            teacher.incremental_module._teacher_module = None
        self._teacher_module = teacher
        self.current_task += 1

    @torch.no_grad()
    def teacher_forward(self, x):
        """Run the frozen teacher on the current batch, return its outputs dict."""
        if self._teacher_module is None:
            return None
        return self._teacher_module(
            x,
            labels=None,
            anomaly_classes=None,
            compute_open_world=True,
            compute_incremental=False,
            compute_invariance=False,
        )

    def update_memory(self, h_int: torch.Tensor, labels: torch.Tensor,
                      anomaly_classes: torch.Tensor, interaction_dict: dict,
                      task_id: int):
        """Per-batch memory bookkeeping. Exemplars and snapshot of M_int."""
        for b in range(h_int.shape[0]):
            cls = int(anomaly_classes[b].item())
            self.memory_bank.update_exemplars(cls, h_int[b], self.exemplar_per_class)
        for cls in anomaly_classes.unique():
            mask = anomaly_classes == cls
            proto = h_int[mask].mean(dim=0).mean(dim=0)
            self.memory_bank.update_prototypes(int(cls.item()), proto)
        self.memory_bank.update_interactions(task_id, interaction_dict)

    def compute_incremental_loss(self, model, interaction_dict: dict,
                                 features: torch.Tensor, anomaly_classes: torch.Tensor,
                                 teacher_outputs: dict = None) -> tuple:
        """L_inc with same-input teacher distillation when available."""
        device = features.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # ---- L_logit: KL distillation if a teacher exists, else CE ----
        video_features = features.mean(dim=1)
        new_logits = self.logit_classifier(video_features)
        if teacher_outputs is not None and self._teacher_module is not None:
            teacher_inc = self._teacher_module.incremental_module
            with torch.no_grad():
                old_video = teacher_outputs["h_int"].mean(dim=1)
                old_logits = teacher_inc.logit_classifier(old_video)
            logit_loss = self.logit_distill_loss(new_logits, old_logits)
        else:
            logit_loss = F.cross_entropy(new_logits, anomaly_classes)
        total_loss = total_loss + self.eta_logit * logit_loss
        loss_dict["logit"] = logit_loss.item()

        # ---- eta1 * L_int_distill ----
        if teacher_outputs is not None:
            distill_loss = self.int_distill_loss(
                interaction_dict, teacher_outputs["interaction_dict"],
            )
            total_loss = total_loss + self.eta_distill * distill_loss
            loss_dict["int_distill"] = distill_loss.item()
        elif self.current_task > 0:
            old_int = self.memory_bank.get_old_interactions(self.current_task - 1, device)
            if old_int is not None:
                distill_loss = self.int_distill_loss(interaction_dict, old_int)
                total_loss = total_loss + self.eta_distill * distill_loss
                loss_dict["int_distill"] = distill_loss.item()

        # ---- eta2 * L_coef ----
        new_weights = {
            "beta_and": interaction_dict["beta_and"],
            "beta_or": interaction_dict["beta_or"],
            "gamma_temp": interaction_dict["gamma_temp"],
        }
        old_weights = None
        if teacher_outputs is not None:
            t_int = teacher_outputs["interaction_dict"]
            old_weights = {
                "beta_and": t_int["beta_and"].detach(),
                "beta_or": t_int["beta_or"].detach(),
                "gamma_temp": t_int["gamma_temp"].detach(),
            }
        elif self.current_task > 0:
            old_weights = self.memory_bank.get_old_weights(self.current_task - 1, device)
        if old_weights is not None:
            coef_loss = self.coef_stability_loss(new_weights, old_weights)
            total_loss = total_loss + self.eta_coef * coef_loss
            loss_dict["coef_stability"] = coef_loss.item()

        # ---- eta3 * L_proto ----
        proto_losses = []
        for cls, old_proto in self.memory_bank.prototypes.items():
            cls_mask = anomaly_classes == cls
            if cls_mask.any():
                new_proto = features[cls_mask].mean(dim=0).mean(dim=0)
                proto_losses.append(F.mse_loss(new_proto, old_proto.to(device)))
        if proto_losses:
            proto_loss = torch.stack(proto_losses).mean()
            total_loss = total_loss + self.eta_proto * proto_loss
            loss_dict["proto_replay"] = proto_loss.item()

        # Cache the *current* task's weights for fallback reference.
        self.memory_bank.update_interaction_weights(
            self.current_task,
            interaction_dict["beta_and"].detach(),
            interaction_dict["beta_or"].detach(),
            interaction_dict["gamma_temp"].detach(),
        )

        loss_dict["total_inc"] = total_loss.item()
        return total_loss, loss_dict
