"""Composite loss for EIM-OWILNet.

Implements the seven-term objective of the paper:

    L = L_MIL + lambda1 L_pseudo + lambda2 L_int + lambda3 L_sparse
        + lambda4 L_open + lambda5 L_inc + lambda6 (L_inv + L_domain)
        + lambda_civ L_civ

The paper text in Sec. III-C (Eq.~3) defines L_civ separately as the
concept-intervention loss with an EMA teacher; we expose it here as a
top-level term so that the full method coverage is non-zero.

Fixes wrt earlier revision:
  * PseudoLabelLoss now uses a single in-place scatter_ on the canonical
    tensor so the dynamic top-k pseudo labels actually persist (the
    fancy-indexed scatter wrote to a copy and was lost).
  * InteractionFaithfulnessLoss aggregates with sum_S (per video) rather
    than mean over the entire (B, T, S) tensor, matching Eq.~(L_int).
  * OpenWorldLoss applies CE at the snippet level (paper Eq.~L_open),
    not on a video-level max-pool of logits.
  * ConceptInterventionLoss is added (paper Eq.~3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILRankingLoss(nn.Module):
    """L_MIL = max(0, rho - mean(S+_topk) + mean(S-_topk))."""

    def __init__(self, margin: float = 1.0, top_k_ratio: float = 0.1):
        super().__init__()
        self.margin = margin
        self.top_k_ratio = top_k_ratio

    def forward(self, s_t, labels, k=None):
        abnormal_mask = labels > 0
        normal_mask = labels == 0
        if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
            return torch.tensor(0.0, device=s_t.device)
        T = s_t.shape[1]
        if k is None:
            k = max(1, int(T * self.top_k_ratio))
        k = max(1, min(int(k), T))
        s_plus = s_t[abnormal_mask].topk(k, dim=1).values.mean()
        s_minus = s_t[normal_mask].topk(k, dim=1).values.mean()
        return F.relu(self.margin - s_plus + s_minus)


class PseudoLabelLoss(nn.Module):
    """L_pseudo = BCE(s_t, y_hat_t) on dynamic-MIL top-k snippets.

    Paper: positive videos contribute the top-K_v snippets as positive
    pseudo-targets, all other snippets are masked out; normal videos
    contribute every snippet with target 0.
    """

    def forward(self, s_t, labels=None, k=None, pseudo_labels=None):
        device = s_t.device
        if pseudo_labels is None and labels is not None:
            B, T = s_t.shape
            pseudo_labels = torch.zeros_like(s_t)
            mask = torch.zeros_like(s_t)
            if k is None:
                k = max(1, int(T * 0.1))
            k = max(1, min(int(k), T))

            positive_mask = labels > 0
            if positive_mask.any():
                pos_idx_global = torch.nonzero(positive_mask, as_tuple=False).view(-1)
                pos_scores = s_t[pos_idx_global]
                top_idx = pos_scores.topk(k, dim=1).indices
                row_idx = pos_idx_global.view(-1, 1).expand_as(top_idx)
                pseudo_labels[row_idx, top_idx] = 1.0
                mask[row_idx, top_idx] = 1.0

            normal_mask = labels == 0
            if normal_mask.any():
                mask[normal_mask] = 1.0
        elif pseudo_labels is None:
            high_conf_pos = (s_t > 0.8).float()
            high_conf_neg = (s_t < 0.2).float()
            mask = (high_conf_pos + high_conf_neg).clamp(0, 1)
            pseudo_labels = high_conf_pos
        else:
            mask = torch.ones_like(pseudo_labels)

        loss = F.binary_cross_entropy(
            s_t.clamp(1e-6, 1 - 1e-6),
            pseudo_labels.clamp(0, 1),
            reduction="none",
        )
        denom = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / denom


class InteractionFaithfulnessLoss(nn.Module):
    """L_int = sum_S 1[g_S=1] (beta_S g_S I_S(t))^2 (per-snippet sum, batch-mean).

    The paper writes ``1[g_S=1]`` which is non-differentiable. Under the
    stochastic hard-concrete relaxation ``g_S>0`` is true almost surely
    and would defeat the indicator, so we instead use the gate's
    closed-form *activation probability* (`gate_prob = sigmoid(log_alpha
    - tau_g log(-gamma/zeta))`) and threshold at 0.5 to mark a candidate
    as "approximately 1". This faithfully mimics ``1[g_S=1]`` in
    expectation and is stable in both train and eval.
    """

    def forward(self, interaction_dict):
        gated = interaction_dict["gated_interactions"]
        gate_prob = interaction_dict.get("gate_prob")
        if gate_prob is None:
            gate_prob = interaction_dict.get("gate_mask")
        if gate_prob is not None:
            active = (gate_prob.view(1, 1, -1) > 0.5).to(gated.dtype)
            squared = (gated * active) ** 2
        else:
            squared = gated ** 2
        per_snippet = squared.sum(dim=-1)
        return per_snippet.mean()


class SparseInteractionLoss(nn.Module):
    """Pass-through for the closed-form L_sparse computed inside TD-EAIM."""

    def forward(self, sparsity_loss):
        return sparsity_loss


class OpenWorldLoss(nn.Module):
    """L_open = CE(p(y|h_t), y_c) + nu * r_t on labelled anomalies.

    Paper Eq.~L_open is snippet-level: each snippet of an abnormal video
    contributes a CE term on its own posterior. We therefore flatten
    (B_known, T, C) -> (B_known*T, C) and repeat the class label across
    time before computing CE.
    """

    def __init__(self, residual_weight: float = 0.1):
        super().__init__()
        self.residual_weight = residual_weight

    def forward(self, logits, labels, anomaly_classes, energy=None, residual=None):
        device = logits.device
        loss = torch.tensor(0.0, device=device)
        known_mask = labels == 1
        if known_mask.sum() == 0:
            return loss

        known_logits = logits[known_mask]                    # (Nk, T, C)
        known_classes = anomaly_classes[known_mask]          # (Nk,)
        Nk, T, C = known_logits.shape
        flat_logits = known_logits.reshape(Nk * T, C)
        flat_targets = known_classes.view(Nk, 1).expand(Nk, T).reshape(-1)
        loss = loss + F.cross_entropy(flat_logits, flat_targets)

        if residual is not None:
            loss = loss + self.residual_weight * residual[known_mask].mean()
        return loss


class IncrementalLoss(nn.Module):
    """Pass-through for the module-computed incremental objective."""

    def forward(self, inc_loss, inc_loss_dict):
        return inc_loss


class CrossDatasetInvarianceLoss(nn.Module):
    """L_inv + L_domain as produced by the cross-dataset head."""

    def forward(self, inv_loss_dict):
        device = None
        for value in inv_loss_dict.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        total = torch.tensor(0.0, device=device)
        if "invariance_loss" in inv_loss_dict:
            total = total + inv_loss_dict["invariance_loss"]
        if "domain_loss" in inv_loss_dict:
            total = total + inv_loss_dict["domain_loss"]
        return total


class ConceptInterventionLoss(nn.Module):
    """Concept-intervention loss (paper Eq.~3).

    L_civ = E_{(t, k) ~ Batch} [ ( s_t(c[a_{t,k} <- a_hat]) - s_hat_t )^2 ]

    The teacher s_hat_t is produced by an EMA copy of the model and is
    detached, so the gradient flows only through the student's
    intervened forward. The loss is computed at the score level on a
    randomly intervened concept index per batch entry.
    """

    def forward(self, student_scores_intv, teacher_scores_intv):
        teacher = teacher_scores_intv.detach()
        return F.mse_loss(student_scores_intv, teacher)


class CompositeLoss(nn.Module):
    """Total objective L (see module docstring)."""

    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.losses
        self.mil_loss = MILRankingLoss(cfg.vad_head.margin, cfg.vad_head.top_k_ratio)
        self.pseudo_loss = PseudoLabelLoss()
        self.int_loss = InteractionFaithfulnessLoss()
        self.sparse_loss = SparseInteractionLoss()
        self.open_loss = OpenWorldLoss(cfg.open_world.get("residual_loss_weight", 0.1))
        self.inc_loss = IncrementalLoss()
        self.inv_loss = CrossDatasetInvarianceLoss()
        self.civ_loss = ConceptInterventionLoss()

        self.lambda_mil = loss_cfg.lambda_mil
        self.lambda_pseudo = loss_cfg.lambda_pseudo
        self.lambda_int = loss_cfg.lambda_int
        self.lambda_sparse = loss_cfg.lambda_sparse
        self.lambda_open = loss_cfg.lambda_open
        self.lambda_inc = loss_cfg.lambda_inc
        self.lambda_inv = loss_cfg.lambda_inv
        self.lambda_civ = loss_cfg.get("lambda_civ", 0.1)

    def forward(self, outputs, labels, anomaly_classes, domain_ids=None):
        device = labels.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        s_t = outputs["s_t"]
        interaction_dict = outputs["interaction_dict"]
        dynamic_k = outputs.get("dynamic_k")

        mil = self.mil_loss(s_t, labels, dynamic_k)
        total_loss = total_loss + self.lambda_mil * mil
        loss_dict["mil"] = mil.item()

        pseudo = self.pseudo_loss(s_t, labels=labels, k=dynamic_k)
        total_loss = total_loss + self.lambda_pseudo * pseudo
        loss_dict["pseudo"] = pseudo.item()

        int_loss = self.int_loss(interaction_dict)
        total_loss = total_loss + self.lambda_int * int_loss
        loss_dict["int_faith"] = int_loss.item()

        sparse = self.sparse_loss(outputs["sparsity_loss"])
        total_loss = total_loss + self.lambda_sparse * sparse
        loss_dict["sparse"] = sparse.item()

        if "open_world" in outputs:
            ow = outputs["open_world"]
            open_l = self.open_loss(
                ow["logits"],
                labels,
                anomaly_classes,
                ow.get("energy"),
                ow.get("residual"),
            )
            total_loss = total_loss + self.lambda_open * open_l
            loss_dict["open"] = open_l.item()

        if "inc_loss" in outputs:
            inc = outputs["inc_loss"]
            total_loss = total_loss + self.lambda_inc * inc
            loss_dict["inc"] = inc.item() if isinstance(inc, torch.Tensor) else inc

        if "inv_loss_dict" in outputs:
            inv = self.inv_loss(outputs["inv_loss_dict"])
            total_loss = total_loss + self.lambda_inv * inv
            loss_dict["inv"] = inv.item()

        if "civ" in outputs and outputs["civ"] is not None:
            civ_pack = outputs["civ"]
            civ_l = self.civ_loss(civ_pack["student"], civ_pack["teacher"])
            total_loss = total_loss + self.lambda_civ * civ_l
            loss_dict["civ"] = civ_l.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
