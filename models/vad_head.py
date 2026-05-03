"""Weakly-Supervised VAD Head with Dynamic MIL (paper Sec. III-C).

Computes the dynamic top-k count K_v = ceil(alpha_MIL * T * Conf_int(V)),
with Conf_int(V) the (per-video) mean magnitude of the active gated
interactions normalised by a running training-set maximum.

Fix wrt earlier revision:
  * The extra `score_refiner` MLP that re-passed the post-sigmoid score
    through another sigmoid network (and silently changed s_t away from
    Eq.~6) has been removed. The TD-EAIM output is the canonical s_t.
  * Conf_int now restricts to the active dictionary (g_S > 0) and
    averages over (S, t) per video before normalising — exactly the
    paper definition mean_{t, S in L_dict} |beta_S g_S I_S(t)| / Z.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeaklySupervisedVADHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vad_cfg = cfg.vad_head
        self.margin = vad_cfg.margin
        self.top_k_ratio = vad_cfg.top_k_ratio
        self.dynamic_top_k = vad_cfg.dynamic_top_k
        self.alpha_topk = vad_cfg.alpha_topk
        # Running max over the training set for the Conf_int normaliser Z.
        self.register_buffer("interaction_conf_max", torch.tensor(1e-6))

    def forward(self, s_t, interaction_dict=None, labels=None):
        """Pass-through s_t (paper s_t is already the TD-EAIM sigmoid)."""
        loss_dict = {}
        if labels is not None:
            mil_loss, dynamic_k = self.compute_mil_loss(s_t, labels, interaction_dict)
            loss_dict["mil_loss"] = mil_loss
            loss_dict["dynamic_k"] = dynamic_k
        return s_t, loss_dict

    def compute_mil_loss(self, s_t, labels, interaction_dict=None):
        abnormal_mask = (labels == 1)
        normal_mask = (labels == 0)
        if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
            return torch.tensor(0.0, device=s_t.device), 1
        if self.dynamic_top_k and interaction_dict is not None:
            k = self._compute_dynamic_k(s_t, interaction_dict)
        else:
            k = max(1, int(s_t.shape[1] * self.top_k_ratio))
        T = s_t.shape[1]
        k = max(1, min(k, T))
        s_plus_mean = s_t[abnormal_mask].topk(k, dim=1).values.mean()
        s_minus_mean = s_t[normal_mask].topk(k, dim=1).values.mean()
        return F.relu(self.margin - s_plus_mean + s_minus_mean), k

    def _compute_dynamic_k(self, s_t, interaction_dict):
        """K_v = ceil(alpha_MIL * T * Conf_int(V)), Conf_int in [0, 1]."""
        gated = interaction_dict.get("gated_interactions")
        gate = interaction_dict.get("gate_mask")
        if gated is None:
            return max(1, int(s_t.shape[1] * self.top_k_ratio))

        # Restrict to the active dictionary L_dict.
        if gate is not None:
            active_mask = (gate.view(1, 1, -1) > 0).to(gated.dtype)
            denom = active_mask.sum().clamp(min=1.0)
            conf_raw = (gated.abs() * active_mask).sum() / denom
        else:
            conf_raw = gated.abs().mean()

        if self.training:
            self.interaction_conf_max.copy_(
                torch.maximum(self.interaction_conf_max, conf_raw.detach()),
            )
        conf = (conf_raw / self.interaction_conf_max.clamp(min=1e-6)).clamp(0.0, 1.0)
        T = s_t.shape[1]
        k = max(1, int(math.ceil(self.alpha_topk * T * conf.item())))
        return min(k, T)

    def get_snippet_predictions(self, s_t, threshold: float = 0.5):
        return (s_t > threshold).long()

    def get_temporal_localization(self, s_t, threshold: float = 0.5):
        preds = self.get_snippet_predictions(s_t, threshold)
        results = []
        for b in range(preds.shape[0]):
            anomaly_indices = preds[b].nonzero(as_tuple=True)[0]
            results.append(self._contiguous_segments(anomaly_indices) if len(anomaly_indices) else [])
        return results

    @staticmethod
    def _contiguous_segments(indices):
        if len(indices) == 0:
            return []
        segments = []
        start = indices[0].item()
        end = start
        for idx in indices[1:]:
            if idx.item() == end + 1:
                end = idx.item()
            else:
                segments.append((start, end))
                start = idx.item()
                end = start
        segments.append((start, end))
        return segments
