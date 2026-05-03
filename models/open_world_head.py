"""Open-World Detection Head (paper Sec. III-D).

Three orthogonal signals are combined by an AND-criterion:
  1. Known-class posterior  p(y | h_t) = softmax( cos(h_t, p_c) / tau )
  2. Energy score           E(h_t)     = -log sum_c exp z_c(h_t)
  3. Interaction residual   r_t        = || h_t - P_known h_t ||_2,
                                         where P_known = U U^T is the
                                         orthogonal projector onto the
                                         linear span of the prototypes.

Fixes wrt earlier revision:
  * P_known is now a true orthogonal projector (SVD of the prototype
    matrix), not a softmax-weighted convex reconstruction.
  * The prototypes p_c are *statistics* (class-mean of h_t) maintained
    by an EMA buffer rather than free parameters.
  * Residual is computed in the original h_t coordinates (no L2-norm
    normalisation, which had collapsed the geometric meaning of the
    distance to the prototype subspace).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class KnownAnomalyClassifier(nn.Module):
    """Cosine prototype classifier with EMA-updated class-mean prototypes."""

    def __init__(self, proto_dim: int, num_classes: int, ema_decay: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        # Prototypes are statistics (class-mean of h_t), not free parameters.
        self.register_buffer("prototypes", torch.randn(num_classes, proto_dim) * 0.02)
        self.register_buffer("prototype_count", torch.zeros(num_classes))
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.ema_decay = float(ema_decay)

    @torch.no_grad()
    def update_prototypes(self, h_t: torch.Tensor, anomaly_classes: torch.Tensor,
                          labels: torch.Tensor):
        """EMA update of class-mean prototypes from labelled abnormal snippets.

        Args:
            h_t: (B, T, D) interaction embeddings (detached upstream).
            anomaly_classes: (B,) class indices in [0, num_classes).
            labels: (B,) video-level labels; only label==1 entries update.
        """
        if h_t.numel() == 0:
            return
        B, T, D = h_t.shape
        for b in range(B):
            if int(labels[b].item()) != 1:
                continue
            cls = int(anomaly_classes[b].item())
            if cls < 0 or cls >= self.num_classes:
                continue
            sample_mean = h_t[b].mean(dim=0)
            if self.prototype_count[cls].item() == 0:
                self.prototypes[cls].copy_(sample_mean)
            else:
                self.prototypes[cls].mul_(self.ema_decay).add_(
                    sample_mean, alpha=1.0 - self.ema_decay,
                )
            self.prototype_count[cls] += 1

    def forward(self, h_t):
        """Return logits and softmax posteriors over known classes.

        Args:
            h_t: (B, T, D)
        Returns:
            logits: (B, T, C)
            probs:  (B, T, C)
        """
        h_norm = F.normalize(h_t, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("btd,cd->btc", h_norm, p_norm)
        logits = sim / self.temperature.clamp(min=0.01)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class EnergyBasedDetector(nn.Module):
    """E(x_t) = -log sum_c exp z_c(x_t)."""

    def __init__(self, num_classes: int, threshold: float = -1.0):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

    def forward(self, logits):
        return -torch.logsumexp(logits, dim=-1)


class InteractionResidualDetector(nn.Module):
    """r_t = || h_t - P_known h_t ||_2 with P_known an orthogonal projector."""

    def __init__(self, threshold: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    @staticmethod
    def _orthogonal_basis(prototypes: torch.Tensor) -> torch.Tensor:
        """Return a (D, r) matrix U whose columns span the prototypes."""
        # prototypes: (C, D)
        try:
            U, S, _ = torch.linalg.svd(prototypes.T, full_matrices=False)
            keep = S > (S.max() * 1e-6 if S.numel() > 0 else 0.0)
            return U[:, keep]
        except RuntimeError:
            # Fallback: orthonormalise via QR on the transposed matrix.
            Q, _ = torch.linalg.qr(prototypes.T)
            return Q

    def forward(self, h_t: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, T, D)
            prototypes: (C, D)
        Returns:
            residual: (B, T)
        """
        U = self._orthogonal_basis(prototypes.detach())  # (D, r)
        # Project: P h = U U^T h. Compute via einsum to keep batched dims.
        coeff = torch.einsum("btd,dr->btr", h_t, U)  # (B, T, r)
        proj = torch.einsum("btr,dr->btd", coeff, U)  # (B, T, D)
        residual = (h_t - proj).norm(dim=-1)
        return residual


class OpenWorldDetectionHead(nn.Module):
    """Combines the three signals via the paper's AND-criterion."""

    def __init__(self, cfg):
        super().__init__()
        ow_cfg = cfg.open_world
        eaim_cfg = cfg.td_eaim
        self.num_classes = ow_cfg.num_prototypes
        proto_dim = eaim_cfg.hidden_dim
        ema_decay = ow_cfg.get("prototype_ema_decay", 0.9)

        self.known_classifier = KnownAnomalyClassifier(proto_dim, self.num_classes, ema_decay)
        self.energy_detector = EnergyBasedDetector(self.num_classes, ow_cfg.energy_threshold)
        self.residual_detector = InteractionResidualDetector(ow_cfg.residual_threshold)

        self.tau_a = ow_cfg.confidence_threshold
        self.tau_e = ow_cfg.energy_threshold
        self.tau_r = ow_cfg.residual_threshold
        self.residual_loss_weight = ow_cfg.get("residual_loss_weight", 0.1)

        self.cluster_method = ow_cfg.cluster_method
        self.cluster_eps = ow_cfg.eps
        self.cluster_min_samples = ow_cfg.min_samples

    def forward(self, s_t, h_int, interaction_dict=None):
        logits, probs = self.known_classifier(h_int)
        energy = self.energy_detector(logits)
        prototypes = self.known_classifier.prototypes
        residual = self.residual_detector(h_int, prototypes)

        max_conf, pred_class = probs.max(dim=-1)
        is_unknown = (
            (max_conf < self.tau_a)
            & (energy > self.tau_e)
            & (residual > self.tau_r)
        )
        unknown_score = residual + energy - max_conf
        return {
            "logits": logits,
            "probs": probs,
            "energy": energy,
            "residual": residual,
            "pred_class": pred_class,
            "max_confidence": max_conf,
            "is_unknown": is_unknown,
            "unknown_score": unknown_score,
        }

    def cluster_unknowns(self, h_int, is_unknown):
        unknown_features = h_int[is_unknown].detach().cpu().numpy()
        return self.cluster_unknown_features(unknown_features)

    def cluster_unknown_features(self, unknown_features):
        if len(unknown_features) < 2:
            return np.array([], dtype=int)
        if self.cluster_method == "hdbscan":
            if not HAS_HDBSCAN:
                raise ImportError(
                    "hdbscan is required for the paper protocol; install with `pip install hdbscan`."
                )
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_min_samples,
                min_samples=self.cluster_min_samples,
            ).fit(unknown_features)
        else:
            clustering = DBSCAN(
                eps=self.cluster_eps, min_samples=self.cluster_min_samples,
            ).fit(unknown_features)
        return clustering.labels_

    def compute_open_world_loss(self, logits, labels, anomaly_classes, residual=None):
        """Snippet-level CE on labelled anomalies + nu * mean residual."""
        device = logits.device
        known_mask = labels == 1
        if known_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        known_logits = logits[known_mask]              # (Nk, T, C)
        known_classes = anomaly_classes[known_mask]    # (Nk,)
        Nk, T, C = known_logits.shape
        flat_logits = known_logits.reshape(Nk * T, C)
        flat_targets = known_classes.view(Nk, 1).expand(Nk, T).reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets)
        if residual is not None:
            loss = loss + self.residual_loss_weight * residual[known_mask].mean()
        return loss
