import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Multi-Source Feature Extraction Module
# Figure 1 in the paper
# ============================================================

class VideoBackbone(nn.Module):
    """Frozen VideoMAE-L feature input; projector is only for legacy validation tensors."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.feature_extractor.video_embed_dim
        # Paper path: dataset supplies pre-extracted VideoMAE-L features with embed_dim.
        # Fallback path: adapt legacy single-stream validation tensors.
        self.projector = nn.Sequential(
            nn.Linear(2048, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.freeze = cfg.feature_extractor.freeze_video

    def forward(self, x):
        """
        Args:
            x: (B, T, D) pre-extracted video features
        Returns:
            (B, T, embed_dim)
        """
        if x.shape[-1] == self.embed_dim:
            return x
        return self.projector(x)


class CLIPFeatureExtractor(nn.Module):
    """Frozen CLIP ViT-L/14 visual feature input."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.feature_extractor.clip_embed_dim
        self.projector = nn.Sequential(
            nn.Linear(2048, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.freeze = cfg.feature_extractor.freeze_clip

    def forward_visual(self, x):
        """Extract CLIP visual features from pre-extracted embeddings."""
        if x.shape[-1] == self.embed_dim:
            return x
        return self.projector(x)

    def forward_text(self, text_embeds):
        """Project text embeddings."""
        return self.projector(text_embeds)


class MotionEncoder(nn.Module):
    """Frozen YOLO+RAFT object-motion feature input."""

    def __init__(self, cfg):
        super().__init__()
        in_dim = 2048  # accept same input dim as pre-extracted features
        out_dim = cfg.feature_extractor.motion_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        if x.shape[-1] == self.encoder[-2].out_features:
            return x
        return self.encoder(x)


class GatedFusion(nn.Module):
    """Gated fusion: g * F_v + (1-g) * concat(F_c, F_o) projected."""

    def __init__(self, dim_v, dim_c, dim_o, fused_dim):
        super().__init__()
        self.proj_v = nn.Linear(dim_v, fused_dim)
        self.proj_co = nn.Linear(dim_c + dim_o, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(dim_v + dim_c + dim_o, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, f_v, f_c, f_o):
        v = self.proj_v(f_v)
        co = self.proj_co(torch.cat([f_c, f_o], dim=-1))
        g = self.gate(torch.cat([f_v, f_c, f_o], dim=-1))
        return g * v + (1 - g) * co


class ConcatFusion(nn.Module):
    """Simple concat + MLP fusion."""

    def __init__(self, dim_v, dim_c, dim_o, fused_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_v + dim_c + dim_o, fused_dim * 2),
            nn.GELU(),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, f_v, f_c, f_o):
        return self.mlp(torch.cat([f_v, f_c, f_o], dim=-1))


class CrossAttentionFusion(nn.Module):
    """Cross-attention based multi-modal fusion."""

    def __init__(self, dim_v, dim_c, dim_o, fused_dim, num_heads=8):
        super().__init__()
        self.proj_v = nn.Linear(dim_v, fused_dim)
        self.proj_c = nn.Linear(dim_c, fused_dim)
        self.proj_o = nn.Linear(dim_o, fused_dim)
        self.cross_attn = nn.MultiheadAttention(fused_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, f_v, f_c, f_o):
        q = self.proj_v(f_v).unsqueeze(1)
        kv = torch.cat([self.proj_c(f_c).unsqueeze(1), self.proj_o(f_o).unsqueeze(1)], dim=1)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.norm(out.squeeze(1) + self.proj_v(f_v))
        return out


class PaperSumFusion(nn.Module):
    """
    Paper Eq. (1): f_t = LN(sum_m delta_m W_m F^m_t) + b_f.
    """

    def __init__(self, dim_v, dim_c, dim_o, fused_dim):
        super().__init__()
        self.proj_v = nn.Linear(dim_v, fused_dim, bias=False)
        self.proj_c = nn.Linear(dim_c, fused_dim, bias=False)
        self.proj_o = nn.Linear(dim_o, fused_dim, bias=False)
        self.norm = nn.LayerNorm(fused_dim)
        self.bias = nn.Parameter(torch.zeros(fused_dim))

    def forward(self, f_v, f_c, f_o):
        fused = self.proj_v(f_v) + self.proj_c(f_c) + self.proj_o(f_o)
        return self.norm(fused) + self.bias


class ModalityDropout(nn.Module):
    """Randomly drop a modality during training."""

    def __init__(self, p_start=0.3, p_end=0.05, anneal_fraction=0.2):
        super().__init__()
        self.p_start = p_start
        self.p_end = p_end
        self.anneal_fraction = max(float(anneal_fraction), 1e-6)
        self.progress = 0.0

    @property
    def p(self):
        ratio = min(max(self.progress / self.anneal_fraction, 0.0), 1.0)
        return self.p_start + ratio * (self.p_end - self.p_start)

    def set_progress(self, progress: float):
        self.progress = min(max(float(progress), 0.0), 1.0)

    def forward(self, *features):
        if self.training:
            mask = torch.rand(len(features), device=features[0].device) > self.p
            if mask.sum() == 0:
                mask[0] = True
            return [f * m.float() for f, m in zip(features, mask)]
        return list(features)


class MultiSourceFeatureExtractor(nn.Module):
    """
    Full multi-source feature extraction pipeline (Step 1).
    Input: three frozen feature streams: VideoMAE-L, CLIP ViT-L/14, and YOLO+RAFT.
    Output: fused multi-modal features f_t
    """

    def __init__(self, cfg):
        super().__init__()
        fe_cfg = cfg.feature_extractor
        self.video_backbone = VideoBackbone(cfg)
        self.clip_extractor = CLIPFeatureExtractor(cfg)
        self.motion_encoder = MotionEncoder(cfg)
        self.modality_dropout = ModalityDropout(
            p_start=fe_cfg.get("modality_dropout_start", 0.3),
            p_end=fe_cfg.get("modality_dropout_end", 0.05),
            anneal_fraction=fe_cfg.get("modality_dropout_anneal_fraction", 0.2),
        )

        dim_v = fe_cfg.video_embed_dim
        dim_c = fe_cfg.clip_embed_dim
        dim_o = fe_cfg.motion_embed_dim
        fused_dim = fe_cfg.fused_dim

        fusion_method = fe_cfg.fusion_method
        if fusion_method == "paper_sum":
            self.fusion = PaperSumFusion(dim_v, dim_c, dim_o, fused_dim)
        elif fusion_method == "gated":
            self.fusion = GatedFusion(dim_v, dim_c, dim_o, fused_dim)
        elif fusion_method == "cross_attn":
            self.fusion = CrossAttentionFusion(dim_v, dim_c, dim_o, fused_dim)
        else:
            self.fusion = ConcatFusion(dim_v, dim_c, dim_o, fused_dim)

        self.fused_dim = fused_dim

    def set_training_progress(self, progress: float):
        self.modality_dropout.set_progress(progress)

    def forward(self, x, return_individual: bool = False):
        """
        Args:
            x: (B, T, D) input snippet features
            return_individual: if True, also returns individual modality features
        Returns:
            f_t: (B, T, fused_dim) fused multi-modal features
            (optionally) f_v, f_c, f_o: individual modality features
        """
        if isinstance(x, dict):
            x_v = x.get("video", x.get("features"))
            x_c = x.get("clip", x_v)
            x_o = x.get("motion", x.get("object_motion", x_v))
        else:
            x_v = x_c = x_o = x

        f_v = self.video_backbone(x_v)
        f_c = self.clip_extractor.forward_visual(x_c)
        f_o = self.motion_encoder(x_o)

        f_v, f_c, f_o = self.modality_dropout(f_v, f_c, f_o)
        f_t = self.fusion(f_v, f_c, f_o)

        if return_individual:
            return f_t, f_v, f_c, f_o
        return f_t
