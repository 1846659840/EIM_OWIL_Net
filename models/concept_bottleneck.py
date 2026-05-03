"""Concept Bottleneck Encoder (paper Sec. III-A).

  a_{t,k} = sigmoid( MLP([f_t, rho_{t,k}]) / tau_c ),
  rho_{t,k} = (1/5) sum_{j=1..5} cos(F^C_t, T_k^{(j)})

Fixes wrt earlier revision:
  * tau_c can now be either (a) a free end-to-end parameter (legacy
    behaviour) or (b) a post-hoc calibrated scalar fitted on a small
    held-out split via NLL minimisation, as the paper requires
    ("temperature scaling on a 200-clip held-out split").
  * `calibrate_temperature` exposes the post-hoc fitting procedure;
    once called, the parameter is frozen so further training does not
    drift the calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPTextPromptBank(nn.Module):
    """Concept prompt bank with five frozen CLIP text embeddings per concept."""

    def __init__(self, num_concepts: int, text_emb_dim: int, n_prompts: int = 5):
        super().__init__()
        self.num_concepts = num_concepts
        self.n_prompts = n_prompts
        self.register_buffer(
            "text_embeddings",
            torch.randn(num_concepts, n_prompts, text_emb_dim) * 0.02,
        )
        self.register_buffer("text_embeddings_set", torch.tensor(False))

    def set_text_embeddings(self, embeddings: torch.Tensor):
        if embeddings.shape[:2] != (self.num_concepts, self.n_prompts):
            raise ValueError(
                f"Expected text embeddings shaped "
                f"({self.num_concepts}, {self.n_prompts}, D), got {tuple(embeddings.shape)}"
            )
        self.text_embeddings = embeddings
        self.text_embeddings_set = torch.tensor(True, device=embeddings.device)

    def forward(self, F_C_t):
        F_C_norm = F.normalize(F_C_t, dim=-1)
        T_norm = F.normalize(self.text_embeddings, dim=-1)
        cos_all = torch.einsum("btd,kpd->btkp", F_C_norm, T_norm)
        return cos_all.mean(dim=-1)


class ConceptBottleneckEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cb_cfg = cfg.concept_bottleneck
        fe_cfg = cfg.feature_extractor
        self.num_concepts = cb_cfg.num_concepts
        input_dim = fe_cfg.fused_dim
        clip_dim = fe_cfg.clip_embed_dim

        self.prompt_bank = CLIPTextPromptBank(
            num_concepts=self.num_concepts,
            text_emb_dim=clip_dim,
            n_prompts=5,
        )

        self.activation_mlp = nn.Sequential(
            nn.Linear(input_dim + 1, cb_cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cb_cfg.dropout),
            nn.Linear(cb_cfg.hidden_dim, 1),
        )
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.register_buffer("temperature_calibrated", torch.tensor(False))

    def _logits(self, f_t, F_C_t=None):
        if F_C_t is not None and self.prompt_bank.text_embeddings_set.item():
            cos_sim = self.prompt_bank(F_C_t)
        else:
            cos_sim = torch.zeros(
                f_t.shape[0], f_t.shape[1], self.num_concepts,
                device=f_t.device, dtype=f_t.dtype,
            )
        B, T, D = f_t.shape
        f_rep = f_t.unsqueeze(2).expand(B, T, self.num_concepts, D)
        mlp_input = torch.cat([f_rep, cos_sim.unsqueeze(-1)], dim=-1)
        logits = self.activation_mlp(mlp_input).squeeze(-1)
        return logits, cos_sim

    def forward(self, f_t, F_C_t=None):
        logits, cos_sim = self._logits(f_t, F_C_t)
        tau_c = self.log_temperature.exp().clamp(min=1e-3)
        A_t = torch.sigmoid(logits / tau_c)
        return A_t, cos_sim

    @torch.no_grad()
    def set_temperature(self, tau_c: float, freeze: bool = False):
        value = torch.tensor(float(tau_c), device=self.log_temperature.device)
        self.log_temperature.data.copy_(value.clamp(min=1e-3).log())
        if freeze:
            self.log_temperature.requires_grad_(False)
            self.temperature_calibrated.fill_(True)

    def calibrate_temperature(self, calibration_loader, target_fn,
                              device: str = "cpu", n_steps: int = 200,
                              lr: float = 1e-2, freeze: bool = True):
        """Fit tau_c by minimising NLL on a held-out 200-clip split.

        Args:
            calibration_loader: yields ``(f_t, F_C_t, y_target)`` triples
                where ``y_target`` is a binary ``(B, T, K)`` tensor of
                concept-presence supervision (or soft probabilities).
            target_fn: callable that turns the loader item into the
                ``(f_t, F_C_t, y_target)`` triple if the loader returns
                a different schema.
            device: device on which to run the fit.
            n_steps: number of LBFGS-style gradient steps.
            lr: learning rate of the temperature optimiser.
            freeze: if True, freeze ``log_temperature`` after fitting.
        """
        prev_grad = self.log_temperature.requires_grad
        self.log_temperature.requires_grad_(True)
        opt = torch.optim.Adam([self.log_temperature], lr=lr)
        seen_logits = []
        seen_targets = []
        for batch in calibration_loader:
            triple = target_fn(batch) if target_fn is not None else batch
            f_t, F_C_t, y = triple
            f_t = f_t.to(device)
            F_C_t = F_C_t.to(device) if F_C_t is not None else None
            y = y.to(device).float().clamp(0.0, 1.0)
            with torch.no_grad():
                logits, _ = self._logits(f_t, F_C_t)
            seen_logits.append(logits.detach())
            seen_targets.append(y)
        if not seen_logits:
            self.log_temperature.requires_grad_(prev_grad)
            return
        logits_all = torch.cat(seen_logits, dim=0)
        y_all = torch.cat(seen_targets, dim=0)
        for _ in range(n_steps):
            opt.zero_grad()
            tau_c = self.log_temperature.exp().clamp(min=1e-3)
            probs = torch.sigmoid(logits_all / tau_c).clamp(1e-6, 1 - 1e-6)
            nll = -(y_all * probs.log() + (1 - y_all) * (1 - probs).log()).mean()
            nll.backward()
            opt.step()
        if freeze:
            self.log_temperature.requires_grad_(False)
            self.temperature_calibrated.fill_(True)
        else:
            self.log_temperature.requires_grad_(prev_grad)

    def get_concept_names(self):
        from datasets.video_dataset import CONCEPT_PROMPTS
        concepts_per_type = self.num_concepts // 4
        names = {}
        for i, ctype in enumerate(["action", "object", "scene", "dynamic"]):
            prompts = CONCEPT_PROMPTS.get(ctype, [])
            for j in range(concepts_per_type):
                idx = i * concepts_per_type + j
                names[idx] = prompts[j] if j < len(prompts) else f"{ctype}_{j}"
        return names

    def intervene_concepts(self, A_t: torch.Tensor, k_indices: torch.Tensor,
                           values: torch.Tensor) -> torch.Tensor:
        """Counterfactual edit of A_t at coordinate (b, t, k) for L_civ.

        Args:
            A_t: (B, T, K) concept activations.
            k_indices: (B,) which concept to clamp per video.
            values: (B,) target values in {0., 1.}.
        Returns:
            A_t with the chosen coordinate replaced by ``values``.
        """
        out = A_t.clone()
        B, T, K = A_t.shape
        b_idx = torch.arange(B, device=A_t.device)
        out[b_idx, :, k_indices] = values.view(B, 1).expand(B, T)
        return out
