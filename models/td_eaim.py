"""Temporal AND-OR Interaction Machine (TD-EAIM).

Implements paper Sec. III-B exactly:

    z_t = b + sum_i w_i a_{t,i} + sum_{S in L_dict \\ L_1} beta_S g_S I_S(t)
    s_t = sigma(z_t)

Key fixes wrt earlier revision:
  * The first-order linear part (sum_i w_i a_{t,i}) is NOT gated by the
    hard-concrete g_S; the paper restricts gating to S in L_dict \\ L_1.
  * L_sparse only sums gate-activation probabilities of higher-order
    candidates (AND/OR/Temp). The first-order part is dense.
  * OR candidate pairs are selected from a region disjoint from AND
    pairs (so OR_{ij} is not the trivial linear combination of
    {a_i, a_j, AND_{ij}}). We pick AND from the highest-MI pairs and
    OR from the next band.
  * Inference uses the deterministic clipped expectation; training
    samples the stochastic logistic noise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """Hard-concrete gate (Louizos et al. 2018) with gamma=-0.1, zeta=1.1."""

    def __init__(self, num_interactions: int, temperature: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.num_interactions = num_interactions
        self.temperature = temperature
        self.gamma = -0.1
        self.zeta = 1.1
        self.epsilon = epsilon
        self.log_alpha = nn.Parameter(torch.zeros(num_interactions))

    def _expected(self):
        expected = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        return expected.clamp(0.0, 1.0)

    def _gate_prob(self):
        return torch.sigmoid(
            self.log_alpha - self.temperature * math.log(-self.gamma / self.zeta)
        )

    def forward(self, deterministic: bool = False):
        if deterministic:
            return self._expected(), self._gate_prob()
        u = torch.rand_like(self.log_alpha).clamp(self.epsilon, 1.0 - self.epsilon)
        logistic = torch.log(u) - torch.log1p(-u)
        s = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = s_bar.clamp(0.0, 1.0)
        return z, self._gate_prob()

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)


class TDEAIM(nn.Module):
    """Temporal AND-OR Interaction Machine."""

    def __init__(self, cfg):
        super().__init__()
        eaim_cfg = cfg.td_eaim
        self.num_concepts = eaim_cfg.num_concepts
        self.num_and_pairs = eaim_cfg.num_and_pairs
        self.num_or_pairs = eaim_cfg.num_or_pairs
        self.num_temporal_pairs = eaim_cfg.num_temporal_pairs
        self.temporal_offsets = list(eaim_cfg.temporal_offsets)
        self.embedding_dim = eaim_cfg.hidden_dim

        self.beta_first = nn.Parameter(torch.randn(self.num_concepts) * 0.01)

        and_i, and_j = self._initial_pairs(self.num_and_pairs, offset=0)
        or_i, or_j = self._initial_pairs(self.num_or_pairs, offset=self.num_and_pairs)
        temp_i, temp_j = self._initial_pairs(
            self.num_temporal_pairs,
            offset=self.num_and_pairs + self.num_or_pairs,
        )
        self.register_buffer("and_i", and_i)
        self.register_buffer("and_j", and_j)
        self.register_buffer("or_i", or_i)
        self.register_buffer("or_j", or_j)
        self.register_buffer("temp_i", temp_i)
        self.register_buffer("temp_j", temp_j)

        self.beta_and = nn.Parameter(torch.randn(self.num_and_pairs) * 0.01)
        self.beta_or = nn.Parameter(torch.randn(self.num_or_pairs) * 0.01)
        self.gamma_temp = nn.Parameter(
            torch.randn(self.num_temporal_pairs * len(self.temporal_offsets)) * 0.01
        )

        # Higher-order interaction count = AND + OR + Temporal*|Delta|
        self.num_higher_order = (
            self.num_and_pairs + self.num_or_pairs
            + self.num_temporal_pairs * len(self.temporal_offsets)
        )
        # Total dictionary L = L_1 union L_AND union L_OR union L_Temp
        self.total_interactions = self.num_concepts + self.num_higher_order

        # Gate is only over higher-order interactions (paper: S in L_dict \ L_1)
        self.hard_concrete_gate = HardConcreteGate(
            num_interactions=self.num_higher_order,
            temperature=eaim_cfg.get("hard_concrete_temp", 0.5),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self._deterministic = False

    def _initial_pairs(self, count: int, offset: int = 0):
        """Pick `count` pairs starting from a deterministic offset.

        Using disjoint slices of the lexicographic combination list
        guarantees that AND, OR, and Temporal start with different
        candidate sets even before the MI-based reselection runs.
        """
        pairs = torch.combinations(torch.arange(self.num_concepts), r=2)
        n = pairs.shape[0]
        if count == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty
        idx = (torch.arange(count) + offset) % n
        chosen = pairs[idx]
        return chosen[:, 0].long(), chosen[:, 1].long()

    def set_deterministic(self, deterministic: bool = True):
        self._deterministic = deterministic

    def train(self, mode: bool = True):
        super().train(mode)
        # eval -> deterministic gate by default
        self._deterministic = not mode
        return self

    def compute_and_interactions(self, A_t):
        return A_t[:, :, self.and_i] * A_t[:, :, self.and_j]

    def compute_or_interactions(self, A_t):
        a_i = A_t[:, :, self.or_i]
        a_j = A_t[:, :, self.or_j]
        return a_i + a_j - a_i * a_j

    def compute_temporal_interactions(self, A_t):
        B, T, _ = A_t.shape
        a_i = A_t[:, :, self.temp_i]
        base_j = A_t[:, :, self.temp_j]
        temporal = []
        for offset in self.temporal_offsets:
            shifted = torch.zeros_like(base_j)
            if offset < T:
                shifted[:, : T - offset, :] = base_j[:, offset:, :]
            temporal.append(a_i * shifted)
        return torch.cat(temporal, dim=-1)

    def forward(self, A_t):
        I_first = A_t
        I_and = self.compute_and_interactions(A_t)
        I_or = self.compute_or_interactions(A_t)
        I_temp = self.compute_temporal_interactions(A_t)

        # Weighted *unmasked* contributions (per S)
        weighted_first = self.beta_first.view(1, 1, -1) * I_first  # (B,T,K)
        weighted_and = self.beta_and.view(1, 1, -1) * I_and        # (B,T,P_AND)
        weighted_or = self.beta_or.view(1, 1, -1) * I_or           # (B,T,P_OR)
        weighted_temp = self.gamma_temp.view(1, 1, -1) * I_temp    # (B,T,P_T*|Delta|)
        weighted_higher = torch.cat([weighted_and, weighted_or, weighted_temp], dim=-1)

        g_S, gate_prob = self.hard_concrete_gate(deterministic=self._deterministic)
        gate = g_S.view(1, 1, -1)

        gated_higher = weighted_higher * gate

        # z_t = b + sum_i w_i a_{t,i}  +  sum_{S in L_high} beta_S g_S I_S(t)
        linear_part = weighted_first.sum(dim=-1)
        higher_part = gated_higher.sum(dim=-1)
        logits = self.bias + linear_part + higher_part
        s_t = torch.sigmoid(logits)

        # h_t and the reported active dictionary are over higher-order S only
        h_int, topm_idx = self._topm_embedding(gated_higher, g_S)

        # L_sparse: closed-form L0 surrogate over higher-order candidates only
        sparsity_loss = gate_prob.sum()

        # all_interactions kept for explanation / faithfulness pipelines
        # (keep the original [first | and | or | temp] layout; first part is
        # intentionally non-gated below get_explanation routes.)
        all_interactions = torch.cat(
            [weighted_first, weighted_and, weighted_or, weighted_temp], dim=-1
        )

        interaction_dict = {
            "I_first": I_first,
            "I_and": I_and,
            "I_or": I_or,
            "I_temp": I_temp,
            "beta_first": self.beta_first,
            "beta_and": self.beta_and,
            "beta_or": self.beta_or,
            "gamma_temp": self.gamma_temp,
            "all_interactions": all_interactions,
            "weighted_first": weighted_first,
            "weighted_higher": weighted_higher,
            "gate_mask": g_S,
            "gate_prob": gate_prob,
            "gated_interactions": gated_higher,
            "topm_idx": topm_idx,
            "h_int": h_int,
        }
        return s_t, interaction_dict, h_int, sparsity_loss

    def _topm_embedding(self, gated_higher, gate):
        """Top-m gated stack (paper Eq. h_t = [beta_S g_S I_S(t)]_{S in L_top-m})."""
        importance = gated_higher.abs().mean(dim=(0, 1))
        active = gate > 0
        if active.any():
            importance = importance * active.to(importance.dtype)
        k = min(self.embedding_dim, importance.numel())
        _, topm_idx = importance.topk(k)
        h_int = gated_higher.index_select(-1, topm_idx)
        if k < self.embedding_dim:
            pad = torch.zeros(
                *h_int.shape[:-1],
                self.embedding_dim - k,
                device=h_int.device,
                dtype=h_int.dtype,
            )
            h_int = torch.cat([h_int, pad], dim=-1)
        return h_int, topm_idx

    def get_top_interactions(self, interaction_dict, top_k: int = 5):
        """Return top-k indices into the union [first | and | or | temp]."""
        all_int = interaction_dict["all_interactions"]
        K = self.num_concepts
        gate = interaction_dict.get("gate_mask")
        importance = all_int.abs().mean(dim=(0, 1))
        if gate is not None:
            # mask higher-order with gate; first-order keeps full importance
            gate_full = torch.cat(
                [torch.ones(K, device=gate.device), (gate > 0).float()], dim=0,
            )
            importance = importance * gate_full
        topk_vals, topk_idx = importance.topk(min(top_k, importance.numel()))
        return topk_idx, topk_vals

    @torch.no_grad()
    def select_pairs_by_mutual_information(self, concept_activations, top_k=None):
        """Re-select AND / OR / Temporal candidate pairs by MI.

        AND uses the highest-MI pairs (most strongly co-occurring,
        captures conjunctive evidence). OR uses pairs ranked just below
        the AND band, so OR_{ij} carries genuine independent semantics
        rather than being the trivial 1+a_i+a_j-AND_{ij} alias.
        Temporal uses pairs that maximise the lagged co-occurrence MI
        but is left to the lex-default offset to keep this routine fast;
        users can override by reassigning self.temp_i / self.temp_j.
        """
        N, K = concept_activations.shape
        if K != self.num_concepts:
            raise ValueError(f"Expected {self.num_concepts} concepts, got {K}")

        a_bin = (concept_activations > 0.5).float()
        p_i = a_bin.mean(dim=0).clamp(1e-8, 1 - 1e-8)
        p_joint = (a_bin.unsqueeze(2) * a_bin.unsqueeze(1)).mean(dim=0)
        mi = torch.zeros(K, K, device=concept_activations.device)
        eps = 1e-8
        states = ((1, 1), (1, 0), (0, 1), (0, 0))
        for x_state, y_state in states:
            if x_state and y_state:
                p_xy = p_joint
            elif x_state and not y_state:
                p_xy = p_i[:, None] - p_joint
            elif not x_state and y_state:
                p_xy = p_i[None, :] - p_joint
            else:
                p_xy = 1 - p_i[:, None] - p_i[None, :] + p_joint
            p_x = p_i[:, None] if x_state else 1 - p_i[:, None]
            p_y = p_i[None, :] if y_state else 1 - p_i[None, :]
            p_xy = p_xy.clamp(eps, 1.0)
            mi += p_xy * torch.log(p_xy / (p_x * p_y + eps) + eps)

        tri_i, tri_j = torch.triu_indices(K, K, offset=1, device=concept_activations.device)
        vals = mi[tri_i, tri_j]
        order = torch.argsort(vals, descending=True)

        and_count = min(self.num_and_pairs, order.numel())
        and_order = order[:and_count]
        self.and_i[:and_count].copy_(tri_i[and_order].to(self.and_i.device))
        self.and_j[:and_count].copy_(tri_j[and_order].to(self.and_j.device))

        # OR pairs come from a *disjoint* later band of the MI ranking so
        # they are not aliased to AND pairs. If we run out of candidates,
        # fall back to circularly shifted AND pairs.
        or_start = and_count
        or_count = min(self.num_or_pairs, max(order.numel() - or_start, 0))
        if or_count > 0:
            or_order = order[or_start:or_start + or_count]
            self.or_i[:or_count].copy_(tri_i[or_order].to(self.or_i.device))
            self.or_j[:or_count].copy_(tri_j[or_order].to(self.or_j.device))
        if or_count < self.num_or_pairs:
            # Pad remaining slots by cycling through a permuted AND set
            extra = self.num_or_pairs - or_count
            shift = (torch.arange(extra) + 1) % and_count
            self.or_i[or_count:].copy_(self.and_j[shift].to(self.or_i.device))
            self.or_j[or_count:].copy_(self.and_i[shift].to(self.or_j.device))

        # Temporal uses the AND ranking (high MI is also a sensible prior
        # for lag-1 co-occurrence); users can override after the fit.
        temp_count = min(self.num_temporal_pairs, and_count)
        if temp_count > 0:
            self.temp_i[:temp_count].copy_(self.and_i[:temp_count])
            self.temp_j[:temp_count].copy_(self.and_j[:temp_count])
        return vals[order[:and_count]]
