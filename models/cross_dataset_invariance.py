"""Cross-Dataset Interaction Invariance head (paper Sec. III-F).

Two losses:
  L_inv     = sum_S Var_d ( beta_S^{(d)} )           — interaction
                                                        weight invariance
  L_domain  = CE( d, D_psi(GRL(h_t)) )               — adversarial
                                                        domain head

Fix wrt earlier revision:
  * L_inv now retains the *current* domain's beta as a differentiable
    entry of the variance stack so gradients can flow back into the
    interaction weights. The previous revision pushed only detached
    cached weights into the stack and produced a constant zero-grad
    invariance loss.
  * `update_domain_weights` is invoked AFTER `compute_invariance_loss`
    in the model forward (see eim_owilnet.py) so the cached entries
    represent past domains, not the current one being trained on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_domains: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, h_int):
        if h_int.dim() == 3:
            h_int = h_int.mean(dim=1)
        return self.classifier(h_int)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)


class CrossDatasetInvariance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cd_cfg = cfg.cross_dataset
        eaim_cfg = cfg.td_eaim
        self.lambda_inv = cd_cfg.lambda_inv
        self.lambda_domain = cd_cfg.lambda_domain
        self.num_domains = cd_cfg.num_domains
        proto_dim = eaim_cfg.hidden_dim

        self.grl = GradientReversalLayer(lambda_grl=1.0)
        self.domain_classifier = DomainClassifier(
            proto_dim, cd_cfg.domain_classifier_hidden, self.num_domains,
        )

        # Per-domain cache of (detached) interaction weights from past batches.
        # Keys are int domain ids.
        self.domain_weights = {}

    def forward(self, h_int, domain_id=None):
        loss_dict = {}
        if domain_id is not None:
            reversed_h = self.grl(h_int)
            domain_logits = self.domain_classifier(reversed_h)
            domain_loss = F.cross_entropy(domain_logits, domain_id)
            loss_dict["domain_loss"] = domain_loss
        return loss_dict

    def compute_invariance_loss(self, interaction_dict, current_domain_ids=None):
        """L_inv = sum_S Var_d( beta_S^{(d)} ) with current beta as a live entry.

        We need at least one cached domain entry to compare against. If
        only the current domain is available, the loss is zero (no
        variance to regularise yet).
        """
        device = interaction_dict["beta_and"].device
        inv_loss = torch.tensor(0.0, device=device)
        keys = ("beta_and", "beta_or", "gamma_temp")

        if current_domain_ids is not None:
            current_unique = set(int(d.item()) for d in current_domain_ids.unique())
        else:
            current_unique = set()

        for key in keys:
            if key not in interaction_dict:
                continue
            cur = interaction_dict[key]                   # differentiable
            past = []
            min_len = cur.shape[0]
            for d_id, dw in self.domain_weights.items():
                if d_id in current_unique:
                    continue
                if key not in dw:
                    continue
                past.append(dw[key].to(device))
                min_len = min(min_len, past[-1].shape[0])
            if len(past) == 0:
                continue
            stacked = [cur[:min_len]] + [w[:min_len] for w in past]
            stacked = torch.stack(stacked, dim=0)
            # Var across domain axis, sum over S
            inv_loss = inv_loss + stacked.var(dim=0, unbiased=False).sum()
        return inv_loss

    def update_domain_weights(self, domain_id: int, interaction_dict):
        """Cache a detached snapshot for past-domain reference."""
        self.domain_weights[int(domain_id)] = {
            "beta_and": interaction_dict["beta_and"].detach().cpu(),
            "beta_or": interaction_dict["beta_or"].detach().cpu(),
            "gamma_temp": interaction_dict["gamma_temp"].detach().cpu(),
        }
