"""
EIM-OWILNet Validation Script
Quick validation during development to verify pipeline integrity.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from configs.default import get_default_config
from utils.common import set_seed, count_parameters, move_to_device
from models.eim_owilnet import EIMOWILNet
from losses.composite_loss import CompositeLoss
from metrics.evaluation import VADMetrics


def validate_pipeline(cfg):
    """Validate the entire pipeline with synthetic data."""
    print("=" * 60)
    print("  EIM-OWILNet Pipeline Validation")
    print("=" * 60)

    device = cfg.device
    set_seed(cfg.seed)

    # 1. Create model
    print("\n[1/6] Creating model...")
    model = EIMOWILNet(cfg).to(device)
    model.freeze_backbones()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # 2. Create synthetic data
    print("\n[2/6] Creating synthetic data...")
    B, T, D = 4, 32, 2048
    x = torch.randn(B, T, D).to(device)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long).to(device)
    anomaly_classes = torch.tensor([0, 3, 7, 0], dtype=torch.long).to(device)
    domain_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long).to(device)

    # 3. Forward pass — standard
    print("\n[3/6] Forward pass (standard)...")
    outputs = model(x, labels=labels, anomaly_classes=anomaly_classes)
    print(f"  s_t shape: {outputs['s_t'].shape}")
    print(f"  A_t shape: {outputs['A_t'].shape}")
    print(f"  h_int shape: {outputs['h_int'].shape}")
    print(f"  s_t range: [{outputs['s_t'].min().item():.4f}, {outputs['s_t'].max().item():.4f}]")

    # 4. Forward pass — open world
    print("\n[4/6] Forward pass (open-world)...")
    outputs_ow = model(x, labels=labels, anomaly_classes=anomaly_classes,
                       compute_open_world=True)
    ow = outputs_ow["open_world"]
    print(f"  Pred classes shape: {ow['pred_class'].shape}")
    print(f"  Energy shape: {ow['energy'].shape}")
    print(f"  Residual shape: {ow['residual'].shape}")
    print(f"  Unknown flags: {ow['is_unknown'].sum().item()} / {ow['is_unknown'].numel()}")

    # 5. Forward pass — incremental + invariance
    print("\n[5/6] Forward pass (incremental + cross-dataset)...")
    outputs_inc = model(
        x, labels=labels, anomaly_classes=anomaly_classes,
        domain_ids=domain_ids, task_id=0,
        compute_incremental=True, compute_invariance=True,
    )
    if "inc_loss" in outputs_inc:
        print(f"  Incremental loss: {outputs_inc['inc_loss'].item():.4f}")
    if "inv_loss_dict" in outputs_inc:
        for k, v in outputs_inc["inv_loss_dict"].items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  Invariance {k}: {val:.4f}")

    # 6. Loss computation
    print("\n[6/6] Loss computation...")
    criterion = CompositeLoss(cfg)
    total_loss, loss_dict = criterion(outputs_ow, labels, anomaly_classes)
    print(f"  Total loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Backward pass test
    total_loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    print(f"  Parameters with gradients: {len(grad_norms)}")

    # Explanation test
    print("\n[Bonus] Explanation generation...")
    explanations = model.get_explanation(outputs_ow, top_k=5)
    for i, exp in enumerate(explanations):
        print(f"  Top-{i+1}: [{exp['type']}] {exp['description']} (value={exp['value']:.4f})")

    # Metrics test
    print("\n[Bonus] Metrics computation...")
    s_np = outputs["s_t"].detach().cpu().numpy().flatten()
    l_np = torch.cat([frame_labels for frame_labels in
                      [torch.zeros(T), torch.ones(T), torch.ones(T), torch.zeros(T)]]).numpy()
    # Ensure binary labels
    l_np = (l_np > 0.5).astype(float)
    s_np = np.clip(s_np, 0, 1)
    if len(np.unique(l_np)) >= 2:
        metrics = VADMetrics.compute_all(l_np, s_np)
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  AP: {metrics['ap']:.4f}")

    print("\n" + "=" * 60)
    print("  Pipeline validation PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet Pipeline Validation")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    cfg.fp16 = False  # Disable for validation
    cfg.concept_bottleneck.require_text_embeddings = False  # synthetic validation has no CLIP text asset

    try:
        success = validate_pipeline(cfg)
        print(f"\nValidation: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"\nValidation FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
