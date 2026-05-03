"""
EIM-OWILNet Testing Script
Full evaluation across all protocols with all metrics from the paper.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from configs.default import get_default_config
from utils.common import set_seed, load_checkpoint, load_config, move_to_device
from datasets.video_dataset import create_dataloaders, create_cross_dataset_loaders
from models.eim_owilnet import EIMOWILNet
from losses.composite_loss import CompositeLoss
from metrics.evaluation import (
    VADMetrics, OpenWorldMetrics, IncrementalMetrics, ExplanationMetrics, evaluate_full,
)


@torch.no_grad()
def test_standard(model, test_loader, cfg):
    """Protocol A: Standard VAD evaluation."""
    model.eval()
    all_scores = []
    all_labels = []
    all_classes = []

    for batch in tqdm(test_loader, desc="Testing (Standard)"):
        snippets, labels, frame_labels, anomaly_classes, video_names = batch
        snippets = move_to_device(snippets, cfg.device)

        outputs = model(snippets, compute_open_world=True)

        B, T = outputs["s_t"].shape
        for b in range(B):
            all_scores.append(outputs["s_t"][b].cpu().numpy())
            all_labels.append(frame_labels[b].cpu().numpy())
            all_classes.append(anomaly_classes[b].item())

    all_scores_flat = np.concatenate(all_scores)
    all_labels_flat = np.concatenate(all_labels)

    metrics = VADMetrics.compute_all(all_labels_flat, all_scores_flat)
    return metrics


@torch.no_grad()
def test_open_world(model, test_loader, cfg):
    """Protocol C: Open-world evaluation."""
    model.eval()
    all_scores = []
    all_labels = []
    all_classes = []
    all_pred_classes = []
    all_confidences = []
    all_is_unknown = []
    all_unknown_scores = []
    unknown_features = []
    unknown_true_classes = []

    for batch in tqdm(test_loader, desc="Testing (Open-World)"):
        snippets, labels, frame_labels, anomaly_classes, video_names = batch
        snippets = move_to_device(snippets, cfg.device)

        outputs = model(snippets, labels=labels.to(cfg.device),
                       anomaly_classes=anomaly_classes.to(cfg.device),
                       compute_open_world=True)

        s_t = outputs["s_t"]
        ow = outputs["open_world"]
        pred_unknown_mask = ow["is_unknown"]
        if pred_unknown_mask.any():
            unknown_features.append(outputs["h_int"][pred_unknown_mask].detach().cpu())
            cls_grid = anomaly_classes.to(cfg.device).view(-1, 1).expand_as(pred_unknown_mask)
            unknown_true_classes.append(cls_grid[pred_unknown_mask].detach().cpu().numpy())

        B = s_t.shape[0]
        for b in range(B):
            max_score, max_idx = s_t[b].max(dim=0)
            all_scores.append(max_score.cpu().item())
            all_labels.append(labels[b].item())

            cls = anomaly_classes[b].item()
            all_classes.append(cls)

            max_conf, pred_cls = ow["max_confidence"][b].max(dim=0)
            all_pred_classes.append(ow["pred_class"][b, max_idx].item())
            all_confidences.append(max_conf.cpu().item())
            all_is_unknown.append(ow["is_unknown"][b, max_idx].item())
            all_unknown_scores.append(ow["unknown_score"][b, max_idx].cpu().item())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_classes = np.array(all_classes)
    all_pred_classes = np.array(all_pred_classes)
    all_confidences = np.array(all_confidences)
    all_unknown_scores = np.array(all_unknown_scores)
    all_is_unknown = np.array(all_is_unknown).astype(bool)
    cluster_labels = None
    if unknown_features:
        features = torch.cat(unknown_features, dim=0).numpy()
        cluster_labels = model.open_world_head.cluster_unknown_features(features)

    vad_metrics = VADMetrics.compute_all((all_labels > 0).astype(float), all_scores)
    ow_metrics = OpenWorldMetrics.compute_all(
        all_classes,
        all_pred_classes,
        all_confidences,
        cfg.data.seen_classes,
        unknown_score=all_unknown_scores,
        pred_is_unknown=all_is_unknown,
        cluster_labels=cluster_labels,
        cluster_true_labels=np.concatenate(unknown_true_classes) if unknown_true_classes else None,
        known_binary_labels=(all_labels > 0).astype(float),
        known_scores=all_scores,
        known_auc_mask=all_labels != 2,
    )

    return {"vad": vad_metrics, "open_world": ow_metrics}


@torch.no_grad()
def test_explanation(model, test_loader, cfg, k=5):
    """Explanation quality evaluation."""
    model.eval()
    metric_lists = defaultdict(list)

    for batch in tqdm(test_loader, desc="Testing (Explanation)"):
        snippets, labels, frame_labels, anomaly_classes, video_names = batch
        snippets = move_to_device(snippets, cfg.device)

        curves = model.compute_interaction_perturbation_scores(snippets, labels=labels.to(cfg.device), top_k=k)
        curves_np = {
            name: [v.cpu().numpy() for v in value] if isinstance(value, list) else value.cpu().numpy()
            for name, value in curves.items()
            if name in ["full", "without_top", "only_top", "deletion", "insertion"]
        }
        metrics = ExplanationMetrics.compute_from_curves(curves_np)
        for name, value in metrics.items():
            metric_lists[name].append(value)

    return {name: float(np.mean(values)) for name, values in metric_lists.items()}


def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet Testing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--protocol", type=str, default="standard",
                        choices=["standard", "open_world", "explanation", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="test_results.json")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config:
        cfg = load_config(args.config)
    cfg.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    model = EIMOWILNet(cfg).to(cfg.device)
    load_checkpoint(args.checkpoint, model)
    print(f"[Model] Loaded from {args.checkpoint}")

    _, _, test_loader = create_dataloaders(cfg, args.protocol if args.protocol != "all" else "standard")

    results = {}
    if args.protocol in ["standard", "all"]:
        results["standard"] = test_standard(model, test_loader, cfg)
        print(f"\n[Standard VAD]")
        for k, v in results["standard"].items():
            print(f"  {k}: {v:.4f}")

    if args.protocol in ["open_world", "all"]:
        results["open_world"] = test_open_world(model, test_loader, cfg)
        print(f"\n[Open-World]")
        for group, metrics in results["open_world"].items():
            print(f"  {group}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

    if args.protocol in ["explanation", "all"]:
        results["explanation"] = test_explanation(model, test_loader, cfg)
        print(f"\n[Explanation]")
        for k, v in results["explanation"].items():
            print(f"  {k}: {v:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
