"""
EIM-OWILNet five-seed evaluation script.
Launches independent training runs with the paper seeds,
then aggregates results with mean +/- std.

Designed for A100 80G environment.
Can run the five paper seeds sequentially or in parallel via torch.multiprocessing.
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.multiprocessing as mp
import numpy as np
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from configs.default import get_default_config
from utils.common import set_seed, AverageMeter, save_checkpoint, count_parameters, move_to_device
from datasets.video_dataset import create_dataloaders
from models.eim_owilnet import EIMOWILNet
from losses.composite_loss import CompositeLoss
from metrics.evaluation import VADMetrics, evaluate_full


def single_run(cfg, run_id, seed, gpu_id, results_queue=None):
    """
    Execute a single training + evaluation run.
    Args:
        cfg: base config
        run_id: run index
        seed: random seed for this run
        gpu_id: GPU device id
        results_queue: multiprocessing queue for results
    """
    # Set unique seed
    set_seed(seed)
    run_cfg = deepcopy(cfg)
    run_cfg.seed = seed
    run_cfg.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    run_name = f"run_{run_id}_seed{seed}"
    log_dir = os.path.join(cfg.logging.log_dir, run_name)
    ckpt_dir = os.path.join(cfg.logging.checkpoint_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Cross-Val Run {run_id+1}/{run_cfg.cross_val.num_runs} | Seed: {seed} | GPU: {gpu_id}")
    print(f"  Log: {log_dir}")
    print(f"{'='*60}")

    # Use the canonical three-stage training implementation from train.py so
    # cross-seed reporting follows the same 50/30/20 protocol as the paper.
    run_cfg.logging.log_dir = log_dir
    run_cfg.logging.checkpoint_dir = ckpt_dir
    from train import train_standard

    test_metrics = train_standard(run_cfg)
    run_result = {
        "run_id": run_id,
        "seed": seed,
        "gpu": gpu_id,
        "best_val_auc": test_metrics["vad"]["roc_auc"],
        "test_metrics": test_metrics["vad"],
        "open_world_metrics": test_metrics.get("open_world", {}),
        "history": {},
        "n_params": None,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    if results_queue is not None:
        results_queue.put(run_result)
    return run_result

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(run_cfg, "standard")

    # Model
    model = EIMOWILNet(run_cfg).to(run_cfg.device)
    model.freeze_backbones()
    n_params = count_parameters(model)

    criterion = CompositeLoss(run_cfg)
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        weight_decay=run_cfg.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=run_cfg.fp16)

    # Scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = run_cfg.train.warmup_epochs * steps_per_epoch
    total_steps = run_cfg.train.epochs * steps_per_epoch
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_auc = 0.0
    history = {"train_loss": [], "val_auc": [], "val_ap": []}

    for epoch in range(run_cfg.train.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Run{run_id} Epoch{epoch}", leave=False)
        for batch in pbar:
            snippets, labels, frame_labels, anomaly_classes, _ = batch
            snippets = move_to_device(snippets, run_cfg.device)
            labels = labels.to(run_cfg.device)
            anomaly_classes = anomaly_classes.to(run_cfg.device)

            with torch.cuda.amp.autocast(enabled=run_cfg.fp16):
                outputs = model(snippets, labels=labels, anomaly_classes=anomaly_classes,
                               compute_open_world=True)
                total_loss, loss_dict = criterion(outputs, labels, anomaly_classes)

            optimizer.zero_grad()
            if run_cfg.fp16:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_cfg.train.grad_clip)
                optimizer.step()

            scheduler.step()
            epoch_loss += total_loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # --- Validate ---
        if (epoch + 1) % run_cfg.logging.eval_every == 0:
            model.eval()
            all_scores = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    snippets, labels, frame_labels, anomaly_classes, _ = batch
                    snippets = move_to_device(snippets, run_cfg.device)
                    with torch.cuda.amp.autocast(enabled=run_cfg.fp16):
                        outputs = model(snippets)
                    s_t = outputs["s_t"].cpu().numpy()
                    fl = frame_labels.numpy()
                    all_scores.append(s_t.flatten())
                    all_labels.append(fl.flatten())

            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            all_labels = (all_labels > 0.5).astype(float)

            if len(np.unique(all_labels)) >= 2:
                val_auc = VADMetrics.compute_roc_auc(all_labels, all_scores)
                val_ap = VADMetrics.compute_ap(all_labels, all_scores)
            else:
                val_auc = 0.0
                val_ap = 0.0

            history["val_auc"].append(val_auc)
            history["val_ap"].append(val_ap)

            print(f"  [Run{run_id} Epoch{epoch}] Loss: {avg_loss:.4f} | "
                  f"AUC: {val_auc:.4f} | AP: {val_ap:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_auc,
                }, os.path.join(ckpt_dir, "best_model.pt"))

    # --- Test on best model ---
    print(f"\n  [Run{run_id}] Testing with best model (AUC={best_auc:.4f})...")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=run_cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    all_test_scores = []
    all_test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            snippets, labels, frame_labels, anomaly_classes, _ = batch
            snippets = move_to_device(snippets, run_cfg.device)
            with torch.cuda.amp.autocast(enabled=run_cfg.fp16):
                outputs = model(snippets)
            all_test_scores.append(outputs["s_t"].cpu().numpy().flatten())
            all_test_labels.append(frame_labels.numpy().flatten())

    all_test_scores = np.concatenate(all_test_scores)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_labels = (all_test_labels > 0.5).astype(float)

    if len(np.unique(all_test_labels)) >= 2:
        test_metrics = VADMetrics.compute_all(all_test_labels, all_test_scores)
    else:
        test_metrics = {"roc_auc": 0.0, "ap": 0.0, "far_at_0_5": 0.0}

    run_result = {
        "run_id": run_id,
        "seed": seed,
        "gpu": gpu_id,
        "best_val_auc": best_auc,
        "test_metrics": test_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "n_params": n_params,
    }

    # Save run results
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(run_result, f, indent=2, default=str)

    print(f"\n  [Run{run_id} Results]")
    print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"    AP: {test_metrics['ap']:.4f}")
    print(f"    FAR@0.5: {test_metrics['far_at_0_5']:.4f}")

    if results_queue is not None:
        results_queue.put(run_result)

    return run_result


def aggregate_results(results):
    """Aggregate five-seed results with mean +/- std."""
    print(f"\n{'='*60}")
    print(f"  Aggregated Results ({len(results)} seeds)")
    print(f"{'='*60}")

    metrics_keys = results[0]["test_metrics"].keys()
    agg = {}

    for key in metrics_keys:
        values = [r["test_metrics"][key] for r in results]
        agg[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "values": values,
        }
        print(f"  {key}: {agg[key]['mean']:.4f} +/- {agg[key]['std']:.4f} "
              f"(runs: {[f'{v:.4f}' for v in values]})")

    # Per-run summary
    print(f"\n  Per-run AUC:")
    for r in results:
        print(f"    Run {r['run_id']} (seed={r['seed']}): "
              f"AUC={r['test_metrics']['roc_auc']:.4f}, "
              f"AP={r['test_metrics']['ap']:.4f}")

    return agg


def run_sequential(cfg, seeds, gpu_id):
    """Run all seed jobs sequentially on a single GPU."""
    results = []
    for run_id, seed in enumerate(seeds):
        result = single_run(cfg, run_id, seed, gpu_id)
        results.append(result)
        torch.cuda.empty_cache()
    return results


def run_parallel(cfg, seeds, gpu_ids):
    """Run models in parallel on multiple GPUs."""
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()
    processes = []

    for run_id, (seed, gpu) in enumerate(zip(seeds, gpu_ids)):
        p = ctx.Process(target=single_run, args=(cfg, run_id, seed, gpu, results_queue))
        p.start()
        processes.append(p)

    results = []
    for _ in range(len(seeds)):
        results.append(results_queue.get())

    for p in processes:
        p.join()

    results.sort(key=lambda r: r["run_id"])
    return results


def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet five-seed evaluation")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["sequential", "parallel"],
                        help="Run mode (sequential on 1 GPU or parallel across seed GPUs)")
    parser.add_argument("--output", type=str, default="cross_val_results.json")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config:
        from utils.common import load_config
        cfg = load_config(args.config)

    seeds = cfg.cross_val.seeds if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    cfg.cross_val.num_runs = len(seeds)

    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)

    print(f"[Cross-Validation] Mode: {args.mode}")
    print(f"[Cross-Validation] Seeds: {seeds}")
    print(f"[Cross-Validation] GPUs: {gpu_ids}")
    print(f"[Cross-Validation] Dataset: {cfg.data.dataset}")
    print(f"[Cross-Validation] Epochs: {cfg.train.epochs}")
    print(f"[Cross-Validation] Batch size: {cfg.train.batch_size}")
    print(f"[Cross-Validation] Device: A100 80G assumed")

    start_time = time.time()

    if args.mode == "sequential":
        gpu_id = gpu_ids[0]
        results = run_sequential(cfg, seeds, gpu_id)
    else:
        if len(gpu_ids) < len(seeds):
            print("[Warning] Parallel mode needs one GPU per seed. Falling back to sequential.")
            results = run_sequential(cfg, seeds, gpu_ids[0])
        else:
            results = run_parallel(cfg, seeds, gpu_ids)

    elapsed = time.time() - start_time
    agg = aggregate_results(results)

    # Save final results
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "config": dict(cfg),
        "elapsed_seconds": elapsed,
        "per_run_results": results,
        "aggregated": {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                       for k, v in agg.items()},
    }

    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2, default=str)

    print(f"\n[Total time] {elapsed/60:.1f} minutes")
    print(f"[Results saved] {args.output}")


if __name__ == "__main__":
    main()
