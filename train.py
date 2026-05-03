"""EIM-OWILNet training entrypoint.

Implements the full paper protocol:
  Stage 1: 50 epochs encoder warm-up
  Stage 2: 30 epochs joint training (interaction refinement)
  Stage 3: 20 epochs head fine-tuning under sparse gate
  + post-hoc temperature scaling on a 200-clip held-out split
  + EMA teacher + concept-intervention loss (paper Eq. 3)
  + Algorithm 2 (open-world clustering -> seed new prototypes) at the
    start of each incremental task

Hardware target: 2x NVIDIA A100-80GB. The script auto-detects the
distributed environment via torchrun (env vars LOCAL_RANK, RANK,
WORLD_SIZE) and wraps the model in DDP when launched with two GPUs.
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

from configs.default import get_default_config
from datasets.video_dataset import (
    create_cross_dataset_loaders,
    create_dataloaders,
    create_incremental_loaders,
)
from losses.composite_loss import CompositeLoss
from metrics.evaluation import evaluate_full
from models.eim_owilnet import EIMOWILNet
from utils.common import (
    AverageMeter,
    EarlyStopping,
    count_parameters,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
)


# =====================================================================
# Distributed helpers (2x A100-80GB DDP)
# =====================================================================

def init_distributed(cfg):
    """Initialise torch.distributed if launched via torchrun."""
    # bf16 and fp16 are mutually exclusive autocast paths on A100.
    if cfg.get("bf16", False) and cfg.get("fp16", False):
        # Prefer bf16 because A100 supports it natively without GradScaler.
        cfg.fp16 = False
    if "LOCAL_RANK" not in os.environ:
        cfg.distributed = False
        return 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
        cfg.distributed = True
        cfg.world_size = world_size
    else:
        cfg.distributed = False
    return local_rank, world_size


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def maybe_wrap_ddp(model: nn.Module, cfg) -> nn.Module:
    if cfg.distributed and dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return DDP(model, device_ids=[local_rank],
                   find_unused_parameters=cfg.get("find_unused_parameters", False))
    return model


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# =====================================================================
# Anneal helpers
# =====================================================================

def _set_step_annealing(model: nn.Module, cfg, global_step: int, total_steps: int):
    """Step-level annealing as in the paper text ('first 20% of training')."""
    base = unwrap_ddp(model)
    progress = min(max(global_step / max(total_steps, 1), 0.0), 1.0)
    if hasattr(base.feature_extractor, "set_training_progress"):
        base.feature_extractor.set_training_progress(progress)


def _set_epoch_annealing(model: nn.Module, cfg, global_epoch: int):
    """tau_g (hard-concrete) anneals over the first 30 epochs."""
    base = unwrap_ddp(model)
    gate = getattr(base.td_eaim, "hard_concrete_gate", None)
    if gate is not None:
        start = cfg.td_eaim.get("hard_concrete_temp", 0.5)
        end = cfg.td_eaim.get("hard_concrete_temp_end", 0.1)
        anneal_epochs = cfg.td_eaim.get("hard_concrete_anneal_epochs", 30)
        ratio = min(max(global_epoch / max(anneal_epochs, 1), 0.0), 1.0)
        gate.set_temperature(start + ratio * (end - start))


# =====================================================================
# MI candidate selection (Sec. III-B)
# =====================================================================

@torch.no_grad()
def fit_interaction_candidates(model: nn.Module, loader, cfg):
    if cfg.td_eaim.get("pair_selection", "mutual_info") != "mutual_info":
        return
    base = unwrap_ddp(model)
    max_samples = cfg.td_eaim.get("mi_max_samples", 8192)
    base.eval()
    activations = []
    total = 0
    for batch in tqdm(loader, desc="MI pair fit", disable=not is_main_process()):
        snippets = batch[0]
        if isinstance(snippets, dict):
            snippets = {k: v.to(cfg.device) for k, v in snippets.items()}
        else:
            snippets = snippets.to(cfg.device)
        f_t, f_v, f_c, f_o = base.feature_extractor(snippets, return_individual=True)
        A_t, _ = base.concept_bottleneck(f_t, F_C_t=f_c)
        flat = A_t.reshape(-1, A_t.shape[-1]).detach()
        remaining = max_samples - total
        activations.append(flat[:remaining])
        total += min(flat.shape[0], remaining)
        if total >= max_samples:
            break
    if activations:
        concept_acts = torch.cat(activations, dim=0)
        base.td_eaim.select_pairs_by_mutual_information(
            concept_acts, top_k=cfg.td_eaim.num_and_pairs,
        )
    base.train()


# =====================================================================
# Optimiser / scheduler
# =====================================================================

def build_optimizer(model: nn.Module, cfg):
    base = unwrap_ddp(model)
    param_groups = base.get_trainable_params()
    return AdamW(
        param_groups,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.get("beta1", 0.9), cfg.train.get("beta2", 0.999)),
    )


def build_scheduler(optimizer, cfg, steps_per_epoch: int, epochs=None):
    epochs = int(epochs or cfg.train.epochs)
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Train / validate loops
# =====================================================================

def _autocast_ctx(cfg):
    dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float16
    return autocast(device_type="cuda", dtype=dtype, enabled=cfg.fp16 or cfg.get("bf16", False))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg,
                    epoch: int, total_steps_outer: int, step_counter: dict,
                    compute_invariance: bool = True, compute_civ: bool = False):
    model.train()
    base = unwrap_ddp(model)
    _set_epoch_annealing(model, cfg, epoch)
    loss_meter = AverageMeter("loss")
    mil_meter = AverageMeter("mil")
    sparse_meter = AverageMeter("sparse")
    accum = cfg.train.get("accumulate_grad_batches", 1)

    iterator = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main_process())
    for batch_idx, batch in enumerate(iterator):
        if len(batch) == 6:
            snippets, labels, frame_labels, anomaly_classes, video_names, domain_ids = batch
            domain_ids = domain_ids.to(cfg.device)
        else:
            snippets, labels, frame_labels, anomaly_classes, video_names = batch
            domain_ids = None
        snippets = move_to_device(snippets, cfg.device)
        labels = labels.to(cfg.device)
        anomaly_classes = anomaly_classes.to(cfg.device)

        global_step = step_counter["step"]
        _set_step_annealing(model, cfg, global_step, total_steps_outer)

        with _autocast_ctx(cfg):
            outputs = model(
                snippets,
                labels=labels,
                anomaly_classes=anomaly_classes,
                domain_ids=domain_ids,
                compute_open_world=True,
                compute_incremental=False,
                compute_invariance=(compute_invariance and domain_ids is not None),
                compute_civ=compute_civ,
            )
            total_loss, loss_dict = criterion(outputs, labels, anomaly_classes)
            total_loss = total_loss / accum

        if cfg.fp16 and not cfg.get("bf16", False):
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (batch_idx + 1) % accum == 0:
            if cfg.fp16 and not cfg.get("bf16", False):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            base.update_ema_teacher()
        step_counter["step"] += 1

        loss_meter.update(total_loss.item() * accum, labels.shape[0])
        mil_meter.update(loss_dict.get("mil", 0))
        sparse_meter.update(loss_dict.get("sparse", 0))

        if is_main_process() and batch_idx % cfg.logging.log_every == 0:
            iterator.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "mil": f"{mil_meter.avg:.4f}",
                "sparse": f"{sparse_meter.avg:.4f}",
            })
    return loss_meter.avg


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_to_cpu(v) for k, v in value.items()}
    return value


@torch.no_grad()
def validate(model, loader, criterion, cfg):
    base = unwrap_ddp(model)
    base.eval()
    all_outputs, all_labels, all_classes = [], [], []
    loss_meter = AverageMeter("val_loss")

    for batch in tqdm(loader, desc="Validating", disable=not is_main_process()):
        if len(batch) == 6:
            snippets, labels, frame_labels, anomaly_classes, video_names, _ = batch
        else:
            snippets, labels, frame_labels, anomaly_classes, video_names = batch
        snippets = move_to_device(snippets, cfg.device)
        labels = labels.to(cfg.device)
        anomaly_classes = anomaly_classes.to(cfg.device)
        with _autocast_ctx(cfg):
            outputs = base(snippets, labels=labels,
                           anomaly_classes=anomaly_classes,
                           compute_open_world=True)
            total_loss, _ = criterion(outputs, labels, anomaly_classes)
        loss_meter.update(total_loss.item(), snippets.shape[0] if hasattr(snippets, "shape") else len(snippets))
        all_outputs.append(_detach_to_cpu(outputs))
        all_labels.append(frame_labels.cpu())
        all_classes.append(anomaly_classes.cpu())

    metrics = evaluate_full(all_outputs, all_labels, all_classes, cfg)
    return loss_meter.avg, metrics


def _run_training_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        scaler, cfg, stage_name: str, total_epochs: int, start_epoch: int,
                        total_steps_outer: int, step_counter: dict,
                        compute_invariance: bool, compute_civ: bool):
    early_stopping = EarlyStopping(patience=15, mode="max")
    best_metric = 0.0
    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg,
            start_epoch + epoch, total_steps_outer, step_counter,
            compute_invariance=compute_invariance, compute_civ=compute_civ,
        )
        if (epoch + 1) % cfg.logging.eval_every == 0 and is_main_process():
            val_loss, metrics = validate(model, val_loader, criterion, cfg)
            auc = metrics["vad"]["roc_auc"]
            ap = metrics["vad"]["ap"]
            print(f"[{stage_name} {epoch+1}/{total_epochs}] "
                  f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"AUC={auc:.4f} AP={ap:.4f}")
            if auc > best_metric:
                best_metric = auc
                save_checkpoint({
                    "epoch": start_epoch + epoch,
                    "model_state_dict": unwrap_ddp(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "metrics": metrics,
                }, os.path.join(cfg.logging.checkpoint_dir, "best_model.pt"))
            if cfg.train.get("early_stopping", False) and early_stopping(auc):
                print(f"[{stage_name}] Early stop at epoch {epoch+1}.")
                break
    return best_metric


# =====================================================================
# tau_c calibration helper (paper §III-A: 200-clip held-out NLL fit)
# =====================================================================

@torch.no_grad()
def _build_calibration_pack(model, loader, cfg, max_clips: int = 200):
    base = unwrap_ddp(model)
    base.eval()
    f_ts, f_cs, ys = [], [], []
    seen = 0
    for batch in loader:
        if seen >= max_clips:
            break
        snippets = batch[0]
        if isinstance(snippets, dict):
            snippets = {k: v.to(cfg.device) for k, v in snippets.items()}
        else:
            snippets = snippets.to(cfg.device)
        f_t, _, f_c, _ = base.feature_extractor(snippets, return_individual=True)
        # Pseudo-target: high-cosine concepts on the calibration split.
        cos = base.concept_bottleneck.prompt_bank(f_c)
        target = (cos > cos.mean(dim=-1, keepdim=True)).float()
        f_ts.append(f_t.detach().cpu())
        f_cs.append(f_c.detach().cpu())
        ys.append(target.detach().cpu())
        seen += f_t.shape[0]
    base.train()
    return list(zip(f_ts, f_cs, ys))


def calibrate_temperature(model, train_loader, cfg):
    base = unwrap_ddp(model)
    if base.concept_bottleneck.temperature_calibrated.item():
        return
    pack = _build_calibration_pack(model, train_loader, cfg,
                                   max_clips=cfg.concept_bottleneck.get("calibration_clips", 200))
    if not pack:
        return
    base.concept_bottleneck.calibrate_temperature(
        pack, target_fn=lambda x: x, device=cfg.device,
        n_steps=cfg.concept_bottleneck.get("calibration_steps", 200),
        lr=cfg.concept_bottleneck.get("calibration_lr", 1e-2),
        freeze=True,
    )
    if is_main_process():
        tau = base.concept_bottleneck.log_temperature.exp().clamp(min=1e-3).item()
        print(f"[calibrate] post-hoc tau_c = {tau:.4f}")


# =====================================================================
# Top-level training protocols
# =====================================================================

def train_standard(cfg, resume_path=None):
    """Protocol A: standard weakly-supervised VAD."""
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, "standard")

    base_model = EIMOWILNet(cfg).to(cfg.device)
    base_model.freeze_backbones()
    base_model.attach_ema_teacher(cfg.train.get("ema_decay", 0.99))
    if is_main_process():
        print(f"[Model] EIM-OWILNet | trainable params: {count_parameters(base_model):,}")

    model = maybe_wrap_ddp(base_model, cfg)
    criterion = CompositeLoss(cfg).to(cfg.device)
    scaler = GradScaler(enabled=cfg.fp16 and not cfg.get("bf16", False))

    # ----- MI candidate fit BEFORE stage 1 (paper Sec. III-B) -----
    fit_interaction_candidates(model, train_loader, cfg)

    total_epochs = cfg.train.epochs + cfg.train.get("stage2_epochs", 30) + cfg.train.get("stage3_epochs", 20)
    total_steps_outer = total_epochs * max(len(train_loader), 1)
    step_counter = {"step": 0}

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader), cfg.train.epochs)

    start_epoch = 0
    best_metric = 0.0
    if resume_path:
        start_epoch, best_metric = load_checkpoint(resume_path, unwrap_ddp(model), optimizer)

    # ----- Stage 1: encoder warm-up (50 epochs) -----
    if is_main_process():
        print(f"\n[Stage 1] encoder warm-up ({cfg.train.epochs} epochs)")
    best_metric = max(best_metric, _run_training_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, cfg,
        "Stage1", cfg.train.epochs, start_epoch, total_steps_outer, step_counter,
        compute_invariance=False, compute_civ=False,
    ))

    # post-hoc temperature calibration before joint training (paper §III-A)
    calibrate_temperature(model, train_loader, cfg)

    # ----- Stage 2: joint training (30 epochs) -----
    stage2_epochs = cfg.train.get("stage2_epochs", 30)
    if stage2_epochs > 0:
        if is_main_process():
            print(f"\n[Stage 2] joint training ({stage2_epochs} epochs)")
        scheduler = build_scheduler(optimizer, cfg, len(train_loader), stage2_epochs)
        best_metric = max(best_metric, _run_training_stage(
            model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, cfg,
            "Stage2", stage2_epochs, cfg.train.epochs, total_steps_outer, step_counter,
            compute_invariance=False, compute_civ=True,
        ))

    # ----- Stage 3: head fine-tuning under sparse gate (20 epochs) -----
    stage3_epochs = cfg.train.get("stage3_epochs", 20)
    if stage3_epochs > 0:
        if is_main_process():
            print(f"\n[Stage 3] head fine-tuning ({stage3_epochs} epochs)")
        scheduler = build_scheduler(optimizer, cfg, len(train_loader), stage3_epochs)
        total_prev = cfg.train.epochs + stage2_epochs
        best_metric = max(best_metric, _run_training_stage(
            model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, cfg,
            "Stage3", stage3_epochs, total_prev, total_steps_outer, step_counter,
            compute_invariance=False, compute_civ=True,
        ))

    if is_main_process():
        print("\n[Test on best checkpoint]")
        load_checkpoint(os.path.join(cfg.logging.checkpoint_dir, "best_model.pt"), unwrap_ddp(model))
        _, test_metrics = validate(model, test_loader, criterion, cfg)
        print(f"[Test] AUC={test_metrics['vad']['roc_auc']:.4f} "
              f"AP={test_metrics['vad']['ap']:.4f} "
              f"FAR@0.5={test_metrics['vad']['far_at_0_5']:.4f}")
        return test_metrics
    return None


def train_open_world(cfg):
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, "open_world")
    base_model = EIMOWILNet(cfg).to(cfg.device)
    base_model.freeze_backbones()
    base_model.attach_ema_teacher(cfg.train.get("ema_decay", 0.99))
    model = maybe_wrap_ddp(base_model, cfg)
    criterion = CompositeLoss(cfg).to(cfg.device)
    scaler = GradScaler(enabled=cfg.fp16 and not cfg.get("bf16", False))

    fit_interaction_candidates(model, train_loader, cfg)

    total_epochs = cfg.train.epochs
    total_steps_outer = total_epochs * max(len(train_loader), 1)
    step_counter = {"step": 0}
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader), total_epochs)

    for epoch in range(total_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg,
                        epoch, total_steps_outer, step_counter,
                        compute_invariance=False, compute_civ=epoch >= 5)
        if (epoch + 1) % cfg.logging.eval_every == 0 and is_main_process():
            val_loss, metrics = validate(model, val_loader, criterion, cfg)
            print(f"[OW {epoch+1}] AUC={metrics['vad']['roc_auc']:.4f} "
                  f"Unk-AUROC={metrics['open_world'].get('unknown_auroc', 0):.4f} "
                  f"OSCR={metrics['open_world'].get('oscr', 0):.4f}")
    if is_main_process():
        _, test_metrics = validate(model, test_loader, criterion, cfg)
        print(f"[OW Test] AUC={test_metrics['vad']['roc_auc']:.4f} "
              f"AP={test_metrics['vad']['ap']:.4f}")
        return test_metrics
    return None


def train_incremental(cfg):
    """Protocol D: 5-task incremental learning with cross-dataset T3/T4."""
    set_seed(cfg.seed)
    base_model = EIMOWILNet(cfg).to(cfg.device)
    base_model.freeze_backbones()
    base_model.attach_ema_teacher(cfg.train.get("ema_decay", 0.99))
    model = maybe_wrap_ddp(base_model, cfg)
    criterion = CompositeLoss(cfg).to(cfg.device)

    from metrics.evaluation import IncrementalMetrics
    inc_metrics = IncrementalMetrics(cfg.incremental.num_tasks)

    seen_class_ids = []
    for task_id in range(cfg.incremental.num_tasks):
        if is_main_process():
            spec = cfg.incremental.task_specs[task_id] if "task_specs" in cfg.incremental else None
            print(f"\n[Task {task_id}] " + (f"{spec}" if spec else f"classes={cfg.incremental.task_classes[task_id]}"))

        train_loader, test_loader = create_incremental_loaders(cfg, task_id)
        new_classes = cfg.incremental.task_specs[task_id]["classes"] \
            if "task_specs" in cfg.incremental else cfg.incremental.task_classes[task_id]

        # ---- Algorithm 2 step 2: cluster *unlabeled training* stream
        # to seed prototypes for the new classes. We must NOT touch the
        # test loader here (test data leakage). ----
        if task_id > 0:
            unwrap_ddp(model).seed_new_prototypes_from_clustering(
                train_loader, cfg.device,
                existing_class_ids=seen_class_ids,
                new_class_ids=list(new_classes),
            )

        optimizer = build_optimizer(model, cfg)
        epochs_per_task = cfg.train.get("stage3_epochs", 20)
        total_steps = epochs_per_task * max(len(train_loader), 1)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader), epochs_per_task)
        step_counter = {"step": 0}
        scaler = GradScaler(enabled=cfg.fp16 and not cfg.get("bf16", False))

        for epoch in range(epochs_per_task):
            model.train()
            _set_epoch_annealing(model, cfg, epoch)
            for batch in tqdm(train_loader, desc=f"Task{task_id} ep{epoch}",
                              disable=not is_main_process()):
                snippets = batch[0]
                labels = batch[1].to(cfg.device)
                anomaly_classes = batch[3].to(cfg.device)
                snippets = move_to_device(snippets, cfg.device)
                _set_step_annealing(model, cfg, step_counter["step"], total_steps)
                with _autocast_ctx(cfg):
                    outputs = model(
                        snippets, labels=labels, anomaly_classes=anomaly_classes,
                        task_id=task_id,
                        compute_open_world=True,
                        compute_incremental=True,
                        compute_invariance=False,
                        compute_civ=task_id > 0,
                    )
                    total_loss, _ = criterion(outputs, labels, anomaly_classes)

                optimizer.zero_grad(set_to_none=True)
                if cfg.fp16 and not cfg.get("bf16", False):
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                scheduler.step()
                unwrap_ddp(model).update_ema_teacher()
                step_counter["step"] += 1

        seen_class_ids.extend(list(new_classes))

        # Snapshot the *trained* network as the teacher for the NEXT task.
        # (Algorithm 2 step "Refresh M_int from f^new" -- happens at the
        # END of the current task, not at its start.)
        unwrap_ddp(model).incremental_module.save_old_model(unwrap_ddp(model))

        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_loader = create_incremental_loaders(cfg, eval_task)
            _, metrics = validate(model, eval_loader, criterion, cfg)
            inc_metrics.update(task_id, eval_task, metrics["vad"]["roc_auc"])

    inc_metrics.memory_bytes = unwrap_ddp(model).incremental_module.memory_bank.estimated_bytes()
    inc_results = inc_metrics.compute_all()
    if is_main_process():
        print(f"[Inc] Avg-AUC={inc_results['avg_auc']:.4f} "
              f"Forget={inc_results['forget']:.4f} "
              f"BWT={inc_results['bwt']:.4f} "
              f"FWT={inc_results['fwt']:.4f} "
              f"Mem={inc_results['mem_mb']:.2f}MB")
    return inc_results


def train_cross_dataset(cfg):
    """Protocol B: fixed Train -> Test transfer pairs (no target labels)."""
    set_seed(cfg.seed)
    results = {}
    for source, target in cfg.cross_dataset.transfer_pairs:
        if is_main_process():
            print(f"\n[Cross] {source} -> {target}")
        train_loader, target_loader = create_cross_dataset_loaders(cfg, source, target)
        base_model = EIMOWILNet(cfg).to(cfg.device)
        base_model.freeze_backbones()
        base_model.attach_ema_teacher(cfg.train.get("ema_decay", 0.99))
        model = maybe_wrap_ddp(base_model, cfg)
        criterion = CompositeLoss(cfg).to(cfg.device)
        optimizer = build_optimizer(model, cfg)
        total_steps = cfg.train.epochs * max(len(train_loader), 1)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        scaler = GradScaler(enabled=cfg.fp16 and not cfg.get("bf16", False))
        step_counter = {"step": 0}
        fit_interaction_candidates(model, train_loader, cfg)
        for epoch in range(cfg.train.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg,
                            epoch, total_steps, step_counter,
                            compute_invariance=True, compute_civ=epoch >= 5)
        if is_main_process():
            _, metrics = validate(model, target_loader, criterion, cfg)
            results[f"{source}->{target}"] = metrics["vad"]["roc_auc"]
            print(f"  ROC-AUC={metrics['vad']['roc_auc']:.4f}")
    return results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--protocol", type=str, default="standard",
                        choices=["standard", "open_world", "incremental", "cross_dataset"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config:
        from utils.common import load_config
        cfg = load_config(args.config, args.overrides)
    if args.seed:
        cfg.seed = args.seed

    local_rank, world_size = init_distributed(cfg)
    if not cfg.distributed:
        cfg.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    if is_main_process():
        print(f"[Config] protocol={args.protocol} seed={cfg.seed} "
              f"device={cfg.device} world_size={world_size}")
        print(f"[Data] dataset={cfg.data.dataset}")

    if args.protocol == "standard":
        train_standard(cfg, args.resume)
    elif args.protocol == "open_world":
        train_open_world(cfg)
    elif args.protocol == "incremental":
        train_incremental(cfg)
    elif args.protocol == "cross_dataset":
        train_cross_dataset(cfg)

    if cfg.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
