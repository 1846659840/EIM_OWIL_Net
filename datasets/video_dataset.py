"""Unified video dataset loader for UCF-Crime / XD-Violence / ShanghaiTech / UBnormal.

Implements the dataset protocol described in:
  - Paper Sec. IV-A:
      UCF-Crime    1900 videos, 13 anomaly classes, 1610/290 split
      XD-Violence  4754 multi-source clips, 6 violence classes, 3954/800 split
      ShanghaiTech 437 weakly-labeled, binary anomaly protocol
      UBnormal     543 synthetic videos, seen/unseen open-set splits
  - Appendix Table I (open-world Seen-8 / Unseen-5; incremental T0..T4)

Each sample returns:
    (snippet_features_or_dict, video_label, frame_labels,
     anomaly_class, video_name)

When `task_specs` is supplied (incremental protocol with cross-dataset
T_3, T_4), the loader switches the underlying dataset_name accordingly.
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)
from torch.utils.data.distributed import DistributedSampler


# ============================================================
# 96 concept vocabulary (paper Appendix B.1).
# ============================================================
CONCEPT_PROMPTS = {
    "action": [
        "fighting", "running", "falling", "stealing", "aiming",
        "kicking", "punching", "pushing", "grabbing", "escaping",
        "sneaking", "carrying", "threatening", "hugging", "walking",
        "sitting", "standing", "kneeling", "climbing", "crawling",
        "jumping", "dancing", "screaming", "panic-running",
    ],
    "object": [
        "car", "fire", "smoke", "weapon", "knife", "gun", "bag",
        "mask", "fence", "glass", "bottle", "stick", "helmet",
        "blood", "debris", "vehicle-truck", "motorcycle", "bicycle",
        "baton", "baggage", "electronic-device", "document", "ladder", "rope",
    ],
    "scene": [
        "road", "store", "subway", "campus", "office", "parking",
        "alley", "plaza", "station", "gym", "restaurant", "mall",
        "bank", "gas-station", "bridge", "intersection", "sidewalk",
        "escalator", "lobby", "hallway", "classroom", "courtyard",
        "factory", "garage",
    ],
    "dynamic": [
        "collision", "sudden-stop", "fast-motion", "crowd-escape",
        "group-formation", "chase", "fall-impact", "scatter", "gather",
        "freeze", "sway", "struggle", "pursuit", "retreat", "bounce",
        "swing", "throw", "catch", "drop", "slip", "drag",
        "push-down", "lift-up", "spin",
    ],
}

# 5 CLIP text prompt templates per concept.
CONCEPT_PROMPT_TEMPLATES = [
    "a photo of {}",
    "a video of {}",
    "a scene showing {}",
    "an example of {}",
    "{} in a surveillance video",
]


# Canonical anomaly class indices used across the four datasets.
# Classes are aligned with the alphabetical UCF-Crime ordering used by
# the original benchmark; cross-dataset experiments map their native
# labels into this index space via *_CLASS_TO_GLOBAL.
UCF_CLASS_NAMES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism", "Normal",
]
UCF_CLASS_TO_GLOBAL = {name.lower(): i for i, name in enumerate(UCF_CLASS_NAMES)}

XD_CLASS_NAMES = ["Fighting", "Shooting", "Riot", "Abuse", "CarAccident", "Explosion"]
XD_CLASS_TO_GLOBAL = {
    "fighting": 6, "shooting": 9, "riot": 3, "abuse": 0,
    "caraccident": 7, "explosion": 5,
}

SHT_CLASS_NAMES = ["AbnormalEvent"]
# ShanghaiTech is binary; we use a dataset-local anomaly index of 0
# (was previously 13, which collided with UCF "Normal" -> task filter
# would silently drop every SHT anomaly when task_classes=[0]).
SHT_CLASS_TO_LOCAL = {"abnormalevent": 0}

UB_CLASS_NAMES = [
    "Fighting", "Shooting", "Robbery", "Assault", "Vandalism",
    "Stealing", "Sleeping", "Lying", "Falling", "Smoke", "Fire", "Crash", "Running",
]
# UBnormal uses a dataset-local anomaly index 0..12 for the 13 types.
UB_CLASS_TO_LOCAL = {name.lower(): i for i, name in enumerate(UB_CLASS_NAMES)}


# ============================================================
# Snippet helpers
# ============================================================

def _snippet_length(snippets):
    if isinstance(snippets, dict):
        first = next(iter(snippets.values()))
        return first.shape[0]
    return snippets.shape[0]


def _slice_snippets(snippets, start, end):
    if isinstance(snippets, dict):
        return {k: v[start:end] for k, v in snippets.items()}
    return snippets[start:end]


def _pad_snippets(snippets, pad_len):
    if pad_len <= 0:
        return snippets
    if isinstance(snippets, dict):
        return {k: torch.cat([v, v[-1:].repeat(pad_len, 1)], dim=0) for k, v in snippets.items()}
    return torch.cat([snippets, snippets[-1:].repeat(pad_len, 1)], dim=0)


def _stack_snippets(snippets_list):
    if isinstance(snippets_list[0], dict):
        keys = snippets_list[0].keys()
        return {k: torch.stack([item[k] for item in snippets_list]) for k in keys}
    return torch.stack(snippets_list)


# ============================================================
# Dataset
# ============================================================

class AnomalyVideoDataset(Dataset):
    """Unified dataset for all four benchmarks."""

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        split: str = "train",
        snippet_len: int = 16,
        num_snippets: int = 64,
        stride: int = 8,
        frame_size: int = 224,
        seen_classes: list = None,
        unseen_classes: list = None,
        task_classes: list = None,
        mode: str = "standard",
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.snippet_len = snippet_len
        self.num_snippets = num_snippets
        self.stride = stride
        self.frame_size = frame_size
        self.seen_classes = seen_classes or []
        self.unseen_classes = unseen_classes or []
        self.task_classes = task_classes
        self.mode = mode
        self.samples = []
        self._load_annotations()

    def _annotation_paths(self):
        anno_dir = os.path.join(self.data_root, self.dataset_name, "annotations")
        candidates = [
            os.path.join(anno_dir, f"{self.split}_list.txt"),
            os.path.join(anno_dir, f"{self.split}.txt"),
            os.path.join(anno_dir, f"{self.split}.list"),
        ]
        return [p for p in candidates if os.path.exists(p)]

    def _load_annotations(self):
        paths = self._annotation_paths()
        if not paths:
            self._generate_dummy_annotations()
            return
        with open(paths[0], "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                video_name, label, anomaly_class = parts[0], int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                video_name, label, anomaly_class = parts[0], int(parts[1]), 0
            else:
                continue

            # Open-world filtering for UCF Seen-8 / Unseen-5 protocol.
            if self.mode == "open_world" and label == 1:
                if anomaly_class in self.unseen_classes:
                    if self.split == "train":
                        continue
                    label = 2  # unknown anomaly

            # Incremental task class filter (per-dataset).
            if self.task_classes is not None and label == 1:
                if anomaly_class not in self.task_classes:
                    continue

            video_path = os.path.join(self.data_root, self.dataset_name, "videos", video_name)
            feat_basename = (
                video_name.replace(".mp4", ".pt").replace(".avi", ".pt").replace(".mkv", ".pt")
            )
            if not feat_basename.endswith(".pt"):
                feat_basename = f"{video_name}.pt"
            feat_path = os.path.join(self.data_root, self.dataset_name, "features", feat_basename)
            self.samples.append({
                "video_path": video_path,
                "feat_path": feat_path,
                "label": label,
                "anomaly_class": anomaly_class,
                "video_name": video_name,
            })

    def _generate_dummy_annotations(self):
        if self.split == "train":
            n = 100
        elif self.split == "val":
            n = 20
        else:
            n = 30
        for i in range(n):
            label = 0 if i % 3 == 0 else 1
            anomaly_class = i % 13 if label == 1 else 13
            self.samples.append({
                "video_path": "",
                "feat_path": "",
                "label": label,
                "anomaly_class": anomaly_class,
                "video_name": f"dummy_{self.dataset_name}_{self.split}_{i:04d}",
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        anomaly_class = sample["anomaly_class"]
        video_name = sample["video_name"]
        feat_path = sample["feat_path"]
        if feat_path and os.path.exists(feat_path):
            data = torch.load(feat_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                if all(k in data for k in ("video", "clip", "motion")):
                    snippets = {k: data[k] for k in ("video", "clip", "motion")}
                elif all(k in data for k in ("video_features", "clip_features", "motion_features")):
                    snippets = {
                        "video": data["video_features"],
                        "clip": data["clip_features"],
                        "motion": data["motion_features"],
                    }
                else:
                    snippets = data.get("features", data.get("snippet_features"))
                frame_labels = data.get("frame_labels", torch.zeros(_snippet_length(snippets)))
            else:
                snippets = data
                frame_labels = torch.zeros(snippets.shape[0])
        else:
            T = self.num_snippets if self.split == "train" else 128
            snippets = torch.randn(T, 2048)
            frame_labels = torch.zeros(T)
            if label == 1:
                a = random.randint(0, T // 2)
                b = min(a + random.randint(T // 8, T // 4), T)
                frame_labels[a:b] = 1.0

        if self.split == "train" and self.num_snippets > 0:
            snippets, frame_labels = self._fixed_length_sample(snippets, frame_labels)
        return snippets, label, frame_labels, anomaly_class, video_name

    def _fixed_length_sample(self, snippets, frame_labels):
        T = _snippet_length(snippets)
        target_T = self.num_snippets
        if T >= target_T:
            start = random.randint(0, T - target_T) if T > target_T else 0
            return _slice_snippets(snippets, start, start + target_T), frame_labels[start:start + target_T]
        pad_len = target_T - T
        snippets = _pad_snippets(snippets, pad_len)
        frame_labels = torch.cat([frame_labels, frame_labels[-1:].repeat(pad_len)], dim=0)
        return snippets, frame_labels


# ============================================================
# Sampler / collate
# ============================================================

class PairedBatchSampler(Sampler):
    """16 normal + 16 abnormal samples per batch (paper §IV-B)."""

    def __init__(self, dataset, batch_size: int, seed: int = 42):
        if batch_size % 2 != 0:
            raise ValueError("Paired batching requires an even batch size")
        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.normal = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]
        self.abnormal = [i for i, s in enumerate(dataset.samples) if s["label"] > 0]

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        normal = rng.permutation(self.normal).tolist()
        abnormal = rng.permutation(self.abnormal).tolist()
        n_batches = min(len(normal), len(abnormal)) // self.half
        for i in range(n_batches):
            batch = normal[i * self.half:(i + 1) * self.half]
            batch += abnormal[i * self.half:(i + 1) * self.half]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(self.normal), len(self.abnormal)) // self.half


class DomainTaggedDataset(Dataset):
    """Wrap a dataset and append a domain id per sample."""

    def __init__(self, dataset, domain_id: int):
        self.dataset = dataset
        self.domain_id = domain_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], self.domain_id)


def _make_collate(allow_domain: bool = False):
    def collate_fn(batch):
        has_domain = len(batch[0]) == 6
        snippets_list, labels, fls, classes, names, domains = [], [], [], [], [], []
        max_len = max(_snippet_length(b[0]) for b in batch)
        for item in batch:
            if has_domain:
                snippets, label, fl, ac, vn, dom = item
                domains.append(int(dom))
            else:
                snippets, label, fl, ac, vn = item
            T = _snippet_length(snippets)
            if T < max_len:
                pad = max_len - T
                snippets = _pad_snippets(snippets, pad)
                fl = torch.cat([fl, fl[-1:].repeat(pad)], dim=0)
            snippets_list.append(snippets)
            labels.append(label)
            fls.append(fl)
            classes.append(ac)
            names.append(vn)
        out = (
            _stack_snippets(snippets_list),
            torch.tensor(labels, dtype=torch.long),
            torch.stack(fls),
            torch.tensor(classes, dtype=torch.long),
            names,
        )
        if has_domain and allow_domain:
            out = (*out, torch.tensor(domains, dtype=torch.long))
        return out
    return collate_fn


# ============================================================
# Public loader factories
# ============================================================

def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _per_rank_batch_size(cfg) -> int:
    """Paper hardware target: global batch=32 split across world_size=2.

    DistributedSampler hands every rank ``len(dataset) // world_size``
    indices, so each DataLoader iteration produces ``per_rank_batch``
    samples; effective global batch = ``per_rank_batch * world_size``.
    """
    if _is_dist():
        ws = dist.get_world_size()
        return max(1, cfg.train.batch_size // ws)
    return cfg.train.batch_size


def create_dataloaders(cfg, split_type: str = "standard"):
    ds_cfg = cfg.data
    dataset_name = ds_cfg.dataset
    train_ds = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=dataset_name,
        split="train", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_train, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size,
        seen_classes=ds_cfg.seen_classes, unseen_classes=ds_cfg.unseen_classes,
        mode=split_type,
    )
    val_ds = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=dataset_name,
        split="val", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_test, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size,
        seen_classes=ds_cfg.seen_classes, unseen_classes=ds_cfg.unseen_classes,
        mode=split_type,
    )
    test_ds = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=dataset_name,
        split="test", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_test, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size,
        seen_classes=ds_cfg.seen_classes, unseen_classes=ds_cfg.unseen_classes,
        mode=split_type,
    )

    collate_fn = _make_collate(allow_domain=False)
    per_rank_bs = _per_rank_batch_size(cfg)
    if cfg.train.get("paired_batch", False) and not _is_dist():
        # PairedBatchSampler is single-process only; under DDP we fall
        # back to a DistributedSampler with a per-rank batch.
        train_loader = DataLoader(
            train_ds,
            batch_sampler=PairedBatchSampler(train_ds, per_rank_bs, cfg.seed),
            num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
    elif _is_dist():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=per_rank_bs, sampler=train_sampler,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=per_rank_bs, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
            pin_memory=True, drop_last=True,
        )

    val_sampler = DistributedSampler(val_ds, shuffle=False) if _is_dist() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if _is_dist() else None
    val_loader = DataLoader(
        val_ds, batch_size=per_rank_bs, shuffle=False,
        sampler=val_sampler, num_workers=cfg.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=per_rank_bs, shuffle=False,
        sampler=test_sampler, num_workers=cfg.num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def create_cross_dataset_loaders(cfg, source_dataset: str, target_dataset: str):
    ds_cfg = cfg.data
    source_names = source_dataset.split("+")
    source_parts = [
        DomainTaggedDataset(
            AnomalyVideoDataset(
                data_root=ds_cfg.data_root, dataset_name=name,
                split="train", snippet_len=ds_cfg.snippet_len,
                num_snippets=ds_cfg.num_snippets_train, stride=ds_cfg.stride,
                frame_size=ds_cfg.frame_size, mode="standard",
            ),
            domain_id=i,
        )
        for i, name in enumerate(source_names)
    ]
    source_train = source_parts[0] if len(source_parts) == 1 else ConcatDataset(source_parts)
    target_test = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=target_dataset,
        split="test", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_test, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size, mode="standard",
    )
    collate_fn = _make_collate(allow_domain=True)
    per_rank_bs = _per_rank_batch_size(cfg)
    source_sampler = DistributedSampler(source_train, shuffle=True, drop_last=True) if _is_dist() else None
    target_sampler = DistributedSampler(target_test, shuffle=False) if _is_dist() else None
    source_loader = DataLoader(
        source_train, batch_size=per_rank_bs,
        shuffle=(source_sampler is None), sampler=source_sampler,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    target_loader = DataLoader(
        target_test, batch_size=per_rank_bs,
        shuffle=False, sampler=target_sampler,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    return source_loader, target_loader


def create_incremental_loaders(cfg, task_id: int):
    """Cross-dataset incremental loader (paper main.tex T0..T4).

    Honours ``cfg.incremental.task_specs`` if present (each entry has
    ``dataset`` and ``classes`` keys); otherwise falls back to
    ``cfg.incremental.task_classes`` on a single dataset.
    """
    ds_cfg = cfg.data
    inc_cfg = cfg.incremental
    if "task_specs" in inc_cfg and inc_cfg.task_specs is not None and task_id < len(inc_cfg.task_specs):
        spec = inc_cfg.task_specs[task_id]
        dataset_name = spec["dataset"]
        task_classes = list(spec["classes"])
    else:
        dataset_name = ds_cfg.dataset
        task_classes = list(inc_cfg.task_classes[task_id])

    train_ds = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=dataset_name,
        split="train", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_train, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size,
        task_classes=task_classes, mode="incremental",
    )
    test_ds = AnomalyVideoDataset(
        data_root=ds_cfg.data_root, dataset_name=dataset_name,
        split="test", snippet_len=ds_cfg.snippet_len,
        num_snippets=ds_cfg.num_snippets_test, stride=ds_cfg.stride,
        frame_size=ds_cfg.frame_size, mode="incremental",
    )
    collate_fn = _make_collate(allow_domain=False)
    per_rank_bs = _per_rank_batch_size(cfg)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if _is_dist() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if _is_dist() else None
    train_loader = DataLoader(
        train_ds, batch_size=per_rank_bs,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=per_rank_bs,
        shuffle=False, sampler=test_sampler,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    return train_loader, test_loader
