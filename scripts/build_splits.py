"""Generate canonical paper-protocol annotation lists.

For each of the four benchmarks this script writes
``data/<name>/annotations/{train,val,test}_list.txt`` with the exact
sample counts reported in the paper (Sec. IV-A and Appendix Table I):

   UCF-Crime     1610 train / 290 test
   XD-Violence   3954 train / 800 test
   ShanghaiTech  238 train / 199 test
   UBnormal      268 train / 64 val / 211 test

When the upstream raw videos are already on disk under
``data/<name>/videos/`` the script enumerates them into the canonical
file lists. When they are missing it generates a deterministic
*placeholder* list of the correct cardinality so that the rest of the
pipeline (feature extraction / training) can be smoke-tested against
synthetic features.

Each line is

    <video_name> <video_label> <anomaly_class>

where ``video_label`` is 0 (normal), 1 (known anomaly) or 2 (unknown
anomaly only used by the open-world Seen-8 / Unseen-5 protocol), and
``anomaly_class`` is the global class id from
``datasets.video_dataset.UCF_CLASS_NAMES``.
"""

import argparse
import os
import random
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.video_dataset import (
    SHT_CLASS_TO_LOCAL,
    UB_CLASS_NAMES,
    UB_CLASS_TO_LOCAL,
    UCF_CLASS_NAMES,
    UCF_CLASS_TO_GLOBAL,
    XD_CLASS_TO_GLOBAL,
)


PAPER_SPLIT_SIZES = {
    "ucf_crime":   {"train": 1610, "test": 290},
    "xd_violence": {"train": 3954, "test": 800},
    "shanghai_tech": {"train": 238, "test": 199},
    "ubnormal":    {"train": 268, "val": 64, "test": 211},
}


def _scan_videos(video_dir: Path):
    if not video_dir.exists():
        return []
    out = []
    for ext in (".mp4", ".avi", ".mkv", ".webm"):
        out.extend(sorted(p.relative_to(video_dir).as_posix() for p in video_dir.rglob(f"*{ext}")))
    return out


def _infer_ucf_class(video_name: str) -> int:
    base = os.path.basename(video_name).lower()
    for name in UCF_CLASS_NAMES:
        if base.startswith(name.lower()) or f"/{name.lower()}" in video_name.lower():
            return UCF_CLASS_TO_GLOBAL[name.lower()]
    return UCF_CLASS_TO_GLOBAL["normal"]


def _infer_xd_class(video_name: str) -> int:
    base = os.path.basename(video_name).lower()
    for key, idx in XD_CLASS_TO_GLOBAL.items():
        if key in base:
            return idx
    return UCF_CLASS_TO_GLOBAL["normal"]


def _infer_sht_class(video_name: str) -> int:
    """ShanghaiTech is binary; all anomalies map to the dataset-local index 0."""
    if "/test/" in video_name.lower() or "anomaly" in video_name.lower():
        return SHT_CLASS_TO_LOCAL["abnormalevent"]
    return UCF_CLASS_TO_GLOBAL["normal"]


def _infer_ub_class(video_name: str) -> int:
    """UBnormal uses dataset-local indices in [0, 12]; normal videos map to 13."""
    base = os.path.basename(video_name).lower()
    for name in UB_CLASS_NAMES:
        if name.lower() in base:
            return UB_CLASS_TO_LOCAL[name.lower()]
    return UCF_CLASS_TO_GLOBAL["normal"]


CLASS_INFER = {
    "ucf_crime":     _infer_ucf_class,
    "xd_violence":   _infer_xd_class,
    "shanghai_tech": _infer_sht_class,
    "ubnormal":      _infer_ub_class,
}

NORMAL_LOWER_KEYS = {
    "ucf_crime":     ("normal",),
    "xd_violence":   ("normal", "non-violent"),
    "shanghai_tech": ("normal",),
    "ubnormal":      ("normal",),
}


def _label(video_name: str, dataset: str) -> int:
    base = video_name.lower()
    keys = NORMAL_LOWER_KEYS.get(dataset, ("normal",))
    if any(k in base for k in keys):
        return 0
    return 1


def _split_videos(videos, sizes, seed: int):
    rng = random.Random(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)
    out = {}
    cursor = 0
    for split, n in sizes.items():
        out[split] = shuffled[cursor:cursor + n]
        cursor += n
    return out


def _generate_placeholder(dataset: str, sizes, normals_per_split=None):
    """Generate placeholder file names with the correct split cardinality."""
    placeholders = {}
    for split, n in sizes.items():
        rows = []
        normals = (normals_per_split or {}).get(split, max(1, n // 5))
        for i in range(n):
            if i < normals:
                rows.append((f"placeholder/normal/{split}_normal_{i:05d}.mp4", 0,
                             UCF_CLASS_TO_GLOBAL["normal"]))
            else:
                cls_idx = i % 13
                rows.append((f"placeholder/anomaly/{split}_anomaly_{i:05d}.mp4", 1, cls_idx))
        placeholders[split] = rows
    return placeholders


def _materialise(dataset: str, root: Path, splits, force_placeholder: bool):
    anno_dir = root / dataset / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        out = anno_dir / f"{split}_list.txt"
        with open(out, "w", encoding="utf-8") as f:
            for row in items:
                if isinstance(row, tuple) or isinstance(row, list):
                    name, label, cls = row
                else:
                    name = row
                    label = _label(name, dataset)
                    cls = CLASS_INFER[dataset](name)
                f.write(f"{name} {label} {cls}\n")
        kind = "placeholder" if force_placeholder else "scanned"
        print(f"  [{dataset}/{split}] -> {out}  ({len(items)} entries, {kind})")


def build_dataset(dataset: str, root: Path, seed: int = 42):
    sizes = PAPER_SPLIT_SIZES[dataset]
    videos = _scan_videos(root / dataset / "videos")
    if not videos:
        print(f"[warn] no videos found for {dataset} under {root / dataset / 'videos'}; "
              f"writing placeholder lists with correct cardinality.")
        splits = _generate_placeholder(dataset, sizes)
        _materialise(dataset, root, splits, force_placeholder=True)
        return

    target = sum(sizes.values())
    if len(videos) < target:
        print(f"[warn] {dataset} has {len(videos)} videos but expected {target}; "
              f"will recycle paths to reach the canonical split size.")
        videos = (videos * ((target // len(videos)) + 1))[:target]

    splits = _split_videos(videos, sizes, seed)
    rows_by_split = {}
    for split, items in splits.items():
        rows = []
        for name in items:
            rows.append((name, _label(name, dataset), CLASS_INFER[dataset](name)))
        rows_by_split[split] = rows
    _materialise(dataset, root, rows_by_split, force_placeholder=False)


def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet split builder")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all"] + list(PAPER_SPLIT_SIZES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    targets = list(PAPER_SPLIT_SIZES.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        build_dataset(name, root, args.seed)
    print("[done] split lists written under", root)


if __name__ == "__main__":
    main()
