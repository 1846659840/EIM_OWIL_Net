"""Dataset download orchestrator for EIM-OWILNet.

The paper protocol uses four public benchmarks:

  1. UCF-Crime    (Sultani et al., CVPR 2018)        — Dropbox / official
  2. XD-Violence  (Wu et al., ECCV 2020)             — RoseLab / official
  3. ShanghaiTech (Liu et al., CVPR 2018)            — Dropbox / official
  4. UBnormal     (Acsintoae et al., CVPR 2022)      — Drive / official

Distribution licenses prohibit blanket re-hosting, so this script does
not bundle the raw videos. Instead it (a) records the canonical
download URLs the user must visit, (b) verifies their checksums, and
(c) places everything under the layout

    data/<dataset>/
        videos/        # raw .mp4 files (or .avi / .mkv as released)
        features/      # pre-extracted .pt files (1024-d VideoMAE-L,
                       #                          768-d CLIP-ViT-L/14,
                       #                          4096-d YOLO+RAFT)
        annotations/
            train_list.txt
            test_list.txt
            (val_list.txt for UBnormal)

Run:
    python scripts/download_datasets.py --dataset all      # print URLs
    python scripts/download_datasets.py --dataset ucf_crime --download
    python scripts/download_datasets.py --dataset all --features

The --features flag triggers ``scripts/extract_features.py`` to
populate ``features/`` once raw videos are present.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------
# Canonical download manifests
# ---------------------------------------------------------------
# Each manifest is a list of (filename, url, sha256) tuples; sha256 may
# be left as None when the upstream provider does not publish one.

DATASET_MANIFESTS: Dict[str, Dict] = {
    "ucf_crime": {
        "homepage": "https://www.crcv.ucf.edu/projects/real-world/",
        "license": "Research-only; cite Sultani et al. 2018.",
        "files": [
            (
                "Anomaly-Videos.zip",
                "https://www.crcv.ucf.edu/projects/real-world/UCF_Crimes/UCF_Crimes.zip",
                None,
            ),
            (
                "Annotation.zip",
                "https://www.crcv.ucf.edu/projects/real-world/Action_Regnition_splits.zip",
                None,
            ),
        ],
        "split_sizes": {"train": 1610, "test": 290},
        "expected_videos": 1900,
        "structure": ["Anomaly-Videos", "Normal_Videos_for_Event_Recognition"],
    },
    "xd_violence": {
        "homepage": "https://roc-ng.github.io/XD-Violence/",
        "license": "Research-only; cite Wu et al. 2020.",
        "files": [
            (
                "RGB.zip",
                "https://roc-ng.github.io/XD-Violence/RGB/",  # placeholder; user must request
                None,
            ),
            (
                "annotations.zip",
                "https://roc-ng.github.io/XD-Violence/annotations.zip",
                None,
            ),
        ],
        "split_sizes": {"train": 3954, "test": 800},
        "expected_videos": 4754,
        "structure": ["videos"],
    },
    "shanghai_tech": {
        "homepage": "https://svip-lab.github.io/dataset/campus_dataset.html",
        "license": "Research-only; cite Liu et al. 2018.",
        "files": [
            (
                "ShanghaiTech.tar.gz",
                "https://onedrive.live.com/?authkey=...&id=...",  # request from authors
                None,
            ),
        ],
        "split_sizes": {"train": 238, "test": 199},
        "expected_videos": 437,
        "structure": ["training", "testing"],
    },
    "ubnormal": {
        "homepage": "https://github.com/lilygeorgescu/UBnormal",
        "license": "CC-BY-4.0 (synthetic).",
        "files": [
            (
                "UBnormal.zip",
                "https://github.com/lilygeorgescu/UBnormal/releases/download/v1.0/UBnormal.zip",
                None,
            ),
        ],
        "split_sizes": {"train": 268, "val": 64, "test": 211},
        "expected_videos": 543,
        "structure": ["training", "validation", "test"],
    },
}


def _print_manifest(name: str, manifest: Dict):
    print(f"\n=== {name} ===")
    print(f"  homepage : {manifest['homepage']}")
    print(f"  license  : {manifest['license']}")
    print(f"  expected : {manifest['expected_videos']} videos | "
          f"split={manifest['split_sizes']}")
    print("  files    :")
    for fn, url, sha in manifest["files"]:
        print(f"    - {fn}")
        print(f"        url    : {url}")
        if sha:
            print(f"        sha256 : {sha}")


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"[skip] {dest} already exists.")
        return True
    print(f"[get ] {url} -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except Exception as e:
        print(f"[fail] {url}: {e}")
        return False


def _verify(path: Path, sha256: Optional[str]) -> bool:
    if sha256 is None:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != sha256:
        print(f"[fail] sha256 mismatch for {path}: got {digest}, want {sha256}")
        return False
    return True


def _ensure_layout(data_root: Path, dataset: str):
    base = data_root / dataset
    for sub in ("videos", "features", "annotations"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def fetch_dataset(name: str, data_root: Path, do_download: bool):
    manifest = DATASET_MANIFESTS[name]
    base = _ensure_layout(data_root, name)
    archive_dir = base / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    _print_manifest(name, manifest)
    if not do_download:
        print(f"  (preview only; pass --download to fetch into {archive_dir})")
        return

    for fn, url, sha in manifest["files"]:
        dest = archive_dir / fn
        if not _download(url, dest):
            print(
                f"  Manual fallback: please download {fn} from the homepage "
                f"and copy it into {archive_dir}"
            )
            continue
        if not _verify(dest, sha):
            sys.exit(2)
    print(f"[ok ] {name} archives staged at {archive_dir}")


def extract_features(name: str, data_root: Path):
    """Invoke scripts/extract_features.py for `name`."""
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("extract_features.py")),
        "--dataset", name, "--data_root", str(data_root),
    ]
    print("[run ] " + " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="EIM-OWILNet dataset downloader")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "ucf_crime", "xd_violence", "shanghai_tech", "ubnormal"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--download", action="store_true",
                        help="actually fetch the archives (otherwise just preview)")
    parser.add_argument("--features", action="store_true",
                        help="run feature extraction after archives are present")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    targets: List[str]
    if args.dataset == "all":
        targets = list(DATASET_MANIFESTS.keys())
    else:
        targets = [args.dataset]

    for name in targets:
        fetch_dataset(name, data_root, args.download)
        if args.features:
            extract_features(name, data_root)


if __name__ == "__main__":
    main()
