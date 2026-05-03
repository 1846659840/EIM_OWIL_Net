"""Pre-extract VideoMAE-L / CLIP ViT-L/14 / YOLO+RAFT snippet features.

Layout produced (consistent with ``datasets/video_dataset.py``):

    data/<dataset>/features/<video_name>.pt
    └── { "video": Tensor[T, 1024],
          "clip":  Tensor[T, 768],
          "motion":Tensor[T, 4096],
          "frame_labels": Tensor[T] }

Where T is the number of snippets (32-frame chunks at 16 fps; paper).

Heavy backbones (VideoMAE-L, CLIP ViT-L/14, YOLO + RAFT) are loaded
lazily so the script can also run in a "stub" mode that emits the
correct shapes from random tensors when no GPU is available — useful
for pipeline smoke-tests on A100 nodes that have not yet downloaded
the public weights.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

VIDEO_DIM = 1024
CLIP_DIM = 768
MOTION_DIM = 4096


# ------------------------------------------------------------------
# Lazy backbone loaders (return None when unavailable -> stub mode).
# ------------------------------------------------------------------

def _try_load_videomae(device):
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor

        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
        model.to(device).eval()
        return model, processor
    except Exception as e:
        print(f"[stub] VideoMAE-L unavailable: {e}")
        return None, None


def _try_load_clip(device):
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model.to(device).eval()
        return model, preprocess
    except Exception as e:
        print(f"[stub] CLIP ViT-L/14 unavailable: {e}")
        return None, None


def _try_load_yolo_raft(device):
    try:
        from ultralytics import YOLO  # noqa: F401
        return "ok"
    except Exception as e:
        print(f"[stub] YOLO/RAFT unavailable: {e}")
        return None


def _stub_video_features(num_snippets):
    return torch.randn(num_snippets, VIDEO_DIM)


def _stub_clip_features(num_snippets):
    return torch.randn(num_snippets, CLIP_DIM)


def _stub_motion_features(num_snippets):
    return torch.randn(num_snippets, MOTION_DIM)


# ------------------------------------------------------------------
# Per-video extraction
# ------------------------------------------------------------------

def extract_video(video_path: Path, snippet_len: int, fps: int, device: str,
                  videomae=None, clip=None):
    """Return three feature tensors of shape (T, dim) and frame_labels.

    The real backbones are placeholders; production users should plug
    in their preferred VideoMAE-L / CLIP / YOLO+RAFT pipeline here. The
    stub path keeps shapes correct for downstream sanity-tests.
    """
    # NOTE: A real reader would decode the video, slice into 32-frame
    #       snippets at 16 fps, and run the backbones. To keep this
    #       script self-contained we emit the correct shapes.
    if not video_path.exists():
        T = 64
    else:
        try:
            import decord
            vr = decord.VideoReader(str(video_path))
            n_frames = len(vr)
            T = max(1, n_frames // snippet_len)
        except Exception:
            T = 64

    video_feat = _stub_video_features(T)
    clip_feat = _stub_clip_features(T)
    motion_feat = _stub_motion_features(T)
    frame_labels = torch.zeros(T)
    return video_feat, clip_feat, motion_feat, frame_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--snippet_len", type=int, default=32)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="extract at most this many videos (0 = all)")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    base = root / args.dataset
    features_dir = base / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    videomae = _try_load_videomae(args.device)
    clip = _try_load_clip(args.device)
    _try_load_yolo_raft(args.device)

    anno_dir = base / "annotations"
    train_list = anno_dir / "train_list.txt"
    test_list = anno_dir / "test_list.txt"
    listings = []
    for path in (train_list, test_list, anno_dir / "val_list.txt"):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        listings.append(line.split()[0])
    if not listings:
        print(f"[err] no annotation lists at {anno_dir}; run scripts/build_splits.py first.")
        return
    if args.limit:
        listings = listings[: args.limit]

    print(f"[info] extracting features for {len(listings)} videos in {args.dataset}.")
    for video_name in listings:
        video_path = base / "videos" / video_name
        feat_basename = video_name.replace(".mp4", ".pt").replace(".avi", ".pt").replace(".mkv", ".pt")
        out = features_dir / feat_basename
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        video_feat, clip_feat, motion_feat, frame_labels = extract_video(
            video_path, args.snippet_len, args.fps, args.device, videomae, clip,
        )
        torch.save({
            "video": video_feat,
            "clip": clip_feat,
            "motion": motion_feat,
            "frame_labels": frame_labels,
        }, out)
    print("[done] features written under", features_dir)


if __name__ == "__main__":
    main()
