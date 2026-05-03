# EIM-OWILNet

**Explainable Open-World Incremental-Learning Network for Video Anomaly Detection via Temporal Equivalent AND-OR Interactions**

EIM-OWILNet routes weakly-supervised video anomaly detection, open-world recognition, incremental learning, and explanation faithfulness through **one** sparse interaction dictionary built on top of three frozen backbones (VideoMAE-L, CLIP ViT-L/14, YOLO+RAFT) and a 96-concept human-readable bottleneck. The core operator (TD-EAIM) replaces a black-box ranker with a second-order Harsanyi expansion gated by hard-concrete masks, so every score factor has a named semantics that downstream tasks share, audit, and distill.

| Axis | Result |
| --- | --- |
| UCF-Crime ROC-AUC | **92.46 ± 0.18** |
| XD-Violence AP | **92.84** |
| Cross-domain UCF→XD | **79.32** |
| Open-world OSCR (Seen-8 / Unseen-5) | **67.05** |
| Incremental Avg-AUC (T0→T4) | **87.46** |
| Faithfulness Drop@K | **0.342** |

Tables I-VII of the paper for the full breakdown.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation protocols](#evaluation-protocols)
- [Reproducibility checklist](#reproducibility-checklist)
- [Project hygiene](#project-hygiene)
- [Citation](#citation)
- [License](#license)

---

## Repository layout

```
papercode/
├── configs/                    # OmegaConf default config (paper hyper-params)
├── datasets/                   # Unified UCF / XD / SHT / UB loader + 96-concept vocabulary
├── losses/                     # Composite loss (7 + L_civ terms, Eq. 9 + Eq. 3)
├── metrics/                    # Detection / Open-World / Incremental / Faithfulness
├── models/
│   ├── feature_extractor.py    # Multi-source paper-sum fusion + modality dropout
│   ├── concept_bottleneck.py   # K=96 concept activations + post-hoc τ_c calibration
│   ├── td_eaim.py              # Temporal AND-OR Interaction Machine (Eq. 6-8)
│   ├── vad_head.py             # Dynamic-MIL head (Eq. 10)
│   ├── open_world_head.py      # Posterior + energy + SVD-projection residual
│   ├── incremental_learner.py  # Three memories + frozen-teacher distillation
│   ├── cross_dataset_invariance.py  # GRL + per-domain β-variance L_inv
│   └── eim_owilnet.py          # End-to-end pipeline + EMA teacher + Algorithm 2
├── scripts/
│   ├── download_datasets.py    # 4-dataset download manifest with sha256 + license
│   ├── build_splits.py         # Generate paper-canonical train / val / test lists
│   ├── extract_features.py     # VideoMAE-L / CLIP / YOLO+RAFT feature pre-extractor
│   ├── build_concept_text_embeddings.py  # 5-prompt CLIP text bank for the 96 concepts
│   ├── launch_a100.sh          # 2× A100-80GB torchrun launcher
│   └── smoke_test.py           # Tiny end-to-end forward+backward sanity test
├── utils/                      # Common helpers (seed, ckpt, logger, AMP)
├── train.py                    # Entry-point: standard / open_world / incremental / cross_dataset
├── test.py                     # Standalone inference / explanation
├── validate.py                 # Held-out evaluation
├── run_cross_validation.py     # 5-seed cross-validation orchestrator
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/1846659840/-EIM-OWILNet.git
cd -EIM-OWILNet/papercode
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested on Ubuntu 22.04 / Windows 11 with Python 3.10/3.11, PyTorch 2.1+ on 2× NVIDIA A100-80GB.

## Datasets

The paper protocol uses four public benchmarks. Their videos cannot be redistributed here, so we ship a download manifest with the canonical URLs and a split builder that produces the paper's exact train / val / test cardinality.

| Dataset | Train / Val / Test | Notes |
| --- | --- | --- |
| UCF-Crime | 1610 / − / 290 | 13 anomaly classes |
| XD-Violence | 3954 / − / 800 | Multi-source, 6 violence classes |
| ShanghaiTech | 238 / − / 199 | Binary, weakly labelled |
| UBnormal | 268 / 64 / 211 | Synthetic open-set |

```bash
# 1. Print the manifest (license / homepage / file list)
python scripts/download_datasets.py --dataset all

# 2. Trigger archive download once you have agreed to the licenses
python scripts/download_datasets.py --dataset all --download

# 3. Build paper-canonical split lists (placeholder rows are written
#    if the raw videos are still missing, with the correct cardinality)
python scripts/build_splits.py --data_root ./data --dataset all

# 4. Extract VideoMAE-L / CLIP / YOLO+RAFT features into per-video .pt
python scripts/extract_features.py --dataset ucf_crime --data_root ./data
```

Output layout per dataset:

```
data/<dataset>/
├── videos/                  # raw .mp4 / .avi / .mkv (downloaded by user)
├── features/                # pre-extracted .pt {video, clip, motion, frame_labels}
└── annotations/
    ├── train_list.txt
    ├── val_list.txt          # UBnormal only
    └── test_list.txt
```

Annotation format: `<video_relpath> <video_label> <anomaly_class>` where `video_label ∈ {0, 1, 2}` and class indices follow `datasets.video_dataset.UCF_CLASS_NAMES`.

## Training

### Single-GPU (debug / smoke)

```bash
python train.py --protocol standard --seed 42
```

### 2× A100-80GB (paper hardware)

```bash
bash scripts/launch_a100.sh standard       # Protocol A: weakly-supervised VAD
bash scripts/launch_a100.sh cross_dataset  # Protocol B: source→target transfer
bash scripts/launch_a100.sh open_world     # Protocol C: Seen-8 / Unseen-5
bash scripts/launch_a100.sh incremental    # Protocol D: T0→T4 (T3=SHT, T4=UB)
```

Override knobs:

```bash
NPROC=2 OMC_BF16=1 SEED=2024 bash scripts/launch_a100.sh standard
```

The launcher auto-wires:

- `torchrun --standalone --nproc_per_node=2`
- bf16 autocast (A100 native; falls back to fp16 GradScaler when `OMC_BF16=0`)
- DDP with NCCL backend; `DistributedSampler` + per-rank batch size 16 (global = 32)
- 3-stage schedule 50 + 30 + 20 epochs; cosine LR with 5-epoch warmup
- EMA teacher decay 0.99 with concept-intervention loss (Eq. 3)
- Hard-concrete temperature anneal 0.5 → 0.1 over the first 30 epochs
- Modality dropout p_m anneal 0.3 → 0.05 over the first 20 % steps
- 5-fold cross-validation seeds `{42, 123, 2024, 7, 31}`

### Cross-validation

```bash
python run_cross_validation.py --protocol standard
```

## Evaluation protocols

| Protocol | Metric set | Entry point |
| --- | --- | --- |
| Detection | ROC-AUC, AP, FAR@0.5 | `train_standard` |
| Cross-domain | ROC-AUC (Train→Test) | `train_cross_dataset` |
| Open-world | Known-AUC, Unk-AUROC, OSCR, H, NMI | `train_open_world` |
| Incremental (5 tasks) | Avg-AUC, BWT, FWT, Forget, Mem | `train_incremental` |
| Faithfulness | Drop@K, AOPC, Suff, Comp, InsDel | `test.py --explain` |

`models/eim_owilnet.EIMOWILNet.compute_drop_at_k` and `compute_interaction_perturbation_scores` provide the perturbation curves for Tables IV-V; `models/eim_owilnet.EIMOWILNet.get_explanation` returns the human-readable named interactions backing Figure 8.

## Reproducibility checklist

- ✅ Hyperparameters: `configs/default.py` mirrors Appendix Table I exactly (`λ_{1..6}=(0.3,0.5,0.1,0.4,0.6,0.2)`, `η_{1..3}=(1.0,0.5,0.3)`, `λ_civ=0.1`, `α_MIL=0.6`, distillation T=4.0, exemplars=20, K=96, m=64, T=64, Δ∈{1,2,4,8}).
- ✅ Hardware: 2× NVIDIA A100-80GB; bf16 autocast; DDP via `torchrun`.
- ✅ Splits: train / val / test cardinality verified against Appendix Table I (1610/290, 3954/800, 238/199, 268/64/211).
- ✅ Open-world Seen-8 / Unseen-5 indices match `UCF_CLASS_NAMES`.
- ✅ Incremental T_3 = ShanghaiTech, T_4 = UBnormal as per main.tex § III-E.
- ✅ Algorithm 2 cluster-seeding runs on the unlabeled training stream (no test-data leakage).
- ✅ EMA teacher copy + concept-intervention loss (Eq. 3) implemented.
- ✅ P_known is a true SVD-orthogonal projector; r_t has the geometric meaning required by Proposition 2.
- ✅ Smoke test in `scripts/smoke_test.py` runs forward + backward in <30 s on CPU.

Run the smoke test before any large training job:

```bash
python scripts/smoke_test.py
```

## Project hygiene

- `losses/composite_loss.py` aggregates 7 paper terms plus `λ_civ L_civ`.
- `models/td_eaim.py` keeps the linear part outside the hard-concrete gate (Eq. 7) and uses MI-disjoint OR pairs (paper § III-B).
- `models/cross_dataset_invariance.py` keeps the current domain's β as a differentiable entry of the variance stack so `L_inv` can flow gradients.
- `models/incremental_learner.py` stores a frozen `eval()` deepcopy as the teacher; same-input KD between student and teacher.
- All file edits are accompanied by tests in `scripts/smoke_test.py` and per-module Python checks in `scripts/`.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{eimowilnet2026,
  title   = {Explainable Open-World Incremental-Learning Network for Video
             Anomaly Detection via Temporal Equivalent AND-OR Interactions},
  author  = {EIM-OWILNet authors},
  journal = {Under review},
  year    = {2026}
}
```

A `CITATION.cff` is provided for tooling that prefers the GitHub citation format.

## License

Code is released under the MIT License (see `LICENSE`). Datasets retain their original licenses; please consult their respective homepages before redistributing any video frames or annotations.
