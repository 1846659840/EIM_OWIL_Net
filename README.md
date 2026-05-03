<div align="center">

# EIM-OWILNet

**Explainable Open-World Incremental-Learning Network for Video Anomaly Detection via Temporal Equivalent AND-OR Interactions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware: 2× A100](https://img.shields.io/badge/Hardware-2%C3%97%20A100--80GB-76B900.svg)](#hardware)
[![Status: Reproducible](https://img.shields.io/badge/status-reproducible-success.svg)](#reproducibility)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[**Paper**](#citation) · [**Quickstart**](#quickstart) · [**Datasets**](#datasets) · [**Training**](#training) · [**Results**](#results) · [**Reproducibility**](#reproducibility) · [**Contributing**](#contributing)

</div>

---

## Overview

EIM-OWILNet routes weakly-supervised video anomaly detection, open-world recognition, class-incremental learning, and explanation faithfulness through **one** sparse interaction dictionary built on top of three frozen backbones (VideoMAE-L, CLIP ViT-L/14, YOLO+RAFT) and a 96-concept human-readable bottleneck. The core operator (TD-EAIM) replaces a black-box ranker with a second-order Harsanyi expansion gated by hard-concrete masks, so every score factor has a named semantics that downstream tasks share, audit, and distill.

```
            ┌─────────────────────────────────────────────────────────────────┐
            │                  EIM-OWILNet end-to-end pipeline                │
            ├─────────────────────────────────────────────────────────────────┤
            │   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
   video ──►│   │VideoMAE-L│ ──►│  paper   │ ──►│ Concept  │ ──► A_t (B,T,96)│
            │   │  CLIP-L  │    │   sum    │    │bottleneck│                 │
            │   │YOLO+RAFT │    │ fusion   │    │  K = 96  │                 │
            │   └──────────┘    └──────────┘    └─────┬────┘                 │
            │                                          ▼                      │
            │                              ┌──────────────────────┐           │
            │                              │      TD-EAIM         │           │
            │                              │  AND ∨ OR ∨ Temporal │           │
            │                              │  hard-concrete gate  │           │
            │                              └──────────┬───────────┘           │
            │                                         ▼                       │
            │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
            │  │   VAD    │ │OpenWorld │ │Incremental│ │ Cross-   │            │
            │  │ MIL head │ │ residual │ │ KD + M_int│ │ Domain   │            │
            │  │   s_t    │ │ + energy │ │ + M_ex    │ │ GRL+L_inv│            │
            │  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
            └─────────────────────────────────────────────────────────────────┘
```

## Highlights

- 🧠 **Single shared dictionary, four heads** — anomaly score, open-world residual, incremental distillation, and cross-domain invariance read the *same* sparse, named interactions.
- 🔍 **Faithful explanations by construction** — removing a top-K named interaction is identically a removal from the score; Drop@K = 0.342 on UCF-Crime, 1.20× DSANet.
- 🌍 **Open-set ready** — orthogonal-projection residual + energy + posterior fused by an AND criterion (Proposition 2), 67.05 OSCR on UCF Seen-8 / Unseen-5.
- 📈 **Continual-learning friendly** — three replay memories (`M_ex`, `M_proto`, `M_int`), interaction-level distillation, and Algorithm 2 cluster-seeding of new prototypes.
- ⚡ **Production knobs** — DDP on 2× A100-80GB with bf16 autocast, EMA teacher, 5-seed cross-validation, deterministic gates at eval, paper-canonical splits.

## Table of contents

- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Hardware](#hardware)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Repository layout

```text
.
├── configs/                    # OmegaConf default config (paper hyper-params)
├── datasets/                   # Unified UCF / XD / SHT / UB loader + 96-concept vocabulary
├── losses/                     # Composite loss (7 + L_civ terms; Eq. 9 + Eq. 3)
├── metrics/                    # Detection / Open-World / Incremental / Faithfulness
├── models/
│   ├── feature_extractor.py    # Multi-source paper-sum fusion + modality dropout
│   ├── concept_bottleneck.py   # K=96 concept activations + post-hoc τ_c calibration
│   ├── td_eaim.py              # Temporal AND-OR Interaction Machine (Eq. 6-8)
│   ├── vad_head.py             # Dynamic-MIL head (Eq. 10)
│   ├── open_world_head.py      # Posterior + energy + SVD-projection residual
│   ├── incremental_learner.py  # Three memories + frozen-teacher distillation
│   ├── cross_dataset_invariance.py  # GRL + per-domain β-variance L_inv
│   └── eim_owilnet.py          # Top-level pipeline + EMA teacher + Algorithm 2
├── scripts/
│   ├── download_datasets.py    # 4-dataset download manifest (URL / sha256 / license)
│   ├── build_splits.py         # Generate paper-canonical train / val / test lists
│   ├── extract_features.py     # VideoMAE-L / CLIP / YOLO+RAFT feature pre-extractor
│   ├── build_concept_text_embeddings.py  # 5-prompt CLIP text bank for the 96 concepts
│   ├── launch_a100.sh          # 2× A100-80GB torchrun launcher
│   └── smoke_test.py           # Tiny end-to-end forward+backward sanity test
├── utils/                      # Helpers (seed, ckpt, logger, AMP)
├── train.py                    # CLI: standard / open_world / incremental / cross_dataset
├── test.py                     # Standalone inference / explanation
├── validate.py                 # Held-out evaluation
├── run_cross_validation.py     # 5-seed cross-validation orchestrator
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Quickstart

```bash
# Clone, install, smoke-test, train (UCF-Crime, single GPU)
git clone https://github.com/1846659840/-EIM-OWILNet.git eim-owilnet
cd eim-owilnet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_splits.py --dataset all --data_root ./data
python scripts/smoke_test.py
python train.py --protocol standard --seed 42
```

The smoke test runs a four-batch forward + backward on synthetic data in under 30 seconds and asserts (a) all 8 loss terms aggregate, (b) the linear part is *not* gated, (c) `‖P² − P‖ < 1e-4`, (d) gradients flow through the full graph. It never needs a GPU or real videos.

## Installation

### Requirements

| Component | Version |
| --- | --- |
| Python | 3.10 or 3.11 |
| PyTorch | 2.1+ |
| CUDA | 11.8 / 12.1 (matching PyTorch wheels) |
| HDBSCAN | 0.8.33+ |
| OpenCV | 4.7+ |

```bash
pip install -r requirements.txt
```

Heavyweight backbones are loaded lazily and degrade to a stub mode when the public weights are absent, so the full requirements are only needed for end-to-end training.

### Optional: editable install

```bash
pip install -e .   # if a setup.py / pyproject.toml is added downstream
```

## Datasets

The paper protocol uses four public benchmarks. The videos cannot be redistributed here, so we ship a download manifest with the canonical URLs and a split builder that produces the paper's exact train / val / test cardinality.

| Dataset | Train | Val | Test | Anomaly classes |
| --- | ---: | ---: | ---: | --- |
| UCF-Crime | 1610 | – | 290 | 13 |
| XD-Violence | 3954 | – | 800 | 6 |
| ShanghaiTech | 238 | – | 199 | binary |
| UBnormal | 268 | 64 | 211 | 13 |

```bash
# 1. Print the manifest (license / homepage / file list)
python scripts/download_datasets.py --dataset all

# 2. Trigger archive download once you have agreed to the upstream licenses
python scripts/download_datasets.py --dataset all --download

# 3. Generate paper-canonical split files (placeholder rows are written
#    when raw videos are missing, with the correct cardinality so the
#    pipeline can be smoke-tested against synthetic features)
python scripts/build_splits.py --data_root ./data --dataset all

# 4. Pre-extract VideoMAE-L / CLIP ViT-L/14 / YOLO+RAFT features
python scripts/extract_features.py --dataset ucf_crime --data_root ./data
```

Output layout per dataset:

```text
data/<dataset>/
├── videos/                  # raw .mp4 / .avi / .mkv (downloaded by user)
├── features/                # pre-extracted .pt {video, clip, motion, frame_labels}
└── annotations/
    ├── train_list.txt       # <video_relpath> <video_label> <anomaly_class>
    ├── val_list.txt         # UBnormal only
    └── test_list.txt
```

`video_label ∈ {0=normal, 1=known anomaly, 2=unknown anomaly}`; class indices follow `datasets.video_dataset.UCF_CLASS_NAMES`.

## Training

### Single GPU (debug)

```bash
python train.py --protocol standard --seed 42
```

### 2× A100-80GB (paper hardware)

```bash
bash scripts/launch_a100.sh standard       # Protocol A — weakly-supervised VAD
bash scripts/launch_a100.sh cross_dataset  # Protocol B — source → target transfer
bash scripts/launch_a100.sh open_world     # Protocol C — Seen-8 / Unseen-5
bash scripts/launch_a100.sh incremental    # Protocol D — T0 → T4 (T3=SHT, T4=UB)
```

Override knobs via env-vars:

```bash
NPROC=2 OMC_BF16=1 SEED=2024 bash scripts/launch_a100.sh standard
```

The launcher automatically wires:

- `torchrun --standalone --nproc_per_node=$NPROC`
- bf16 autocast (A100 native; falls back to fp16 + GradScaler when `OMC_BF16=0`)
- DDP with NCCL backend; `DistributedSampler` + per-rank batch size 16 (global = 32)
- Three-stage schedule: 50 + 30 + 20 epochs with cosine LR and 5-epoch warmup
- EMA teacher (decay 0.99) + concept-intervention loss (Eq. 3)
- Hard-concrete temperature anneal `0.5 → 0.1` over the first 30 epochs
- Modality dropout `p_m` anneal `0.3 → 0.05` over the first 20% of training steps
- Five seeds `{42, 123, 2024, 7, 31}` for the cross-validation orchestrator

### Cross-validation

```bash
python run_cross_validation.py --protocol standard
```

### Resume from checkpoint

```bash
python train.py --protocol standard --resume checkpoints/best_model.pt
```

## Evaluation

| Protocol | Metrics | Entry point |
| --- | --- | --- |
| Detection | ROC-AUC, AP, FAR@0.5 | `train_standard` |
| Cross-domain | ROC-AUC (Train → Test) | `train_cross_dataset` |
| Open-world | Known-AUC, Unk-AUROC, OSCR, H, NMI | `train_open_world` |
| Incremental (5 tasks) | Avg-AUC, BWT, FWT, Forget, Mem (MB) | `train_incremental` |
| Faithfulness | Drop@K, AOPC, Suff, Comp, InsDel | `test.py --explain` |

Generate a JSON evaluation report on the held-out test split:

```bash
python validate.py --ckpt checkpoints/best_model.pt --protocol open_world
```

The model exposes two convenience APIs for explanation work:

- `EIMOWILNet.compute_drop_at_k(x, k=5)` — top-K interaction perturbation curves (Drop@K, AOPC, etc.).
- `EIMOWILNet.get_explanation(outputs, top_k=5)` — list of named human-readable interactions for the figure-style qualitative panels.

## Results

Numbers below are mean ± std over five seeds and reproduce the paper tables.

### Standard weakly-supervised VAD

| Method | UCF-AUC | UCF-AP | XD-AP | SHT-AUC | UB-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| GS-MoE (ICCV'25) | 91.58 | 82.04 | 91.65 | 98.18 | 68.91 |
| APST-Net (TNNLS'26) | 91.74 | 82.38 | 91.94 | 98.42 | 69.18 |
| DSANet (AAAI'26) | 91.92 | 82.65 | 92.18 | 98.61 | 69.55 |
| **EIM-OWILNet (Ours)** | **92.46** | **83.27** | **92.84** | **98.92** | **70.42** |

### Open-world recognition (UCF Seen-8 / Unseen-5)

| Method | Known-AUC | Unk-AUROC | OSCR | H | NMI |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSANet + MaxLogit | 85.23 | 71.05 | 64.92 | 73.46 | 0.395 |
| **EIM-OWILNet** | **86.42** | **73.18** | **67.05** | **75.62** | **0.421** |

### Incremental learning (T0 → T4)

| Method | Avg-AUC | BWT | FWT | Forget | Mem (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DSANet | 86.45 | -2.51 | +3.78 | 3.12 | 224 |
| **EIM-OWILNet** | **87.46** | **-1.84** | **+4.27** | **2.31** | 192 |

### Explanation faithfulness (UCF, K=5)

| Explainer | Drop@K ↑ | AOPC ↑ | Suff ↓ | Comp ↑ | InsDel ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 0.207 | 0.314 | 0.498 | 0.246 | 0.591 |
| DSANet Top-K | 0.284 | 0.418 | 0.392 | 0.336 | 0.692 |
| **EIM-OWILNet** | **0.342** | **0.476** | **0.318** | **0.395** | **0.742** |

Full per-axis tables, ablation sweeps, and qualitative studies live in the paper.

## Reproducibility

A reproducible run requires four things to line up.

### 1. Hyperparameters (Appendix Table I)

All hyperparameters live in [`configs/default.py`](configs/default.py). Critical values used for every reported number:

| Parameter | Value |
| --- | --- |
| `λ_{1..6}` | `(0.3, 0.5, 0.1, 0.4, 0.6, 0.2)` |
| `η_{1..3}` | `(1.0, 0.5, 0.3)` |
| `λ_civ` | `0.1` |
| `α_MIL` | `0.6` |
| `K`, `T`, `m` | `96`, `64`, `64` |
| `Δ` | `{1, 2, 4, 8}` |
| Distillation `T` | `4.0` |
| Exemplars / class | `20` |
| Optimiser | `AdamW(lr=2e-4, β=(0.9, 0.999), wd=1e-4)` |
| Schedule | `50 + 30 + 20` epochs, cosine, warmup 5 |
| Hard-concrete `τ_g` | anneal `0.5 → 0.1` (first 30 epochs) |
| Modality dropout `p_m` | anneal `0.3 → 0.05` (first 20% steps) |
| Batch | `32` (16 normal + 16 abnormal) |
| Seeds | `{42, 123, 2024, 7, 31}` |

### 2. Hardware

| Component | Value |
| --- | --- |
| GPUs | 2× NVIDIA A100-80GB |
| Precision | bf16 (fp16 fallback) |
| Distributed | DDP via `torchrun --nproc_per_node=2`, NCCL backend |

### 3. Splits

`scripts/build_splits.py` writes `data/<name>/annotations/*_list.txt` with the exact cardinality of Appendix Table I (1610/290, 3954/800, 238/199, 268/64/211).

### 4. Sanity test

```bash
python scripts/smoke_test.py
```

Asserts:

- ✅ Composite loss aggregates 7 + `L_civ` terms.
- ✅ TD-EAIM forward respects Eq. 7 (linear part is *not* gated).
- ✅ `PseudoLabelLoss` populates the right indices.
- ✅ `L_inv` flows gradient into β.
- ✅ `P_known` is a real orthogonal projector (`‖P² − P‖ < 1e-4`).
- ✅ EMA teacher concept-intervention loss is finite and differentiable.

## Hardware

Recommended:

- 2× NVIDIA A100-80GB (paper setup)
- 1 TB NVMe scratch (raw videos + extracted features)
- ≥ 64 GB system RAM
- Linux kernel ≥ 5.10 with NCCL 2.18+

Minimum (for ablation / smoke testing):

- 1× NVIDIA RTX 3090 / 4090 or equivalent
- 100 GB scratch

## Project structure

| Path | Purpose |
| --- | --- |
| `configs/default.py` | OmegaConf default config (single source of truth for paper numbers). |
| `models/eim_owilnet.py` | Top-level network; EMA teacher; Algorithm 2 cluster seeding. |
| `models/td_eaim.py` | TD-EAIM operator; hard-concrete gate; MI candidate selection. |
| `models/concept_bottleneck.py` | 96-concept encoder + post-hoc temperature calibration. |
| `models/open_world_head.py` | Posterior / energy / SVD-orthogonal residual fusion. |
| `models/incremental_learner.py` | Frozen-teacher distillation, three replay memories. |
| `models/cross_dataset_invariance.py` | GRL + variance-of-β invariance loss. |
| `losses/composite_loss.py` | Eight-term objective (7 paper terms + `L_civ`). |
| `datasets/video_dataset.py` | Unified loader + dataset-local class spaces. |
| `metrics/evaluation.py` | All five metric families. |
| `scripts/launch_a100.sh` | 2× A100 launcher. |
| `scripts/smoke_test.py` | Sub-30 s end-to-end test. |

## Roadmap

- [x] v1.0.0 – Paper-aligned release with eight-term loss, EMA teacher, Algorithm 2 cluster seeding, DDP on 2× A100-80GB.
- [ ] v1.1.0 – Pre-extracted feature releases for UCF-Crime and XD-Violence (HuggingFace).
- [ ] v1.2.0 – Pretrained checkpoints + W&B logging artefacts for the five seeds.
- [ ] v1.3.0 – ONNX export and inference container (CUDA / CPU).
- [ ] v1.4.0 – Live demo notebook with concept-intervention sliders.

Open an issue / PR if you would like a feature surfaced earlier.

## Contributing

Contributions are warmly welcome. Please:

1. Fork the repository and create a topic branch off `main`.
2. Run `python scripts/smoke_test.py` and add tests for any new behaviour.
3. Format Python with `black` (line length 100) and lint with `ruff`.
4. Open a pull request describing the change, the paper section it touches (if any), and a small benchmark when applicable.

For larger refactors, please open a discussion / RFC issue first so we can align on direction.

### Reporting bugs

When opening an issue please include:

- Python / PyTorch / CUDA version
- Exact command that triggered the issue
- Minimal repro (the smoke test is a good template)
- Full traceback / log

### Code of conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for a per-release breakdown. Recent highlights:

- v1.0.0 — Eight-term loss, EMA teacher, Algorithm 2 cluster seeding on the unlabeled training stream, SVD-orthogonal residual head, EMA prototypes, DDP-ready launcher.

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

A [`CITATION.cff`](CITATION.cff) is provided for tooling that prefers the GitHub citation format. Click *Cite this repository* on the GitHub sidebar to copy a BibTeX entry.

## License

Code is released under the [MIT License](LICENSE).

Datasets retain their original licenses; please consult their respective homepages before redistributing any video frames or annotations:

| Dataset | License |
| --- | --- |
| UCF-Crime | Research-only (Sultani et al., CVPR 2018) |
| XD-Violence | Research-only (Wu et al., ECCV 2020) |
| ShanghaiTech | Research-only (Liu et al., CVPR 2018) |
| UBnormal | CC-BY-4.0 (Acsintoae et al., CVPR 2022) |

## Acknowledgements

EIM-OWILNet builds on top of the open VideoMAE, OpenAI CLIP, and Ultralytics YOLO ecosystems, and benefits from prior work in weakly-supervised VAD (RTFM, MGFN, UR-DMU, VadCLIP, OVVAD, PEL4VAD, GS-MoE, DSANet, APST-Net) and explainable AI (Harsanyi dividend, hard-concrete gating). We thank the maintainers of the four benchmarks for keeping their data accessible to the research community.

---

<div align="center">
<sub>Made with ❤️  for reproducible video understanding research.</sub>
</div>
