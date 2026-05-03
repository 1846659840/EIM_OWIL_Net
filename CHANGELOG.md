# Changelog

All notable changes to EIM-OWILNet are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses Semantic Versioning.

## [1.0.0] - 2026-05-03

### Added
- Concept-intervention loss `L_civ` (Eq. 3) with EMA teacher copy of the model.
- Algorithm 2 cluster-seeding helper that initialises new-class prototypes from HDBSCAN on the unlabeled training stream.
- Post-hoc temperature calibration for the concept bottleneck (`concept_bottleneck.calibrate_temperature`).
- 2× A100-80GB DDP launcher (`scripts/launch_a100.sh`) with bf16 autocast.
- Paper-canonical split builder (`scripts/build_splits.py`) and dataset download manifest (`scripts/download_datasets.py`).
- End-to-end smoke test (`scripts/smoke_test.py`) covering eight loss terms, P_known orthogonality, and gradient flow.

### Changed
- TD-EAIM now keeps the linear first-order part outside the hard-concrete gate (Eq. 7) and exposes `weighted_first` / `weighted_higher` separately.
- Open-world residual head replaces the previous softmax-weighted reconstruction with a true SVD-orthogonal projector `P = U U^T`.
- Known-anomaly prototypes are EMA-updated class means (`register_buffer`) instead of free parameters.
- Cross-dataset invariance loss now keeps the current domain's β as a differentiable stack entry, so gradients reach the interaction weights.
- Incremental learner uses a real frozen-eval deepcopy of the network as the teacher; KD compares teacher and student on the *same* input.
- Open-world CE loss is computed once at the snippet level inside `CompositeLoss`; the duplicate computation in `EIMOWILNet.forward` was removed.
- `InteractionFaithfulnessLoss` thresholds the closed-form `gate_prob > 0.5` instead of the stochastic `gate_mask > 0`.
- `seed_new_prototypes_from_clustering` is invoked on `train_loader` (was `test_loader`) to remove the test-set leakage.
- ShanghaiTech and UBnormal use dataset-local class indices (SHT 0; UB 0..12) so `task_specs` filters are non-empty for T3 and T4.
- Open-world Seen-8 / Unseen-5 indices in `configs/default.py` are realigned with `UCF_CLASS_NAMES` (Seen = `[0,1,2,3,4,6,7,8]`, Unseen = `[5,9,10,11,12]`).

### Fixed
- `PseudoLabelLoss` no longer drops dynamic top-k pseudo labels through a fancy-indexed `scatter_` copy.
- Modality dropout `p_m` anneals over the first 20 % of *training steps* (was epochs).
- `vad_head.WeaklySupervisedVADHead` no longer applies a second sigmoid MLP after the TD-EAIM score.
- `Conf_int(V)` for the dynamic MIL is now computed only over active interactions (`g_S > 0`).
- `train.init_distributed` enforces `bf16 ⊕ fp16` to avoid GradScaler/autocast conflicts on A100.
- `DistributedSampler` is wired through `create_dataloaders`, `create_cross_dataset_loaders`, and `create_incremental_loaders`; per-rank batch size = `batch_size / world_size`.
- `save_old_model` is invoked at the end of each incremental task (was at the start).
