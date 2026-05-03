"""Default config (paper Sec. IV-B + Appendix Table II).

Hardware target: 2x NVIDIA A100-80GB (paper Sec. IV-B).
"""

from omegaconf import OmegaConf


def get_default_config():
    cfg = OmegaConf.create({
        # ---------- general ----------
        "seed": 42,
        "device": "cuda",
        "num_workers": 8,
        # A100 supports both fp16 and bf16; bf16 is more numerically stable.
        "fp16": True,
        "bf16": False,
        "distributed": True,           # 2x A100-80GB DDP
        "world_size": 2,               # paper hardware: 2 GPUs
        "find_unused_parameters": False,

        # ---------- data ----------
        "data": {
            "dataset": "ucf_crime",
            "data_root": "./data",
            "snippet_len": 32,
            "num_snippets_train": 64,
            "num_snippets_test": 0,
            "stride": 8,
            "frame_size": 224,
            # 13 anomaly classes + 1 normal => 14 logit dims (paper UCF protocol)
            "num_classes": 14,
            # Open-world Seen-8 / Unseen-5 (paper Appendix Table I).
            # Indices follow the alphabetical UCF-Crime ordering in
            # datasets.video_dataset.UCF_CLASS_NAMES:
            #   0=Abuse, 1=Arrest, 2=Arson, 3=Assault, 4=Burglary,
            #   5=Explosion, 6=Fighting, 7=RoadAccidents, 8=Robbery,
            #   9=Shooting, 10=Shoplifting, 11=Stealing, 12=Vandalism, 13=Normal.
            "seen_classes":   [0, 1, 2, 3, 4, 6, 7, 8],  # Abuse, Arrest, Arson,
                                                         # Assault, Burglary, Fighting,
                                                         # RoadAccidents, Robbery
            "unseen_classes": [5, 9, 10, 11, 12],        # Explosion, Shooting,
                                                         # Shoplifting, Stealing, Vandalism
            # Per-axis split sizes (Appendix Table I).
            "split_sizes": {
                "ucf_crime": {"train": 1610, "test": 290},
                "xd_violence": {"train": 3954, "test": 800},
                "shanghai_tech": {"train": 238, "test": 199},
                "ubnormal": {"train": 268, "val": 64, "test": 211},
            },
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        },

        # ---------- feature extractor ----------
        "feature_extractor": {
            "video_backbone": "videomae_l_k400",
            "video_embed_dim": 1024,
            "clip_model": "ViT-L/14",
            "clip_embed_dim": 768,
            "motion_encoder": "yolo_raft",
            "motion_embed_dim": 4096,
            "fusion_method": "paper_sum",
            "fused_dim": 1024,
            "freeze_video": True,
            "freeze_clip": True,
            "freeze_motion": True,
            "modality_dropout_start": 0.3,
            "modality_dropout_end": 0.05,
            "modality_dropout_anneal_fraction": 0.2,   # 20% of training steps
        },

        # ---------- concept bottleneck ----------
        "concept_bottleneck": {
            "num_concepts": 96,
            "concept_dim": 256,
            "concept_types": ["action", "object", "scene", "dynamic"],
            "hidden_dim": 64,
            "dropout": 0.1,
            "text_embeddings_path": "./concept_text_embeddings.pt",
            "require_text_embeddings": False,
            "calibration_clips": 200,         # post-hoc temperature scaling clips
            "calibration_steps": 200,
            "calibration_lr": 1e-2,
        },

        # ---------- TD-EAIM ----------
        "td_eaim": {
            "num_concepts": 96,
            "num_and_pairs": 256,
            "num_or_pairs": 256,
            "num_temporal_pairs": 64,
            "temporal_offsets": [1, 2, 4, 8],
            "sparse_lambda": 0.01,
            "gate_threshold": 0.1,
            "hidden_dim": 64,
            "hard_concrete_temp": 0.5,
            "hard_concrete_temp_end": 0.1,
            "hard_concrete_anneal_epochs": 30,
            "pair_selection": "mutual_info",
            "mi_max_samples": 8192,
        },

        # ---------- VAD head ----------
        "vad_head": {
            "margin": 1.0,
            "top_k_ratio": 0.1,
            "dynamic_top_k": True,
            "alpha_topk": 0.6,
        },

        # ---------- open world ----------
        "open_world": {
            "energy_threshold": -1.0,
            "confidence_threshold": 0.5,
            "residual_threshold": 1.0,
            "residual_loss_weight": 0.1,
            "num_prototypes": 14,
            "proto_dim": 64,
            "prototype_ema_decay": 0.9,
            "cluster_method": "hdbscan",
            "eps": 0.5,
            "min_samples": 5,
        },

        # ---------- incremental learning ----------
        "incremental": {
            "memory_size": 500,
            "exemplar_per_class": 20,
            "eta_distill": 1.0,
            "eta_coef": 0.5,
            "eta_proto": 0.3,
            "eta_logit": 1.0,
            "distill_temperature": 4.0,
            "num_tasks": 5,
            # Paper main.tex: T_3 += ShanghaiTech, T_4 += UBnormal.
            # Each entry: { "dataset": <name>, "classes": [int idx within
            # that dataset]} so the loader knows where to read videos.
            # task_classes are interpreted in the *dataset-local* anomaly
            # class space written by scripts/build_splits.py.
            # ShanghaiTech is binary so its only anomaly class is 0.
            # UBnormal uses 0..12 for the 13 anomaly types listed in
            # video_dataset.UB_CLASS_NAMES.
            "task_specs": [
                {"dataset": "ucf_crime",     "classes": [0, 1, 2, 3]},
                {"dataset": "ucf_crime",     "classes": [4, 5, 6]},
                {"dataset": "ucf_crime",     "classes": [7, 8, 9]},
                {"dataset": "shanghai_tech", "classes": [0]},
                {"dataset": "ubnormal",      "classes": list(range(13))},
            ],
            # Legacy (Appendix Table I, UCF-only flat indices). Used as
            # fallback when a single-dataset incremental schedule is
            # requested via --schedule=ucf_only.
            "task_classes": [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9],
                              [10, 11, 12], [13]],
        },

        # ---------- cross-dataset invariance ----------
        "cross_dataset": {
            "lambda_inv": 0.1,
            "lambda_domain": 0.1,
            "domain_classifier_hidden": 256,
            "num_domains": 4,
            # Paper transfer pairs (Sec. IV-D Table III).
            "transfer_pairs": [
                ["ucf_crime", "xd_violence"],
                ["ucf_crime", "shanghai_tech"],
                ["xd_violence", "ucf_crime"],
                ["ucf_crime+xd_violence", "ubnormal"],
            ],
        },

        # ---------- training ----------
        # 2x A100-80GB; AdamW (beta1=0.9, beta2=0.999, wd=1e-4); 50+30+20.
        "train": {
            "epochs": 50,
            "stage2_epochs": 30,
            "stage3_epochs": 20,
            "batch_size": 32,
            "per_gpu_batch_size": 16,         # 16 per A100-80GB
            "lr": 2e-4,
            "backbone_lr": 1e-5,
            "weight_decay": 1e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "warmup_epochs": 5,
            "scheduler": "cosine",
            "grad_clip": 1.0,
            "accumulate_grad_batches": 1,
            "paired_batch": True,
            "ema_decay": 0.99,
        },

        # ---------- losses ----------
        # paper: L_total = L_MIL + lambda1 L_pseudo + lambda2 L_int
        #                   + lambda3 L_sparse + lambda4 L_open
        #                   + lambda5 L_inc + lambda6 (L_inv + L_domain)
        #                   + lambda_civ L_civ
        "losses": {
            "lambda_mil": 1.0,
            "lambda_pseudo": 0.3,
            "lambda_int": 0.5,
            "lambda_sparse": 0.1,
            "lambda_open": 0.4,
            "lambda_inc": 0.6,
            "lambda_inv": 0.2,
            "lambda_civ": 0.1,
        },

        # ---------- evaluation ----------
        "eval": {
            "metrics": ["roc_auc", "ap", "far_at_0_5"],
            "open_world_metrics": ["known_auc", "unknown_auroc", "oscr", "h_score", "nmi"],
            "incremental_metrics": ["avg_auc", "bwt", "fwt", "forget", "mem_mb"],
            "explanation_metrics": ["drop_at_k", "aopc", "suff", "comp", "insdel"],
        },

        # ---------- logging ----------
        "logging": {
            "log_dir": "./logs",
            "checkpoint_dir": "./checkpoints",
            "save_every": 5,
            "eval_every": 1,
            "log_every": 10,
        },

        # ---------- cross validation ----------
        "cross_val": {
            "num_runs": 5,
            "seeds": [42, 123, 2024, 7, 31],
            "run_name": "eim_owilnet_cv",
        },
    })
    return cfg
