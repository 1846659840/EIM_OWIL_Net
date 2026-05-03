#!/usr/bin/env bash
# Launch EIM-OWILNet on the paper hardware (2x NVIDIA A100-80GB).
#
# Usage:
#   bash scripts/launch_a100.sh standard
#   bash scripts/launch_a100.sh open_world
#   bash scripts/launch_a100.sh incremental
#   bash scripts/launch_a100.sh cross_dataset
#
# Notes:
#   * Distributed via torchrun; world_size=2 by default, override with
#     NPROC=N when needed.
#   * Mixed precision uses bf16 by default (A100 native); switch with
#     OMC_BF16=0 to fall back to fp16.

set -e

PROTOCOL="${1:-standard}"
NPROC="${NPROC:-2}"
SEED="${SEED:-42}"
USE_BF16="${OMC_BF16:-1}"

PYTHONPATH="$(pwd)" \
torchrun --standalone --nproc_per_node="${NPROC}" \
    train.py \
    --protocol "${PROTOCOL}" \
    --seed "${SEED}" \
    bf16=${USE_BF16}
