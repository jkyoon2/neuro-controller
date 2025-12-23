#!/bin/bash
# VAE Multi-Task Training Script
# - All configurations are in vae_config.yaml
# - Simply edit the YAML file to change training parameters

set -e  # Exit on error

# ========================================================================
# Configuration
# ========================================================================
PROJECT_ROOT="/home/juliecandoit98/neurocontroller"
CONFIG_FILE="${PROJECT_ROOT}/zsceval/scripts/overcooked/config/vae_config.yaml"

# Optional: Override config file
# CONFIG_FILE="path/to/custom_config.yaml"

# Optional: Quick CLI overrides (for testing)
EPOCHS=""          # e.g., "--epochs 10"
BATCH_SIZE=""      # e.g., "--batch-size 32"
DEVICE=""          # e.g., "--device cpu"
SAVE_PATH=""       # e.g., "--save-path ./my_vae.pt"

# ========================================================================
# Execution
# ========================================================================
echo "=========================================="
echo "VAE Multi-Task Training"
echo "Started at: $(date)"
echo "=========================================="
echo ""
echo "Configuration file: ${CONFIG_FILE}"
echo ""

# Build command
CMD="python zsceval/scripts/overcooked/train/train_vae.py --config ${CONFIG_FILE}"

# Add overrides if specified
[ -n "${EPOCHS}" ] && CMD="${CMD} ${EPOCHS}"
[ -n "${BATCH_SIZE}" ] && CMD="${CMD} ${BATCH_SIZE}"
[ -n "${DEVICE}" ] && CMD="${CMD} ${DEVICE}"
[ -n "${SAVE_PATH}" ] && CMD="${CMD} ${SAVE_PATH}"

echo "Command: ${CMD}"
echo ""
echo "=========================================="
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Execute
${CMD}

# ========================================================================
# Completion
# ========================================================================
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training Completed Successfully"
    echo "Finished at: $(date)"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "✗ Training Failed"
    echo "Finished at: $(date)"
    echo "=========================================="
    exit 1
fi
