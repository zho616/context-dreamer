#!/usr/bin/env bash
# Edit the settings below, then run: bash scripts/test_metadrive_render.sh
set -euo pipefail

# ========= Model and rendering settings =========
# Paths can be absolute or relative to the project root.
CHECKPOINT="logdir/compare_roundabout_500k_20261013_170401/r2dreamer/latest.pt"

# Model interface: edit load_model/reset/act in this module to test your model.
# Use "dreamer" for the built-in loader, or "your_module:your_factory" for your own.
POLICY="scripts.metadrive_policy_template:load_policy"

# Leave empty to load CHECKPOINT's sibling .hydra/config.yaml automatically.
CONFIG=""
RENDER="both"                      # both | 3d | topdown | none
FPS="10"                           # Display pacing only; 0 = no delay
OUTPUT="logdir/render_test/latest_test.json"

# Leave empty to inherit the model's training evaluation settings.
EPISODES=""
SEED_START=""
DEVICE=""
MAX_STEPS=""                       # Empty = training episode horizon

# Optional topdown PNG output. NO_WINDOW=true requires topdown or none.
SAVE_TOPDOWN=""
NO_WINDOW=false

# Existing Conda environment; no activation/install is needed.
PYTHON_BIN="/home/server/anaconda3/envs/r2dreamer/bin/python"
# ===============================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found: $PYTHON_BIN. Edit PYTHON_BIN in this script." >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT. Edit CHECKPOINT in this script." >&2
    exit 1
fi
if [[ -z "$CONFIG" ]]; then
    CONFIG="$(dirname -- "$CHECKPOINT")/.hydra/config.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Training config not found: $CONFIG. Set CONFIG to the model's training YAML." >&2
    exit 1
fi

args=("$SCRIPT_DIR/test_metadrive_render.py"
      --checkpoint "$CHECKPOINT" --policy "$POLICY" --config "$CONFIG"
      --render "$RENDER" --fps "$FPS" --output "$OUTPUT")
[[ -z "$EPISODES" ]] || args+=(--episodes "$EPISODES")
[[ -z "$SEED_START" ]] || args+=(--seed-start "$SEED_START")
[[ -z "$DEVICE" ]] || args+=(--device "$DEVICE")
[[ -z "$MAX_STEPS" ]] || args+=(--max-steps "$MAX_STEPS")
[[ -z "$SAVE_TOPDOWN" ]] || args+=(--save-topdown "$SAVE_TOPDOWN")
if [[ "$NO_WINDOW" == true ]]; then
    args+=(--no-window)
fi

echo "Policy: $POLICY"
echo "Checkpoint: $CHECKPOINT"
echo "Training configuration: $CONFIG"
exec "$PYTHON_BIN" "${args[@]}"
