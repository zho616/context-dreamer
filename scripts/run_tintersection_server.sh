#!/usr/bin/env bash
# Reproducible fresh training; the Python runner freezes code per experiment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
variant="both"
gpu="0"
output=""
steps="500000"
smoke="false"
dry_run="false"

while (($#)); do
  case "$1" in
    --variant) variant="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --smoke) smoke="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    --help|-h)
      echo "Usage: $0 --output FRESH_DIRECTORY [--variant both|baseline|shallow] [--gpu 0] [--steps 500000] [--smoke] [--dry-run]"
      echo "Optional environment: PYTHON_BIN, METADRIVE_ROOT. One GPU: both sequentially; separate GPUs: two separate variants/outputs."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$output" ]] || { echo "--output is required" >&2; exit 2; }
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be positive" >&2; exit 2; }
[[ "$variant" == "both" || "$variant" == "baseline" || "$variant" == "shallow" ]] || {
  echo "--variant must be both, baseline or shallow" >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ -n "${METADRIVE_ROOT:-}" ]]; then
  [[ -d "$METADRIVE_ROOT/metadrive" ]] || { echo "METADRIVE_ROOT must contain metadrive/" >&2; exit 2; }
  export PYTHONPATH="$METADRIVE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
cd -- "$repo_dir"
"$python_bin" "$script_dir/server_preflight.py"
command=("$python_bin" -u "$script_dir/train_metadrive_compare.py"
  --logdir "$output" --source-from "$repo_dir" --methods r2dreamer
  --map-block T --chassis-randomization --mass-friction-randomization --steps "$steps")
if [[ "$variant" != "baseline" ]]; then
  command+=(--context-variants dali_shallow --context-swap-weight 1)
fi
if [[ "$variant" == "both" ]]; then command+=(--include-baseline); fi
if [[ "$smoke" == "true" ]]; then command+=(--smoke); else command+=(--evaluate-best); fi
if [[ "$dry_run" == "true" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
