"""Snapshot a whitelist of lightweight experiment records; never copy checkpoints."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "01_500K_baseline": "chassis_baselines_500k_20261015_020330/r2dreamer",
    "dreamerv3_500K_baseline": "chassis_baselines_500k_20261015_020330/dreamerv3",
    "03_500K_state4_recurrent": "context_kinematics_500k_20261015_180340/r2dreamer_context",
    "09_500K_deep_swap1": "context_swap_deep1_500k_20261020/training/r2dreamer_dali_deep",
    "10_500K_DALI_joint_swap1": "context_swap_dali_joint_500k_20261020_run1/training/r2dreamer_dali_deep_joint",
    "11_500K_DALI_shallow_swap1": "context_swap_dali_shallow_500k_20261021/training/r2dreamer_dali_shallow",
    "T_mass_friction_baseline": "tintersection_mass_friction_500k_20261021/training/r2dreamer",
    "T_mass_friction_shallow": "tintersection_mass_friction_500k_20261021/training/r2dreamer_dali_shallow",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "snapshot_20261021")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory to avoid mixing snapshots.")
    entries = []

    def copy(src, relative):
        if not src.is_file():
            return
        # Read once: the last metrics line can be incomplete while training runs.
        data = src.read_bytes()
        if src.suffix == ".jsonl":
            lines = data.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):
                lines.pop()
            data = b"".join(lines)
        dest = args.output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entries.append({"path": str(relative), "source": str(src.relative_to(ROOT)),
                        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    for label, run in RUNS.items():
        source = ROOT / "logdir" / run
        for name in ("metrics.jsonl", ".hydra/config.yaml", ".hydra/overrides.yaml"):
            copy(source / name, Path("runs") / label / name)
        for path in (source / "evaluation").glob("*/*"):
            if path.name in {"summary.json", "provenance.json", "episodes.jsonl"}:
                copy(path, Path("runs") / label / path.relative_to(source))

    extras = [
        ("chassis_baselines_500k_20261015_020330/best_eval100_20261015_095447", "baseline_eval"),
        ("context_kinematics_500k_20261015_180340/comparison_20261016_011017", "03_eval"),
        ("world_model_injection_comparison_20261021", "world_model"),
        ("context_exchange_prediction_20261021", "context_exchange"),
        ("driving_stability_20261021", "driving_stability"),
        ("training_curves_01_vs_11_20261021/tensorboard_ema099", "training_curves_ema099"),
    ]
    allowed = {"summary.json", "provenance.json", "episodes.jsonl", "REPORT.md", "README.md",
               "CONDITION_REPORT.md", "comparison.json", "results.json", "condition_results.json",
               "source_data.csv", "tensorboard_check.json", "selected_episode.json"}
    for directory, label in extras:
        source = ROOT / "logdir" / directory
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            # Exclude dozens of individual case images; keep the selected full episode.
            if "examples" in relative.parts or "selected_controls" in relative.parts:
                continue
            figure = (path.suffix in {".png", ".pdf", ".svg"} and
                      (label == "training_curves_ema099" or "full_episode" in relative.parts
                       or path.stem == "stability_overview"))
            if path.name in allowed or figure:
                copy(path, Path("analysis") / label / relative)
    source = ROOT / "logdir/tintersection_mass_friction_500k_20261021"
    for name in ("training/status.json", "validation.json", "unit_tests.log"):
        copy(source / name, Path("T_validation") / name)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "note": "Snapshot only. T shallow may still be running; metrics keep complete lines only. Source paths refer to the original local logdir. Checkpoints, event files and raw trajectory arrays are excluded.",
                "files": entries}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(entries)} files, {sum(x['bytes'] for x in entries)/2**20:.2f} MiB to {args.output}")


if __name__ == "__main__":
    main()
