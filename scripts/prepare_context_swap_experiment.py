"""Prepare a minimally patched frozen 03_500K_state4_recurrent source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--injection", choices=("recurrent_concat", "dali_deep", "dali_deep_joint", "dali_shallow"), default="dali_deep")
    parser.add_argument("--swap-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.swap_weight < float("inf"):
        parser.error("Swap weight must be finite and nonnegative")
    root = Path(__file__).resolve().parents[1]
    base = root / "logdir/context_kinematics_500k_20261015_180340/source"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "prepared_source"
    shutil.copytree(base, source, ignore=shutil.ignore_patterns("__pycache__"))
    # Snapshot explicit modules rather than replaying a moving git diff against
    # an older source. Actual changes against the baseline are recorded below.
    for name in ("context.py", "context_rssm.py", "dreamer.py", "envs/metadrive.py"):
        shutil.copy2(root / name, source / name)
    config = source / "configs/model/_base_.yaml"
    text = config.read_text()
    assert text.count("  aux_weight: 1.0") == 1
    text = text.replace("  aux_weight: 1.0", f"  aux_weight: 1.0\n  swap_weight: {args.swap_weight}")
    text = text.replace("  injection: recurrent_concat", f"  injection: {args.injection}")
    text += f"\n  actor_critic: {str(args.injection in ('dali_deep', 'dali_deep_joint')).lower()}\n"
    config.write_text(text)
    shutil.copy2(root / "configs/metadrive_context_swap_deep1.yaml", source / "configs/metadrive_context_swap_deep1.yaml")
    shutil.copy2(root / "configs/metadrive_context_swap_dali_joint.yaml", source / "configs/metadrive_context_swap_dali_joint.yaml")
    shutil.copy2(root / "configs/metadrive_context_swap_dali_shallow.yaml", source / "configs/metadrive_context_swap_dali_shallow.yaml")
    # Latest/checkpoint instrumentation fixes are independent of model behavior.
    for name in ("trainer.py", "train.py"):
        shutil.copy2(root / name, source / name)
    hashes = {}
    patch = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(source))
        original = base / rel
        if not original.exists() or path.read_bytes() != original.read_bytes():
            patch.extend(difflib.unified_diff(
                original.read_text().splitlines(keepends=True) if original.exists() else [],
                path.read_text().splitlines(keepends=True), fromfile="a/" + rel, tofile="b/" + rel))
            hashes[rel] = {
                "baseline": hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,
                "experiment": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (output / "provenance.json").write_text(json.dumps({
        "baseline": str(base), "prepared_source": str(source), "changed_files": hashes,
        "injection": args.injection, "swap_weight": args.swap_weight,
        "design": "State4 residual model with current context plumbing; "
                  "exact model/mass/max-steering match across true episodes; full histories only. "
                  "Deep uses recurrent conditioning, prediction heads and actor/critic; Shallow uses posterior only; "
                  "prior/posterior condition indirectly via h. Dynamic rollout is retained.",
        "checkpoint_note": "Current trainer/train checkpoint instrumentation; default 20K latest saves.",
    }, indent=2) + "\n")
    (output / "swap.patch").write_text("".join(patch))
    print(source)


if __name__ == "__main__":
    main()
