"""Reproduce the frozen text demos. Run from the repository root on Apple Silicon.

python examples/run_demo_suite.py --output-root runs/reproduced --evidence-root results/reproduced
No hyperparameter search; every artifact destination is create-only.
"""

import argparse
import json
from pathlib import Path

from ifllm_learn.evidence import export_run, verify_run
from ifllm_learn.textdata import prepare_text_dataset
from ifllm_learn.workflow import load_splits, run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.output_root.exists() or args.evidence_root.exists():
        parser.error("Output/evidence roots must be new; partial runs are retained, never overwritten")
    specification = json.loads((Path(__file__).parent / "demo-suite.json").read_text())
    for name, task in specification["datasets"].items():
        data = args.data_root / f"{name}-demo"
        if not data.exists():
            prepare_text_dataset(name, data, args.data_root / "cache" / "text",
                                 seed=specification["seed"], max_tokens=specification["shared"]["max_tokens"])
        else:
            load_splits(data)
            manifest = json.loads((data / "manifest.json").read_text())
            if manifest.get("name") != name or manifest.get("seed") != specification["seed"] or manifest.get("max_tokens") != specification["shared"]["max_tokens"]:
                raise ValueError(f"Existing dataset {data} differs from frozen suite settings")
        run = args.output_root / name
        report = run_workflow(data, run, seed=specification["seed"], **specification["shared"], **task)
        verify_run(run)
        export_run(run, data, args.evidence_root / name)
        print(json.dumps({"dataset": name, "base_accuracy": report["base"]["accuracy"],
                          "trained_accuracy": report["trained"]["accuracy"], "report": str(run / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
