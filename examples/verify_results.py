"""Verify retained checksums and recompute headline metrics without model weights."""

import argparse
import json
from pathlib import Path

import numpy as np

from ifllm_learn.calibration import apply_prediction_calibration
from ifllm_learn.evidence import read_predictions
from ifllm_learn.metrics import metrics, prediction_arrays
from ifllm_learn.schema import file_sha256


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "SHA256SUMS.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_sha256(directory / name) != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    report = json.loads((directory / "report.json").read_text())
    for filename, section, temperature_file in (
        ("base-test.jsonl", "base", None),
        ("trained-test.jsonl", "trained", None),
        ("trained-test.jsonl", "trained_calibrated", "temperature.json"),
        ("base-test.jsonl", "base_calibrated", "base-temperature.json"),
    ):
        if section not in report:
            continue  # Historical Bitcoin v1 did not measure base calibration.
        predictions = read_predictions(directory / filename)
        temperature = 1.0
        if temperature_file:
            artifact = json.loads((directory / temperature_file).read_text())
            temperature = artifact["temperature"]
            calibrated = apply_prediction_calibration(predictions, artifact)
            retained_name = "base-calibrated-test.jsonl" if section == "base_calibrated" else "calibrated-test.jsonl"
            retained = read_predictions(directory / retained_name)
            if [r["id"] for r in calibrated] != [r["id"] for r in retained]:
                raise ValueError("Calibrated prediction IDs mismatch")
            for a, b in zip(calibrated, retained):
                np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-10, rtol=0)
        values, labels = prediction_arrays(predictions)
        observed = metrics(values, labels, temperature=temperature)
        for key in ("count", "accuracy", "macro_f1", "log_loss", "brier", "ece_10_bins"):
            if not np.isclose(observed[key], report[section][key], atol=1e-10, rtol=0):
                raise ValueError(f"{directory.name} {section}.{key} does not match retained predictions")
        if observed["confusion_matrix"] != report[section]["confusion_matrix"]:
            raise ValueError("Confusion matrix mismatch")
    return {"dataset": directory.name, "files_verified": len(manifest["files"]), "test_rows": report["base"]["count"],
            "base_accuracy": report["base"]["accuracy"], "trained_accuracy": report["trained"]["accuracy"],
            "trained_calibrated_nll": report["trained_calibrated"]["log_loss"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    args = parser.parse_args()
    directories = args.directories or [Path("results") / name for name in ("banking", "clinc", "contractnli", "bitcoin")]
    print(json.dumps([verify(directory) for directory in directories], indent=2))


if __name__ == "__main__":
    main()
