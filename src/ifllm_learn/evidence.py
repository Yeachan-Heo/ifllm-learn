"""Real adapter reload checks and schema-restricted, integrity-bound evidence exports."""

import json
from pathlib import Path
import numpy as np

from .calibration import apply_prediction_calibration
from .schema import file_sha256, load_jsonl, write_json, write_jsonl

CORE_BOUND_FILES = (
    "run.json", "test-input.jsonl", "trained-test.jsonl", "calibrated-test.jsonl", "temperature.json",
    "adapter/adapters.safetensors", "adapter/adapter_config.json", "adapter/training.json", "report.json",
    "base-test.jsonl", "trained-calibration.jsonl",
)
V2_COMPARISONS = ("base-temperature.json", "base-calibration.jsonl", "base-calibrated-test.jsonl", "tfidf-baseline.json")


def read_predictions(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected nonempty prediction JSONL")
    return rows


def _bound_files(root):
    report = json.loads((root / "report.json").read_text())
    version = report.get("format")
    if version not in ("ifllm-learn.experiment.v1", "ifllm-learn.experiment.v2"):
        raise ValueError("Unsupported experiment report format")
    files = list(CORE_BOUND_FILES)
    if version == "ifllm-learn.experiment.v2":
        files.extend(V2_COMPARISONS)
    return files


def verify_run(run_dir):
    from .inference import load_backend, predict_rows

    root = Path(run_dir)
    output = root / "reload-verification.json"
    if output.exists() or (root / "reloaded-test.jsonl").exists():
        raise FileExistsError("Verification outputs must be new; preserve earlier receipts separately")
    bound_files = _bound_files(root)
    before = {name: file_sha256(root / name) for name in bound_files}
    config = json.loads((root / "run.json").read_text())
    rows = load_jsonl(root / "test-input.jsonl")
    if [row["id"] for row in rows] != config["selected_ids"]["test"]:
        raise ValueError("Reload inputs differ from the declared test selection")
    expected = read_predictions(root / "trained-test.jsonl")
    actual = predict_rows(load_backend(root / "adapter", config["bits"]), rows, config["max_tokens"])
    if [row["id"] for row in actual] != [row["id"] for row in expected]:
        raise ValueError("Reload prediction IDs differ")
    for source_row, observed, retained in zip(rows, actual, expected):
        for key in ("model_revision", "adapter_sha256", "precision", "contract_sha256", "option_ids", "prompt_sha256", "label"):
            if key not in retained or observed.get(key) != retained[key]:
                raise ValueError(f"Reload model/input identity mismatch: {key}")
        if source_row["label"] != retained["label"]:
            raise ValueError("Reload input label differs from the measured label")
    if [row["answer"] for row in actual] != [row["answer"] for row in expected]:
        raise ValueError("Reload changed predicted answers")
    delta = max(float(np.max(np.abs(np.asarray(a["logits"]) - b["logits"]))) for a, b in zip(actual, expected))
    if not np.isfinite(delta) or delta > 1e-4:
        raise ValueError(f"Reload logits differ by {delta}")
    artifact = json.loads((root / "temperature.json").read_text())
    calibrated = apply_prediction_calibration(actual, artifact)
    expected_calibrated = read_predictions(root / "calibrated-test.jsonl")
    if [row["id"] for row in calibrated] != [row["id"] for row in expected_calibrated]:
        raise ValueError("Calibrated reload IDs differ")
    if not all(np.allclose(a["probabilities"], b["probabilities"], atol=1e-6, rtol=0)
               for a, b in zip(calibrated, expected_calibrated)):
        raise ValueError("Calibrated reload probabilities differ")
    after = {name: file_sha256(root / name) for name in bound_files}
    if before != after:
        raise ValueError("Run artifacts changed during reload verification")
    write_jsonl(root / "reloaded-test.jsonl", actual)
    receipt = {"format": "ifllm-learn.reload.v2", "rows": len(rows), "max_logit_difference": delta,
               "answers_identical": True, "calibrated_probabilities_match": True,
               "adapter_sha256": actual[0]["adapter_sha256"], "run_sha256": after["run.json"],
               "verified_sha256": after}
    write_json(output, receipt)
    return receipt


def _validate_dataset_manifest(manifest):
    """Allow documented fields/types recursively; never infer a safe payload by key blacklist."""
    number = (int, float)
    member = {"split": str, "group_id": str, "ids": [str], "label_map_sha256": str, "normalized_text_sha256": str}
    exclusion = {
        "split": str, "group_id": str, "ids": [str], "reason": str, "retained_id": str,
        "max_input_tokens": int, "action": str, "members": [member], "excluded_groups": int, "excluded_rows": int,
    }
    coverage = {key: int for key in ("selected_groups", "selected_rows", "deduplicated_groups", "token_excluded_groups", "retained_groups", "retained_rows")}
    coverage["group_retention_fraction"] = number
    quality = {"expected_hours": int, "observed_hours": int, "missing_hours": [str], "incomplete_hours": [str], "policy": str}
    source = {"url": str, "sha256": str, "bytes": int, "revision": str, "license": str,
              "attribution": str, "license_url": str, "month": str, "quality": quality}
    shape = {
        "format": str, "name": str, "benchmark": str, "sources": [source], "seed": int, "max_tokens": int,
        "tokenizer": {"source": str, "revision": (str, type(None)), "class": str, "semif_revision": str},
        "selection_policy": str, "deduplication_policy": str, "partition_rounding": str,
        "counts": ("map", int), "classes": [str], "class_counts": ("map", ("map", int)),
        "group_membership": ("map", [str]), "filter_coverage": ("map", coverage),
        "exclusions": [exclusion], "limitations": [str], "sha256": ("map", str),
        "candle_count": int, "lookback_hours": int, "horizon_hours": int,
        "decision_interval_hours": int, "label_threshold": number, "omitted": ("map", int),
        "boundary_purged": ("map", int), "split_upper_bounds_exclusive": ("map", str),
    }
    def validate(value, specification, location):
        if isinstance(specification, dict):
            if not isinstance(value, dict) or set(value) - set(specification):
                raise ValueError(f"Unsupported provenance fields at {location}")
            for key, item in value.items():
                validate(item, specification[key], f"{location}.{key}")
        elif isinstance(specification, list):
            if not isinstance(value, list):
                raise ValueError(f"Expected provenance list at {location}")
            for item in value:
                validate(item, specification[0], f"{location}[]")
        elif isinstance(specification, tuple) and specification[0] == "map":
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"Expected provenance mapping at {location}")
            for key, item in value.items():
                validate(item, specification[1], f"{location}.{key}")
        else:
            types = specification if isinstance(specification, tuple) else (specification,)
            if type(value) not in types or (type(value) is float and not np.isfinite(value)):
                raise ValueError(f"Invalid provenance value at {location}")
    validate(manifest, shape, "dataset")


def export_run(run_dir, data_dir, output_dir):
    root, data, destination = Path(run_dir), Path(data_dir), Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Evidence destination must be new: {destination}")
    bound_files = _bound_files(root)
    receipt = json.loads((root / "reload-verification.json").read_text())
    if receipt.get("format") != "ifllm-learn.reload.v2" or set(receipt.get("verified_sha256", {})) != set(bound_files):
        raise ValueError("Export requires a complete integrity-bound reload verification receipt")
    for name in bound_files:
        if file_sha256(root / name) != receipt["verified_sha256"][name]:
            raise ValueError(f"Artifact checksum differs from successful verification: {name}")
    report = json.loads((root / "report.json").read_text())
    training = json.loads((root / "adapter" / "training.json").read_text())
    if file_sha256(root / "adapter" / "adapters.safetensors") != training["adapter_sha256"]:
        raise ValueError("Run adapter checksum mismatch")
    if report["adapter_sha256"] != training["adapter_sha256"] or receipt["adapter_sha256"] != training["adapter_sha256"]:
        raise ValueError("Report or verification adapter identity mismatch")
    if receipt["run_sha256"] != file_sha256(root / "run.json") or receipt.get("answers_identical") is not True or receipt.get("calibrated_probabilities_match") is not True:
        raise ValueError("Verification receipt identity/status mismatch")
    for name, expected in report["run"]["data_sha256"].items():
        if name not in ("train", "valid", "calibration", "test") or file_sha256(data / f"{name}.jsonl") != expected:
            raise ValueError("Export dataset does not match the experiment")
    dataset_manifest = json.loads((data / "manifest.json").read_text())
    _validate_dataset_manifest(dataset_manifest)
    required = {
        "report.json": root / "report.json", "run.json": root / "run.json",
        "training.json": root / "adapter" / "training.json",
        "adapter_config.json": root / "adapter" / "adapter_config.json",
        "temperature.json": root / "temperature.json",
        "reload-verification.json": root / "reload-verification.json",
        "dataset-manifest.json": data / "manifest.json",
        "base-test.jsonl": root / "base-test.jsonl",
        "trained-test.jsonl": root / "trained-test.jsonl",
        "calibrated-test.jsonl": root / "calibrated-test.jsonl",
        "trained-calibration.jsonl": root / "trained-calibration.jsonl",
    }
    for name in V2_COMPARISONS:
        if name in bound_files:
            required[name] = root / name
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source in required.items():
        if source.suffix == ".jsonl":
            allowed = {"id", "answer", "label", "option_ids", "logits", "probabilities", "raw_probabilities",
                       "temperature", "input_tokens", "model_revision", "adapter_sha256", "precision",
                       "prompt_sha256", "contract_sha256", "total_seconds", "observed_at", "label_available_at"}
            clean = []
            for row in read_predictions(source):
                item = {key: value for key, value in row.items() if key in allowed}
                item["metadata"] = {key: row.get("metadata", {})[key] for key in ("group_id", "question_id") if key in row.get("metadata", {})}
                clean.append(item)
            write_jsonl(destination / name, clean)
        else:
            with (destination / name).open("xb") as stream:
                stream.write(source.read_bytes())
        files[name] = file_sha256(destination / name)
    manifest = {"format": "ifllm-learn.evidence.v1", "files": files,
                "excludes": ["source document text fields", "raw downloaded datasets", "model weights", "adapter weights"],
                "note": "Structured provenance filtering is not a general-purpose secret scanner. IDs and approved descriptions are retained. Hashes identify artifacts, not model quality."}
    write_json(destination / "SHA256SUMS.json", manifest)
    return {"output": str(destination), "files": len(files), "adapter_sha256": training["adapter_sha256"]}
