"""Reproducible experiments: train/validation/calibration/test stay separate."""

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .calibration import apply_prediction_calibration, fit_prediction_calibration
from .metrics import metrics, prediction_arrays
from .schema import assert_disjoint, assert_temporal_separation, contract_hash, file_sha256, load_jsonl, write_json, write_jsonl

SPLITS = ("train", "valid", "calibration", "test")


def choose_subset(rows, limit):
    if limit is None:
        return rows
    if type(limit) is not int or limit < 1:
        raise ValueError("Subset limits must be positive integers")
    if limit >= len(rows):
        return rows
    grouped = {}
    for row in rows:
        key = row.get("metadata", {}).get("group_id", row["id"])
        grouped.setdefault(key, []).append(row)
    if any(len(group) > 1 for group in grouped.values()):
        sizes = {len(group) for group in grouped.values()}
        if len(sizes) != 1:
            raise ValueError("Limited multi-question datasets require equal-sized complete groups")
        size = sizes.pop()
        count = limit // size
        if count < 1:
            raise ValueError("Subset limit is smaller than a complete document group")
        keys = list(grouped)
        chosen = {keys[i] for i in np.linspace(0, len(keys) - 1, count, dtype=int)}
        return [row for row in rows if row.get("metadata", {}).get("group_id", row["id"]) in chosen]
    # Label-blind, deterministic sampling. No search for favorable test examples.
    return [rows[index] for index in np.linspace(0, len(rows) - 1, limit, dtype=int)]


def validate_splits(groups):
    assert_disjoint(*(groups[name] for name in SPLITS))
    if len({contract_hash(groups[name]) for name in SPLITS}) != 1:
        raise ValueError("All four splits must share the same task contract")
    assert_temporal_separation(*(groups[name] for name in SPLITS))


def load_splits(data_dir):
    data_dir = Path(data_dir)
    groups = {name: load_jsonl(data_dir / f"{name}.jsonl") for name in SPLITS}
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, UnicodeError) as error:
            raise ValueError("Dataset manifest checksum metadata is malformed") from error
        checksums = manifest.get("sha256") if isinstance(manifest, dict) else None
        expected_names = {f"{name}.jsonl" for name in SPLITS}
        if not isinstance(checksums, dict) or set(checksums) != expected_names:
            raise ValueError("Dataset manifest checksum mapping must cover exactly all four splits")
        for name, expected in checksums.items():
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(character not in "0123456789abcdefABCDEF" for character in expected)):
                raise ValueError("Dataset manifest checksum must be a 64-character hexadecimal string")
            if file_sha256(data_dir / name) != expected.lower():
                raise ValueError("Dataset manifest checksum mismatch")
    validate_splits(groups)
    return groups


def source_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_workflow(data_dir, output_dir, *, steps=100, learning_rate=1e-5, num_layers=2,
                 rank=8, max_tokens=2048, seed=0, bits=None, train_limit=None,
                 valid_limit=None, calibration_limit=None, test_limit=None, batch_size=1):
    from .baselines import task_metrics, text_baseline
    from .inference import load_backend, predict_rows
    from .training import train_adapter, validate_training

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Experiment output must be new: {output_dir}")
    all_groups = load_splits(data_dir)
    limits = dict(zip(SPLITS, (train_limit, valid_limit, calibration_limit, test_limit)))
    groups = {name: choose_subset(all_groups[name], limits[name]) for name in SPLITS}
    validate_splits(groups)
    validate_training(groups["train"], groups["valid"], steps, learning_rate, num_layers, rank, max_tokens, bits)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    config = {
        "steps": steps, "learning_rate": learning_rate, "num_layers": num_layers,
        "rank": rank, "max_tokens": max_tokens, "seed": seed, "bits": bits, "batch_size": batch_size,
        "full_counts": {name: len(all_groups[name]) for name in SPLITS},
        "selected_counts": {name: len(groups[name]) for name in SPLITS},
        "selected_ids": {name: [row["id"] for row in groups[name]] for name in SPLITS},
        "subset_policy": "equally spaced input indices, whole document groups retained, no test-label selection",
        "data_sha256": {name: file_sha256(Path(data_dir) / f"{name}.jsonl") for name in SPLITS},
        "source_sha256": source_digest(),
        "hardware": {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version()},
        "test_policy": "fixed budget, final optimizer state, no test-driven checkpoint selection",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run.json", config)
    # Kept locally for a runnable reload check; publishing exports no document text.
    write_jsonl(output_dir / "test-input.jsonl", groups["test"])
    started = time.perf_counter()
    classical = text_baseline(groups["train"], groups["test"], seed=seed)
    write_json(output_dir / "tfidf-baseline.json", classical)
    bundle = load_backend(bits=bits)
    print("Scoring the frozen base model on fixed calibration and test partitions", flush=True)
    base_cal = predict_rows(bundle, groups["calibration"], max_tokens)
    write_jsonl(output_dir / "base-calibration.jsonl", base_cal)
    base_artifact = fit_prediction_calibration(base_cal, output_dir / "base-temperature.json")
    base = predict_rows(bundle, groups["test"], max_tokens)
    write_jsonl(output_dir / "base-test.jsonl", base)
    base_calibrated = apply_prediction_calibration(base, base_artifact)
    write_jsonl(output_dir / "base-calibrated-test.jsonl", base_calibrated)
    training = train_adapter(groups["train"], groups["valid"], output_dir / "adapter",
                             steps=steps, learning_rate=learning_rate, num_layers=num_layers,
                             rank=rank, max_tokens=max_tokens, seed=seed, bits=bits, bundle=bundle, batch_size=batch_size)
    print("Scoring the trained adapter without selecting on test outcomes", flush=True)
    calibration = predict_rows(bundle, groups["calibration"], max_tokens)
    write_jsonl(output_dir / "trained-calibration.jsonl", calibration)
    artifact = fit_prediction_calibration(calibration, output_dir / "temperature.json")
    trained = predict_rows(bundle, groups["test"], max_tokens)
    write_jsonl(output_dir / "trained-test.jsonl", trained)
    calibrated = apply_prediction_calibration(trained, artifact)
    write_jsonl(output_dir / "calibrated-test.jsonl", calibrated)
    base_logits, labels = prediction_arrays(base)
    trained_logits, trained_labels = prediction_arrays(trained)
    if not np.array_equal(labels, trained_labels):
        raise RuntimeError("Evaluation labels changed between readouts")
    choices = base[0]["option_ids"]
    counts = np.asarray([sum(row["label"] == choice for row in groups["train"]) for choice in choices])
    prior = (counts + 1) / (counts.sum() + len(choices))
    prior_logits = np.tile(np.log(prior), (len(labels), 1))
    report = {
        "format": "ifllm-learn.experiment.v2", "run": config, "option_ids": choices,
        "train_frequency_baseline": {"probabilities": prior.tolist(), "smoothing": "add-one", **metrics(prior_logits, labels)},
        "tfidf_baseline": classical,
        "base": metrics(base_logits, labels),
        "base_calibrated": metrics(base_logits, labels, temperature=base_artifact["temperature"]),
        "trained": metrics(trained_logits, labels),
        "trained_calibrated": metrics(trained_logits, labels, temperature=artifact["temperature"]),
        "task_metrics": {"base": task_metrics(base), "base_calibrated": task_metrics(base_calibrated),
                         "trained": task_metrics(trained), "trained_calibrated": task_metrics(calibrated)},
        "temperature": artifact["temperature"], "base_temperature": base_artifact["temperature"],
        "adapter_sha256": training["adapter_sha256"],
        "changed_adapter_tensor_count": len(training["changed_adapter_tensors"]),
        "validation_loss_before": training["validation_loss_before"],
        "validation_loss_after": training["validation_loss_after"],
        "validation_improved": training["validation_improved"],
        "elapsed_seconds_including_model_load": time.perf_counter() - started,
        "limitations": [
            "Fixed-budget single-seed demos, not full benchmark or state-of-the-art claims.",
            "Public data may have occurred in model pretraining; within-experiment split hygiene cannot eliminate pretraining contamination.",
            "Training uses train/valid only; temperatures use calibration only; test scores must not guide tuning.",
            "Scalar temperature preserves argmax and cannot repair an incorrect ranking or guarantee future correctness.",
        ],
    }
    write_json(output_dir / "report.json", report)
    return report
