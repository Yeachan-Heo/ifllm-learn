"""Held-out temperature scaling; this does not train the prediction model."""

import math
import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import negative_log_likelihood, prediction_arrays, softmax, validate_labels, validate_logits
from .schema import write_json

IDENTITY = ("model_revision", "adapter_sha256", "contract_sha256", "option_ids", "precision")


def fit_temperature(logits, labels):
    values, targets = validate_labels(logits, labels)
    baseline = negative_log_likelihood(values, targets)
    result = minimize_scalar(
        lambda log_t: negative_log_likelihood(values, targets, math.exp(log_t)),
        bounds=(-5.0, 5.0), method="bounded", options={"xatol": 1e-7},
    )
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = math.exp(float(result.x)) if result.fun < baseline else 1.0
    return {
        "temperature": temperature, "n": len(values), "nll_before": baseline,
        "nll_after": negative_log_likelihood(values, targets, temperature),
        "optimizer": "bounded scipy minimize_scalar on log(T)", "log_temperature_bounds": [-5, 5],
        "at_bound": abs(float(result.x)) > 4.99,
    }


def prediction_identity(predictions):
    if not predictions:
        raise ValueError("No predictions")
    identity = {key: predictions[0][key] for key in IDENTITY}
    for row in predictions:
        if any(row[key] != value for key, value in identity.items()):
            raise ValueError("Prediction model/adapter/precision/task/option identities do not match")
    return identity


def fit_prediction_calibration(predictions, output_path):
    identity = prediction_identity(predictions)
    values, labels = prediction_arrays(predictions)
    artifact = {
        "format": "ifllm-learn.temperature.v1", **identity,
        **fit_temperature(values, labels),
        "fit_ids": [row["id"] for row in predictions],
        "fit_group_ids": sorted({row.get("metadata", {}).get("group_id") for row in predictions} - {None}),
        "limitation": "A common positive temperature preserves argmax; calibration may not generalize under distribution shift.",
    }
    write_json(output_path, artifact)
    return artifact


def apply_prediction_calibration(predictions, artifact):
    if artifact.get("format") != "ifllm-learn.temperature.v1":
        raise ValueError("Unsupported calibration artifact")
    identity = prediction_identity(predictions)
    if any(identity[key] != artifact.get(key) for key in IDENTITY):
        raise ValueError("Calibration artifact is incompatible with prediction identity")
    fit_ids = set(artifact["fit_ids"])
    if any(row["id"] in fit_ids for row in predictions):
        raise ValueError("Calibration fit IDs cannot be reused for held-out application/evaluation")
    fit_groups = set(artifact.get("fit_group_ids", []))
    if any(row.get("metadata", {}).get("group_id") in fit_groups for row in predictions):
        raise ValueError("Calibration fit groups cannot be reused for held-out evaluation")
    choices = identity["option_ids"]
    if len(choices) < 2 or len(set(choices)) != len(choices):
        raise ValueError("Invalid option IDs")
    ids, result = set(), []
    for row in predictions:
        if row["id"] in ids:
            raise ValueError("Duplicate prediction IDs")
        ids.add(row["id"])
        values = validate_logits(row["logits"])
        if values.ndim != 1 or len(values) != len(choices):
            raise ValueError("Invalid logit vector")
        probabilities = softmax(values, artifact["temperature"])
        result.append({**row, "raw_probabilities": row["probabilities"],
                       "probabilities": probabilities.tolist(),
                       "answer": choices[int(np.argmax(probabilities))],
                       "temperature": artifact["temperature"]})
    return result
