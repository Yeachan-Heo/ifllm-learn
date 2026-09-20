"""Classification metrics computed from native option logits, not generated scores."""

import math
import numpy as np
from scipy.special import logsumexp


def validate_logits(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] < 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite logits with at least two classes")
    return values


def validate_labels(logits, labels):
    values = validate_logits(logits)
    targets = np.asarray(labels)
    if values.ndim != 2 or targets.shape != (len(values),) or targets.dtype.kind not in "iu":
        raise ValueError("Expected 2D logits and matching integer label indices")
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("Label index is outside option range")
    return values, targets.astype(np.int64)


def log_probabilities(logits, temperature=1.0):
    values = validate_logits(logits)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite scalar")
    # Center before scaling to avoid unnecessarily large values.
    with np.errstate(over="raise", invalid="raise"):
        try:
            scaled = (values - values.max(axis=-1, keepdims=True)) / temperature
        except FloatingPointError as error:
            raise ValueError("Logit range exceeds numeric capacity") from error
    return scaled - logsumexp(scaled, axis=-1, keepdims=True)


def softmax(logits, temperature=1.0):
    return np.exp(log_probabilities(logits, temperature))


def negative_log_likelihood(logits, labels, temperature=1.0):
    values, targets = validate_labels(logits, labels)
    logp = log_probabilities(values, temperature)
    return float(-logp[np.arange(len(values)), targets].mean())


def metrics(logits, labels, *, temperature=1.0):
    values, targets = validate_labels(logits, labels)
    probabilities = softmax(values, temperature)
    predicted = probabilities.argmax(axis=1)
    n, classes = values.shape
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    f1 = []
    for index in range(classes):
        denominator = int(confusion[index].sum() + confusion[:, index].sum())
        f1.append(2 * int(confusion[index, index]) / denominator if denominator else 0.0)
    correct = predicted == targets
    confidence = probabilities.max(axis=1)
    bins, ece = [], 0.0
    membership = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        mask = membership == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(accuracy - mean_confidence)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": count,
                     "accuracy": accuracy, "mean_confidence": mean_confidence})
    one_hot = np.eye(classes)[targets]
    return {
        "count": n, "accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1)),
        "log_loss": negative_log_likelihood(values, targets, temperature),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "brier_definition": "mean sum of squared class probability errors",
        "confusion_matrix": confusion.tolist(), "ece_10_bins": ece, "reliability_bins": bins,
    }


def prediction_arrays(predictions):
    if not predictions:
        raise ValueError("Predictions must not be empty")
    ids, choices = set(), predictions[0]["option_ids"]
    if not isinstance(choices, list) or len(choices) < 2 or len(set(choices)) != len(choices):
        raise ValueError("Invalid option IDs")
    logits, labels = [], []
    for row in predictions:
        if row["id"] in ids or row["option_ids"] != choices:
            raise ValueError("Prediction IDs must be unique and ordered option IDs must match")
        ids.add(row["id"])
        if row.get("label") not in choices or len(row["logits"]) != len(choices):
            raise ValueError("Prediction label or logit shape is invalid")
        logits.append(row["logits"])
        labels.append(choices.index(row["label"]))
    values, targets = validate_labels(logits, labels)
    return values, targets
