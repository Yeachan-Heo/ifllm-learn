import copy
import json

import numpy as np
import pytest

from ifllm_learn.calibration import apply_prediction_calibration, fit_prediction_calibration, fit_temperature
from ifllm_learn.metrics import metrics, negative_log_likelihood, prediction_arrays, softmax


def predictions(prefix="fit"):
    logits = [[8, 0], [8, 0], [0, 8], [0, 8]]
    labels = ["yes", "no", "no", "yes"]
    return [{"id": f"{prefix}-{i}", "answer": ["yes", "no"][int(np.argmax(z))],
             "label": label, "option_ids": ["yes", "no"], "logits": z,
             "probabilities": softmax(z).tolist(), "model_revision": "pinned",
             "adapter_sha256": "adapter", "contract_sha256": "task",
             "precision": {"dtype": "float32", "quantization": None}}
            for i, (z, label) in enumerate(zip(logits, labels))]


def test_temperature_improves_overconfident_fit_nll_without_changing_choices():
    logits, labels = prediction_arrays(predictions())
    fit = fit_temperature(logits, labels)
    assert fit["temperature"] > 1
    assert fit["nll_after"] < fit["nll_before"] - 1
    np.testing.assert_array_equal(softmax(logits).argmax(axis=1), softmax(logits, fit["temperature"]).argmax(axis=1))


def test_known_nontrivial_metrics():
    logits = np.log([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    result = metrics(logits, [0, 1, 1])
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["brier"] == pytest.approx((0.08 + 0.98 + 0.02) / 3)
    assert result["log_loss"] == pytest.approx(-np.log([0.8, 0.3, 0.9]).mean())
    assert sum(item["count"] for item in result["reliability_bins"]) == 3


def test_extreme_logits_stay_finite_and_shift_invariant():
    logits = np.array([[10000, -10000], [-10000, 10000]])
    assert negative_log_likelihood(logits, [1, 0]) == 20000
    np.testing.assert_allclose(softmax(logits), softmax(logits + 50000))
    assert np.isfinite(metrics(logits, [1, 0])["log_loss"])


@pytest.mark.parametrize("logits,labels", [([], []), ([[0, float("nan")]], [0]), ([[1]], [0]), ([[1, 2]], [2]), ([[1, 2]], [0.0]), ([[1, 2]], [0, 1])])
def test_reject_invalid_metric_inputs(logits, labels):
    with pytest.raises(ValueError):
        metrics(logits, labels)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        softmax([1, 2], temperature)


def test_artifact_roundtrip_identity_and_fit_id_guards(tmp_path):
    path = tmp_path / "temperature.json"
    artifact = fit_prediction_calibration(predictions(), path)
    assert json.loads(path.read_text()) == artifact
    test = predictions("test")
    calibrated = apply_prediction_calibration(test, artifact)
    assert calibrated[0]["raw_probabilities"] == test[0]["probabilities"]
    assert calibrated[0]["answer"] == test[0]["answer"]
    with pytest.raises(ValueError, match="fit IDs"):
        apply_prediction_calibration(predictions(), artifact)
    for key, value in (("model_revision", "different"), ("adapter_sha256", "different"),
                       ("contract_sha256", "different"), ("option_ids", ["no", "yes"]),
                       ("precision", {"quantization": 4})):
        changed = copy.deepcopy(test)
        for row in changed:
            row[key] = value
        with pytest.raises(ValueError, match="incompatible"):
            apply_prediction_calibration(changed, artifact)
    with pytest.raises(FileExistsError):
        fit_prediction_calibration(predictions(), path)
