import copy

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ifllm_learn.baselines import task_metrics, text_baseline
from ifllm_learn.calibration import apply_prediction_calibration
from ifllm_learn.metrics import softmax
from ifllm_learn.schema import canonical_json


def example(identifier, prompt, label):
    return {
        "id": identifier, "prompt": prompt, "question": "Classify the message.",
        "options": [{"id": "zfruit", "description": "Fruit descriptionsecret"},
                    {"id": "acar", "description": "Vehicle descriptionsecret"}],
        "label": label, "metadata": {"gold_answer": "metadatasecret"},
    }


def text_rows():
    train = [example("train-1", "apple banana fruit", "zfruit"),
             example("train-2", {"z": "banana", "a": "apple fruit"}, "zfruit"),
             example("train-3", "car truck vehicle", "acar"),
             example("train-4", "truck car vehicle", "acar")]
    test = [example("test-1", "apple banana fruit testonlysecret", "zfruit"),
            example("test-2", "car truck vehicle", "acar")]
    return train, test


def test_real_tfidf_fit_uses_only_training_question_and_prompt(monkeypatch):
    observed = {}
    original_fit_transform = TfidfVectorizer.fit_transform
    original_transform = TfidfVectorizer.transform

    def fit_transform(self, raw_documents, y=None):
        documents = list(raw_documents)
        observed["training_text"] = documents
        result = original_fit_transform(self, documents, y)
        observed["vocabulary"] = dict(self.vocabulary_)
        observed["parameters"] = self.get_params()
        return result

    def transform(self, raw_documents):
        documents = list(raw_documents)
        observed["test_text"] = documents
        return original_transform(self, documents)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", fit_transform)
    monkeypatch.setattr(TfidfVectorizer, "transform", transform)
    train, test = text_rows()
    result = text_baseline(train, test)
    assert observed["training_text"] == [row["question"] + "\n" + (
        row["prompt"] if isinstance(row["prompt"], str) else canonical_json(row["prompt"])) for row in train]
    assert observed["test_text"] == [row["question"] + "\n" + row["prompt"] for row in test]
    for forbidden in ("testonlysecret", "zfruit", "acar", "descriptionsecret", "metadatasecret", "train-1"):
        assert forbidden not in observed["vocabulary"]
    assert "apple banana" in observed["vocabulary"]
    assert observed["parameters"]["sublinear_tf"] is True
    assert result["option_ids"] == ["zfruit", "acar"]
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["vocabulary_size"] == len(observed["vocabulary"])
    assert result["train_count"] == 4
    assert result["test_count"] == 2
    assert 0 < result["log_loss"] < np.log(2)
    assert result["fit_seconds"] >= 0
    assert result["predict_seconds"] >= 0


def test_text_baseline_is_deterministic_and_test_labels_are_not_features():
    train, test = text_rows()
    first = text_baseline(train, test, seed=29)
    second = text_baseline(train, test, seed=29)
    assert {key: value for key, value in first.items() if not key.endswith("_seconds")} == {
        key: value for key, value in second.items() if not key.endswith("_seconds")}
    changed = copy.deepcopy(test)
    changed[0]["label"], changed[1]["label"] = changed[1]["label"], changed[0]["label"]
    swapped = text_baseline(train, changed, seed=29)
    assert swapped["accuracy"] == 0
    assert swapped["confusion_matrix"] == [[0, 1], [1, 0]]
    assert swapped["vocabulary_size"] == first["vocabulary_size"]
    assert [row["mean_confidence"] for row in swapped["reliability_bins"]] == [
        row["mean_confidence"] for row in first["reliability_bins"]]


@pytest.mark.parametrize("kind", ["empty_train", "empty_test", "missing_class", "invalid_label", "reordered", "duplicate"])
def test_text_baseline_rejects_invalid_data(kind):
    train, test = text_rows()
    if kind == "empty_train":
        train = []
    elif kind == "empty_test":
        test = []
    elif kind == "missing_class":
        train = train[:2]
    elif kind == "invalid_label":
        test[0]["label"] = "unknown"
    elif kind == "reordered":
        test[0]["options"].reverse()
    elif kind == "duplicate":
        test[0]["id"] = train[0]["id"]
    with pytest.raises(ValueError):
        text_baseline(train, test)


def predictions():
    choices = ["billing", "transfer", "out_of_scope"]
    probabilities = [[0.8, 0.1, 0.1], [0.15, 0.75, 0.1], [0.1, 0.1, 0.8],
                     [0.15, 0.8, 0.05], [0.7, 0.2, 0.1], [0.025, 0.025, 0.95]]
    labels = ["billing", "billing", "billing", "transfer", "out_of_scope", "out_of_scope"]
    return [{"id": f"row-{index}", "option_ids": choices.copy(), "label": label,
             "answer": choices[int(np.argmax(probability))],
             "logits": np.log(probability).tolist(), "probabilities": probability,
             "metadata": {"question_id": "question-a" if index < 5 else "question-b",
                          "group_id": f"contract-{index % 3}"}}
            for index, (label, probability) in enumerate(zip(labels, probabilities))]


def test_nontrivial_task_confusion_oos_and_coverage():
    result = task_metrics(predictions())
    assert result["confusion_matrix"] == [[1, 1, 1], [0, 1, 0], [1, 0, 1]]
    assert result["accuracy"] == 0.5
    assert result["per_class"]["billing"] == pytest.approx({
        "precision": 0.5, "recall": 1 / 3, "f1": 0.4, "support": 3})
    assert result["per_class"]["transfer"] == pytest.approx({
        "precision": 0.5, "recall": 1, "f1": 2 / 3, "support": 1})
    assert result["out_of_scope_recall"] == 0.5
    assert result["in_scope_false_rejection_rate"] == 0.25
    assert result["supported_intent_accuracy"] == 0.5
    curve = result["confidence_coverage_curve"]
    assert [point["threshold"] for point in curve] == [0, 0.5, 0.7, 0.8, 0.9, 0.95]
    assert curve[0] == {"threshold": 0, "coverage": 1, "error": 0.5, "accepted_count": 6}
    assert curve[3] == pytest.approx({"threshold": 0.8, "coverage": 4 / 6, "error": 0.25, "accepted_count": 4})
    assert curve[-1] == pytest.approx({"threshold": 0.95, "coverage": 1 / 6, "error": 0, "accepted_count": 1})


def test_question_metrics_pool_groups_and_weight_questions_equally():
    result = task_metrics(predictions())
    assert result["per_question"]["question-a"]["count"] == 5
    assert result["per_question"]["question-b"]["count"] == 1
    assert result["question_count"] == 2
    assert result["question_row_count"] == 6
    assert result["equal_question_macro_accuracy"] == pytest.approx(0.7)
    assert result["equal_question_macro_f1"] == pytest.approx(((0.4 + 2 / 3) / 3 + 1 / 3) / 2)
    assert result["equal_question_macro_accuracy"] != result["accuracy"]


def test_calibrated_probabilities_drive_confidence_not_unscaled_logits():
    rows = predictions()
    identity = {"model_revision": "pinned", "adapter_sha256": None,
                "contract_sha256": "task", "option_ids": rows[0]["option_ids"],
                "precision": {"dtype": "float32", "quantization": None}}
    for row in rows:
        row.update(identity)
    calibrated = apply_prediction_calibration(rows, {
        **identity, "format": "ifllm-learn.temperature.v1", "temperature": 10, "fit_ids": ["held-out-fit"]})
    result = task_metrics(calibrated)
    assert result["confusion_matrix"] == task_metrics(rows)["confusion_matrix"]
    assert result["confidence_coverage_curve"][1] == {
        "threshold": 0.5, "coverage": 0, "error": None, "accepted_count": 0}
    assert task_metrics(rows)["confidence_coverage_curve"][1]["coverage"] == 1
    for row in calibrated:
        del row["probabilities"]
    assert task_metrics(calibrated)["confidence_coverage_curve"] == result["confidence_coverage_curve"]


def test_probabilities_drive_decisions_and_answers_must_match():
    rows = predictions()[:1]
    rows[0]["probabilities"] = [0.1, 0.8, 0.1]
    rows[0]["answer"] = "transfer"
    result = task_metrics(rows)
    assert result["accuracy"] == 0
    assert result["confusion_matrix"] == [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
    rows[0]["answer"] = "billing"
    with pytest.raises(ValueError, match="answer"):
        task_metrics(rows)


def test_absent_conditional_populations_and_zero_support_classes():
    inside = task_metrics(predictions()[:1])
    assert inside["out_of_scope_recall"] is None
    assert inside["per_class"]["transfer"] == {"precision": 0, "recall": 0, "f1": 0, "support": 0}
    outside = task_metrics(predictions()[-1:])
    assert outside["in_scope_false_rejection_rate"] is None
    assert outside["supported_intent_accuracy"] is None
    rows = predictions()[:1]
    for row in rows:
        row.pop("metadata")
        row["option_ids"][2] = "other"
    result = task_metrics(rows)
    assert "out_of_scope_recall" not in result
    assert "per_question" not in result


def test_missing_probabilities_use_native_logits():
    rows = predictions()
    expected = task_metrics(rows)
    for row in rows:
        del row["probabilities"]
    result = task_metrics(rows)
    assert result["confusion_matrix"] == expected["confusion_matrix"]
    assert result["per_class"] == expected["per_class"]
    np.testing.assert_allclose(softmax(rows[0]["logits"]), [0.8, 0.1, 0.1])


@pytest.mark.parametrize("probabilities", [[0.4, 0.4], [-0.1, 0.5, 0.6], [0.5, 0.5, 0.5],
                                           [float("nan"), 0.5, 0.5], [float("inf"), 0, 0], [1.1, 0, -0.1]])
def test_invalid_probability_vectors_rejected(probabilities):
    rows = predictions()
    rows[0]["probabilities"] = probabilities
    with pytest.raises(ValueError, match="probabilities"):
        task_metrics(rows)


@pytest.mark.parametrize("kind", ["empty", "label", "options", "duplicate", "answer", "metadata", "question_id"])
def test_invalid_task_predictions_rejected(kind):
    rows = predictions()
    if kind == "empty":
        rows = []
    elif kind == "label":
        rows[0]["label"] = "unknown"
    elif kind == "options":
        rows[1]["option_ids"].reverse()
    elif kind == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif kind == "answer":
        rows[0]["answer"] = "unknown"
    elif kind == "metadata":
        rows[0]["metadata"] = []
    elif kind == "question_id":
        rows[0]["metadata"]["question_id"] = ""
    with pytest.raises(ValueError):
        task_metrics(rows)
