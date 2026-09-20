"""Measured text baselines and task-specific descriptive classification metrics."""

from time import perf_counter

import numpy as np

from .metrics import metrics, prediction_arrays, softmax
from .schema import assert_disjoint, canonical_json, validate_example


def text_baseline(train_rows, test_rows, *, seed=17):
    """Fit TF-IDF and logistic regression on training text only; retain no model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if not train_rows or not test_rows:
        raise ValueError("Training and test rows must not be empty")
    for rows in (train_rows, test_rows):
        for row in rows:
            validate_example(row)
    assert_disjoint(train_rows, test_rows)
    options = train_rows[0]["options"]
    option_ids = [option["id"] for option in options]
    for rows in (train_rows, test_rows):
        if any(row["options"] != options for row in rows):
            raise ValueError("Ordered options must match across training and test rows")
    if {row["label"] for row in train_rows} != set(option_ids):
        raise ValueError("Training labels must include every option class")

    def text(row):
        prompt = row["prompt"]
        return row["question"] + "\n" + (prompt if isinstance(prompt, str) else canonical_json(prompt))

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
    classifier = LogisticRegression(C=1, max_iter=1000, random_state=seed)
    train_text = [text(row) for row in train_rows]
    test_text = [text(row) for row in test_rows]
    started = perf_counter()
    features = vectorizer.fit_transform(train_text)
    classifier.fit(features, [row["label"] for row in train_rows])
    fit_seconds = perf_counter() - started
    classes = classifier.classes_.tolist()
    if set(classes) != set(option_ids):
        raise ValueError("Fitted classifier is missing an option class")
    order = [classes.index(option_id) for option_id in option_ids]
    started = perf_counter()
    log_probabilities = classifier.predict_log_proba(vectorizer.transform(test_text))[:, order]
    predict_seconds = perf_counter() - started
    targets = [option_ids.index(row["label"]) for row in test_rows]
    return {
        **metrics(log_probabilities, targets),
        "model_description": "scikit-learn word TF-IDF (1,2) with logistic regression; trained on question and prompt only",
        "hyperparameters": {
            "tfidf": {"analyzer": "word", "ngram_range": [1, 2], "sublinear_tf": True},
            "logistic_regression": {"C": 1, "max_iter": 1000, "random_state": seed},
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "train_count": len(train_rows), "test_count": len(test_rows), "option_ids": option_ids,
    }


def _classification_metrics(targets, predicted, option_ids):
    confusion = np.zeros((len(option_ids), len(option_ids)), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    per_class = {}
    for index, option_id in enumerate(option_ids):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        chosen = int(confusion[:, index].sum())
        per_class[option_id] = {
            "precision": true_positive / chosen if chosen else 0.0,
            "recall": true_positive / support if support else 0.0,
            "f1": 2 * true_positive / (support + chosen) if support + chosen else 0.0,
            "support": support,
        }
    return {
        "count": len(targets), "accuracy": float(np.mean(targets == predicted)),
        "macro_f1": float(np.mean([value["f1"] for value in per_class.values()])),
        "confusion_matrix": confusion.tolist(), "per_class": per_class,
    }


def task_metrics(predictions):
    """Describe decisions using stored probabilities, including calibrated ones.

    Zero-support class scores are zero. Undefined conditional rates are None.
    Coverage/error pairs are observed summaries, not statistical risk guarantees.
    Contract questions pool all groups with the same metadata.question_id.
    """
    values, targets = prediction_arrays(predictions)
    option_ids = predictions[0]["option_ids"]
    if any(not isinstance(option_id, str) or not option_id.strip() for option_id in option_ids):
        raise ValueError("Option IDs must be nonempty strings")
    probabilities = []
    for row, logits in zip(predictions, values):
        if "probabilities" in row:
            probability = np.asarray(row["probabilities"], dtype=np.float64)
        else:
            probability = softmax(logits, row.get("temperature", 1.0))
        if (probability.shape != (len(option_ids),) or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.isclose(probability.sum(), 1.0, rtol=1e-6, atol=1e-8)):
            raise ValueError("Invalid prediction probabilities")
        answer = option_ids[int(probability.argmax())]
        if "answer" in row and row["answer"] != answer:
            raise ValueError("Prediction answer must match probability argmax")
        probabilities.append(probability)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    result = {**_classification_metrics(targets, predicted, option_ids), "option_ids": option_ids}
    if "out_of_scope" in option_ids:
        oos = option_ids.index("out_of_scope")
        outside = targets == oos
        inside = ~outside
        result.update({
            "out_of_scope_recall": float(np.mean(predicted[outside] == oos)) if outside.any() else None,
            "in_scope_false_rejection_rate": float(np.mean(predicted[inside] == oos)) if inside.any() else None,
            "supported_intent_accuracy": float(np.mean(predicted[inside] == targets[inside])) if inside.any() else None,
        })
    confidence = probabilities.max(axis=1)
    curve = []
    for threshold in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        accepted = confidence >= threshold
        curve.append({
            "threshold": threshold, "coverage": float(accepted.mean()),
            "error": float(np.mean(predicted[accepted] != targets[accepted])) if accepted.any() else None,
            "accepted_count": int(accepted.sum()),
        })
    result["confidence_coverage_curve"] = curve
    result["coverage_limitation"] = "Observed confidence coverage and error only; not a risk guarantee."

    questions = {}
    for index, row in enumerate(predictions):
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Prediction metadata must be an object")
        if "question_id" in metadata:
            question_id = metadata["question_id"]
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError("metadata.question_id must be a nonempty string")
            questions.setdefault(question_id, []).append(index)
    if questions:
        per_question = {
            question_id: _classification_metrics(targets[indices], predicted[indices], option_ids)
            for question_id, indices in sorted(questions.items())
        }
        result.update({
            "per_question": per_question,
            "question_count": len(per_question),
            "question_row_count": sum(len(indices) for indices in questions.values()),
            "equal_question_macro_accuracy": float(np.mean([value["accuracy"] for value in per_question.values()])),
            "equal_question_macro_f1": float(np.mean([value["macro_f1"] for value in per_question.values()])),
        })
    return result
