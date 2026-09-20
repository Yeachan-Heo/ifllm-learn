"""Adversarial artifact contracts; synthetic model, real temporary run files.

No model download or MLX invocation is permitted in this suite. Strict failures
are intentional bug reports, not xfails. Execution belongs to the parent gate.
"""

import copy
import hashlib
import json
from pathlib import Path
import runpy
from unittest.mock import Mock

import pytest

from ifllm_learn.calibration import apply_prediction_calibration, fit_prediction_calibration
from ifllm_learn.evidence import export_run, read_predictions, verify_run
from ifllm_learn.metrics import metrics, prediction_arrays, softmax
from ifllm_learn.schema import canonical_json, contract_hash, file_sha256, to_semif, write_json, write_jsonl
from ifllm_learn.workflow import SPLITS, choose_subset, validate_splits

SECRET_PROMPT = "PRIVATE_CONTRACT_TEXT_76b331"
SECRET_SPAN = "PRIVATE_GOLD_EVIDENCE_391fed"


def replace_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def replace_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def stored_run(tmp_path, monkeypatch):
    from ifllm_learn import inference

    root, data = tmp_path / "run", tmp_path / "data"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_bytes(b"synthetic adapter, never loaded")
    adapter_hash = file_sha256(adapter / "adapters.safetensors")
    options = [{"id": "yes", "description": "Entailed"},
               {"id": "no", "description": "Not entailed"}]
    groups = {
        split: [{"id": f"{split}:{index}", "prompt": SECRET_PROMPT,
                 "question": "Does the obligation apply?", "options": options,
                 "label": label, "metadata": {"group_id": f"{split}-document",
                                                "question_id": str(index),
                                                "gold_spans": [SECRET_SPAN]}}
                for index, label in enumerate(("yes", "no"))]
        for split in SPLITS
    }
    for split, rows in groups.items():
        write_jsonl(data / f"{split}.jsonl", rows)
    write_json(data / "manifest.json", {
        "format": "ifllm-learn.textdata.v1", "name": "synthetic-contracts", "sha256": {
            f"{split}.jsonl": file_sha256(data / f"{split}.jsonl") for split in SPLITS}})
    config = {"bits": None, "max_tokens": 256,
              "selected_ids": {split: [row["id"] for row in rows]
                               for split, rows in groups.items()},
              "data_sha256": {split: file_sha256(data / f"{split}.jsonl") for split in SPLITS}}
    write_json(root / "run.json", config)
    write_jsonl(root / "test-input.jsonl", groups["test"])
    write_json(adapter / "adapter_config.json", {"fine_tune_type": "lora"})
    write_json(adapter / "training.json", {"adapter_sha256": adapter_hash})

    def predictions(rows):
        return [{"id": row["id"], "answer": "yes", "label": row["label"],
                 "option_ids": ["yes", "no"], "logits": [2.0, -1.0],
                 "probabilities": softmax([2.0, -1.0]).tolist(),
                 "model_revision": "synthetic-pinned-revision", "adapter_sha256": adapter_hash,
                 "contract_sha256": contract_hash(rows), "precision": "float32",
                 "prompt_sha256": hashlib.sha256(canonical_json(to_semif(row)).encode("utf-8")).hexdigest(),
                 "metadata": copy.deepcopy(row["metadata"])} for row in rows]

    trained, calibration = predictions(groups["test"]), predictions(groups["calibration"])
    base = [{**row, "adapter_sha256": None} for row in trained]
    artifact = fit_prediction_calibration(calibration, root / "temperature.json")
    calibrated = apply_prediction_calibration(trained, artifact)
    for name, rows in (("trained-test", trained), ("base-test", base),
                       ("trained-calibration", calibration), ("calibrated-test", calibrated)):
        write_jsonl(root / f"{name}.jsonl", rows)
    logits, labels = prediction_arrays(trained)
    report = {"format": "ifllm-learn.experiment.v1", "run": config, "adapter_sha256": adapter_hash,
              "base": metrics(logits, labels), "trained": metrics(logits, labels),
              "trained_calibrated": metrics(logits, labels, temperature=artifact["temperature"])}
    write_json(root / "report.json", report)
    backend = object()
    load = Mock(return_value=backend)
    predict = Mock(side_effect=lambda bundle, rows, max_tokens: predictions(rows))
    monkeypatch.setattr(inference, "load_backend", load)
    monkeypatch.setattr(inference, "predict_rows", predict)
    return {"root": root, "data": data, "output": tmp_path / "export",
            "groups": groups, "trained": trained, "load": load, "predict": predict,
            "backend": backend}


def export_verified(fixture):
    verify_run(fixture["root"])
    export_run(fixture["root"], fixture["data"], fixture["output"])
    return fixture["output"]


def retained_verifier():
    path = Path(__file__).resolve().parents[1] / "examples" / "verify_results.py"
    return runpy.run_path(str(path))["verify"]


def test_reload_and_export_fixture_is_verifiable(stored_run):
    output = export_verified(stored_run)
    result = retained_verifier()(output)
    assert result["test_rows"] == 2
    assert result["trained_accuracy"] == 0.5
    stored_run["load"].assert_called_once_with(stored_run["root"] / "adapter", None)
    stored_run["predict"].assert_called_once_with(
        stored_run["backend"], stored_run["groups"]["test"], 256)


def test_export_strips_raw_prompts_and_gold_spans_from_predictions(stored_run):
    root = stored_run["root"]
    for name in ("base-test", "trained-test", "calibrated-test", "trained-calibration"):
        path = root / f"{name}.jsonl"
        rows = read_predictions(path)
        for row in rows:
            row.update(prompt=SECRET_PROMPT, gold_spans=[SECRET_SPAN])
            row["metadata"]["nested"] = {"raw_prompt": SECRET_PROMPT}
        replace_jsonl(path, rows)
    output = export_verified(stored_run)
    for content in snapshot(output).values():
        assert SECRET_PROMPT.encode() not in content
        assert SECRET_SPAN.encode() not in content
    retained = read_predictions(output / "trained-test.jsonl")
    assert retained[0]["metadata"] == {"group_id": "test-document", "question_id": "0"}


def test_export_manifest_cannot_leak_raw_document_or_gold_evidence(stored_run):
    path = stored_run["data"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_preview"] = {"prompt": SECRET_PROMPT, "gold_spans": [SECRET_SPAN]}
    replace_json(path, manifest)
    verify_run(stored_run["root"])
    try:
        export_run(stored_run["root"], stored_run["data"], stored_run["output"])
    except ValueError:
        return  # Rejecting unsafe provenance is as valid as sanitizing it.
    for content in snapshot(stored_run["output"]).values():
        assert SECRET_PROMPT.encode() not in content
        assert SECRET_SPAN.encode() not in content


@pytest.mark.parametrize("corruption", ["dataset", "adapter"])
def test_export_rejects_wrong_dataset_or_adapter_before_output(stored_run, corruption):
    verify_run(stored_run["root"])
    if corruption == "dataset":
        path = stored_run["data"] / "test.jsonl"
        rows = copy.deepcopy(stored_run["groups"]["test"])
        rows[0]["prompt"] = "Substituted dataset with identical IDs"
        replace_jsonl(path, rows)
    else:
        (stored_run["root"] / "adapter" / "adapters.safetensors").write_bytes(b"wrong adapter")
    with pytest.raises(ValueError, match="dataset|checksum"):
        export_run(stored_run["root"], stored_run["data"], stored_run["output"])
    assert not stored_run["output"].exists()


@pytest.mark.parametrize("field,value", [("model_revision", "other-revision"),
                                         ("adapter_sha256", "other-adapter")])
def test_reload_rejects_saved_prediction_identity_substitution(stored_run, field, value):
    path = stored_run["root"] / "trained-test.jsonl"
    rows = read_predictions(path)
    for row in rows:
        row[field] = value
    replace_jsonl(path, rows)
    with pytest.raises(ValueError, match="identity|model|adapter"):
        verify_run(stored_run["root"])
    assert not (stored_run["root"] / "reload-verification.json").exists()


@pytest.mark.parametrize("field,value", [("model_revision", "other-revision"),
                                         ("fit_group_ids", ["test-document"]),
                                         ("fit_ids", ["test:0"])])
def test_reload_rejects_calibration_identity_and_document_leakage(stored_run, field, value):
    path = stored_run["root"] / "temperature.json"
    artifact = json.loads(path.read_text())
    artifact[field] = value
    replace_json(path, artifact)
    with pytest.raises(ValueError, match="identity|fit groups|fit IDs"):
        verify_run(stored_run["root"])
    assert not (stored_run["root"] / "reload-verification.json").exists()


@pytest.mark.parametrize("field,value", [("prompt", "Different contract, same ID"), ("label", "no")])
def test_reload_rejects_test_input_substitution_even_with_same_ids(stored_run, field, value):
    rows = copy.deepcopy(stored_run["groups"]["test"])
    rows[0][field] = value
    replace_jsonl(stored_run["root"] / "test-input.jsonl", rows)
    with pytest.raises(ValueError):
        verify_run(stored_run["root"])
    assert not (stored_run["root"] / "reload-verification.json").exists()


@pytest.mark.parametrize("section,metric", [("base", "accuracy"), ("trained", "macro_f1"),
                                          ("trained_calibrated", "log_loss")])
def test_retained_verifier_recomputes_metrics_despite_rehashed_report(stored_run, section, metric):
    output = export_verified(stored_run)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report[section][metric] += 0.125
    replace_json(report_path, report)
    hash_path = output / "SHA256SUMS.json"
    hashes = json.loads(hash_path.read_text())
    hashes["files"]["report.json"] = file_sha256(report_path)
    replace_json(hash_path, hashes)
    with pytest.raises(ValueError, match="does not match retained predictions"):
        retained_verifier()(output)


def test_create_only_export_and_reload_preserve_existing_bytes(stored_run):
    output = export_verified(stored_run)
    for directory, action in (
        (output, lambda: export_run(stored_run["root"], stored_run["data"], output)),
        (stored_run["root"], lambda: verify_run(stored_run["root"])),
    ):
        before = snapshot(directory)
        with pytest.raises(FileExistsError):
            action()
        assert snapshot(directory) == before
    assert stored_run["load"].call_count == 1


def test_interleaved_contract_questions_are_never_split_or_label_selected():
    rows = [{"id": f"{document}:{question}", "label": str(question % 2),
             "metadata": {"group_id": document}}
            for question in range(17) for document in ("alpha", "beta", "gamma")]
    selected = choose_subset(rows, 35)
    assert len(selected) == 34
    selected_ids = {row["id"] for row in selected}
    for document in ("alpha", "beta", "gamma"):
        complete = {row["id"] for row in rows if row["metadata"]["group_id"] == document}
        assert selected_ids & complete in (set(), complete)
    relabeled = [{**row, "label": "changed"} for row in rows]
    assert [row["id"] for row in choose_subset(relabeled, 35)] == [row["id"] for row in selected]
    with pytest.raises(ValueError, match="complete document"):
        choose_subset(rows, 16)


def test_disjoint_question_ids_do_not_hide_cross_split_document_leakage(stored_run):
    groups = copy.deepcopy(stored_run["groups"])
    groups["test"][0]["metadata"]["group_id"] = "calibration-document"
    with pytest.raises(ValueError, match="group IDs overlap"):
        validate_splits(groups)


V2_COMPARISON_FILES = (
    "base-temperature.json", "base-calibration.jsonl",
    "base-calibrated-test.jsonl", "tfidf-baseline.json",
)
RELOAD_BOUND_FILES = (
    "run.json", "test-input.jsonl", "trained-test.jsonl", "calibrated-test.jsonl",
    "temperature.json", "adapter/adapters.safetensors",
    "adapter/adapter_config.json", "adapter/training.json",
)


@pytest.fixture
def stored_v2_run(stored_run):
    root = stored_run["root"]
    calibration = [{**row, "adapter_sha256": None}
                   for row in read_predictions(root / "trained-calibration.jsonl")]
    write_jsonl(root / "base-calibration.jsonl", calibration)
    artifact = fit_prediction_calibration(calibration, root / "base-temperature.json")
    base = read_predictions(root / "base-test.jsonl")
    write_jsonl(root / "base-calibrated-test.jsonl", apply_prediction_calibration(base, artifact))
    logits, labels = prediction_arrays(base)
    classical = {"method": "synthetic fixed scores", **metrics(logits, labels)}
    write_json(root / "tfidf-baseline.json", classical)
    path = root / "report.json"
    report = json.loads(path.read_text())
    report.update(format="ifllm-learn.experiment.v2", tfidf_baseline=classical,
                  base_temperature=artifact["temperature"],
                  base_calibrated=metrics(logits, labels, temperature=artifact["temperature"]))
    replace_json(path, report)
    return stored_run


def test_v2_comparisons_export_and_recompute(stored_v2_run):
    output = export_verified(stored_v2_run)
    assert all((output / name).is_file() for name in V2_COMPARISON_FILES)
    assert retained_verifier()(output)["test_rows"] == 2


def test_reload_receipt_binds_every_required_artifact(stored_run):
    root = stored_run["root"]
    receipt = verify_run(root)
    assert receipt["format"] == "ifllm-learn.reload.v2"
    assert set(RELOAD_BOUND_FILES) <= receipt["verified_sha256"].keys()
    for name in RELOAD_BOUND_FILES:
        assert receipt["verified_sha256"][name] == file_sha256(root / name)
    assert receipt["run_sha256"] == file_sha256(root / "run.json")
    assert receipt["adapter_sha256"] == file_sha256(root / "adapter/adapters.safetensors")


@pytest.mark.parametrize("name", RELOAD_BOUND_FILES)
def test_export_rejects_post_verification_artifact_mutation(stored_run, name):
    root = stored_run["root"]
    verify_run(root)
    path = root / name
    # Even semantically inert edits invalidate the exact bytes verified earlier.
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        export_run(root, stored_run["data"], stored_run["output"])
    assert not stored_run["output"].exists()


@pytest.mark.parametrize("mutation", ["format", "adapter", "run", "bound_hash", "missing_bound_path"])
def test_export_rejects_corrupted_verification_receipt(stored_run, mutation):
    root = stored_run["root"]
    verify_run(root)
    path = root / "reload-verification.json"
    receipt = json.loads(path.read_text())
    if mutation == "format":
        receipt["format"] = "ifllm-learn.reload.v1"
    elif mutation == "adapter":
        receipt["adapter_sha256"] = "0" * 64
    elif mutation == "run":
        receipt["run_sha256"] = "0" * 64
    elif mutation == "bound_hash":
        receipt["verified_sha256"]["trained-test.jsonl"] = "0" * 64
    else:
        del receipt["verified_sha256"]["trained-test.jsonl"]
    replace_json(path, receipt)
    with pytest.raises(ValueError):
        export_run(root, stored_run["data"], stored_run["output"])
    assert not stored_run["output"].exists()


@pytest.mark.parametrize("name", V2_COMPARISON_FILES)
def test_v2_export_requires_every_declared_comparison_file(stored_v2_run, name):
    root = stored_v2_run["root"]
    verify_run(root)
    (root / name).unlink()
    # Do not let optional receipt binding alone satisfy the report-v2 requirement.
    receipt_path = root / "reload-verification.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["verified_sha256"].pop(name, None)
    replace_json(receipt_path, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        export_run(root, stored_v2_run["data"], stored_v2_run["output"])
    assert not stored_v2_run["output"].exists()


@pytest.mark.parametrize("location", ["top_level", "source"])
def test_export_rejects_unknown_manifest_fields_before_output(stored_run, location):
    verify_run(stored_run["root"])
    path = stored_run["data"] / "manifest.json"
    manifest = json.loads(path.read_text())
    if location == "top_level":
        manifest["notes"] = SECRET_PROMPT
    else:
        manifest["sources"] = [{"url": "https://example.invalid/data", "unexpected_excerpt": SECRET_SPAN}]
    replace_json(path, manifest)
    with pytest.raises(ValueError):
        export_run(stored_run["root"], stored_run["data"], stored_run["output"])
    assert not stored_run["output"].exists()


@pytest.mark.parametrize("payload", [
    {"tokenizer": {"unexpected_excerpt": SECRET_PROMPT}},
    {"exclusions": [{"reason": "duplicate", "unexpected_excerpt": SECRET_PROMPT}]},
    {"exclusions": [{"reason": "conflicting_label_maps", "members": [{"split": "train", "unexpected_excerpt": SECRET_SPAN}]}]},
    {"filter_coverage": {"test": {"unexpected_excerpt": SECRET_PROMPT}}},
    {"counts": {"train": SECRET_PROMPT}},
    {"sources": [{"quality": {"unexpected_excerpt": SECRET_PROMPT}}]},
])
def test_nested_manifest_schema_rejects_undocumented_fields(stored_run, payload):
    verify_run(stored_run["root"])
    path = stored_run["data"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(payload)
    replace_json(path, manifest)
    with pytest.raises(ValueError, match="provenance"):
        export_run(stored_run["root"], stored_run["data"], stored_run["output"])
    assert not stored_run["output"].exists()
