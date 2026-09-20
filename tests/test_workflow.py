import json
from datetime import datetime, timedelta, timezone

import pytest

from ifllm_learn.cli import main
from ifllm_learn.metrics import softmax
from ifllm_learn.schema import contract_hash, file_sha256, load_jsonl, to_semif, validate_example, write_jsonl
from ifllm_learn.workflow import SPLITS, choose_subset, load_splits, validate_splits

ROW = {"id": "sample", "prompt": {"past": [1, 2, 3]}, "question": "Increasing?",
       "options": [{"id": "yes", "description": "Increasing"}, {"id": "no", "description": "Not increasing"}],
       "label": "yes", "metadata": {"future_return": 0.1}}


def test_model_input_never_contains_ground_truth_or_metadata():
    model_input = to_semif(ROW)
    assert set(model_input) == {"id", "state", "question", "options"}
    assert model_input["state"] == ROW["prompt"]
    assert contract_hash([ROW]) == contract_hash([{**ROW, "label": "no", "metadata": {"future_return": -9}}])
    assert contract_hash([ROW]) != contract_hash([{**ROW, "question": "Different?"}])


@pytest.mark.parametrize("change", [{"label": "missing"}, {"prompt": {}}, {"prompt": float("nan")},
                                   {"options": ROW["options"][:1]}, {"options": [ROW["options"][0]] * 2},
                                   {"observed_at": "2025-01-01T00:00:00"}])
def test_invalid_examples_fail(change):
    with pytest.raises(ValueError):
        validate_example({**ROW, **change})


def groups_with_times():
    groups = {}
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for index, name in enumerate(SPLITS):
        observed = start + timedelta(days=index * 10)
        groups[name] = [{**ROW, "id": name, "observed_at": observed.isoformat(),
                         "label_available_at": (observed + timedelta(days=1)).isoformat()}]
    return groups


def test_temporal_boundary_overlap_and_id_leakage_rejected():
    groups = groups_with_times()
    validate_splits(groups)
    groups["train"][0]["label_available_at"] = groups["valid"][0]["observed_at"]
    with pytest.raises(ValueError, match="boundary"):
        validate_splits(groups)
    groups = groups_with_times()
    groups["test"][0]["id"] = groups["train"][0]["id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(groups)


def test_jsonl_create_only_and_duplicate_rejection(tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [ROW])
    assert load_jsonl(path) == [ROW]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [ROW])
    duplicate = tmp_path / "duplicate.jsonl"
    write_jsonl(duplicate, [ROW, ROW])
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl(duplicate)


def test_subsets_are_time_spread_without_labels():
    rows = [{"id": str(i)} for i in range(10)]
    assert [row["id"] for row in choose_subset(rows, 3)] == ["0", "4", "9"]
    with pytest.raises(ValueError):
        choose_subset(rows, 0)


def test_load_all_splits_and_checksum_guard(tmp_path):
    groups = groups_with_times()
    for name in SPLITS:
        write_jsonl(tmp_path / f"{name}.jsonl", groups[name])
    assert load_splits(tmp_path) == groups
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"train.jsonl": "bad"}}))
    with pytest.raises(ValueError, match="checksum"):
        load_splits(tmp_path)


@pytest.mark.parametrize("case", ["missing", "empty", "omitted", "extra", "path_escape",
                                  "null", "list", "string", "numeric_hash", "null_hash",
                                  "short_hash", "nonhex_hash", "mismatch", "nonobject"])
def test_manifest_requires_complete_valid_checksums(tmp_path, case):
    groups = groups_with_times()
    for name in SPLITS:
        write_jsonl(tmp_path / f"{name}.jsonl", groups[name])
    checksums = {f"{name}.jsonl": file_sha256(tmp_path / f"{name}.jsonl") for name in SPLITS}
    manifest = {"sha256": checksums}
    if case == "missing":
        manifest = {}
    elif case == "empty":
        manifest["sha256"] = {}
    elif case == "omitted":
        del checksums["test.jsonl"]
    elif case in ("extra", "path_escape"):
        checksums["extra.jsonl" if case == "extra" else "../test.jsonl"] = "0" * 64
    elif case in ("null", "list", "string"):
        manifest["sha256"] = {"null": None, "list": [], "string": "invalid"}[case]
    elif case == "nonobject":
        manifest = []
    else:
        checksums["test.jsonl"] = {"numeric_hash": 123, "null_hash": None, "short_hash": "a" * 63,
                                    "nonhex_hash": "g" * 64, "mismatch": "0" * 64}[case]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_splits(tmp_path)


@pytest.mark.parametrize("uppercase", [False, True])
def test_complete_manifest_checksums_accept_and_detect_changed_split(tmp_path, uppercase):
    groups = groups_with_times()
    for name in SPLITS:
        write_jsonl(tmp_path / f"{name}.jsonl", groups[name])
    checksums = {f"{name}.jsonl": file_sha256(tmp_path / f"{name}.jsonl") for name in SPLITS}
    if uppercase:
        checksums = {name: value.upper() for name, value in checksums.items()}
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": checksums}))
    assert load_splits(tmp_path) == groups
    with (tmp_path / "test.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_splits(tmp_path)


def test_custom_splits_without_manifest_are_accepted(tmp_path):
    groups = groups_with_times()
    for name in SPLITS:
        write_jsonl(tmp_path / f"{name}.jsonl", groups[name])
    assert not (tmp_path / "manifest.json").exists()
    assert load_splits(tmp_path) == groups


def test_calibrate_and_evaluate_cli_without_model(tmp_path, capsys):
    rows = [{"id": f"cal-{i}", "answer": "yes", "label": label,
             "option_ids": ["yes", "no"], "logits": [5, 0], "probabilities": softmax([5, 0]).tolist(),
             "model_revision": "test", "adapter_sha256": None, "contract_sha256": "task", "precision": "fp32"}
            for i, label in enumerate(["yes", "no", "yes", "no"])]
    inputs = tmp_path / "cal.jsonl"
    artifact = tmp_path / "temperature.json"
    output = tmp_path / "evaluation.json"
    write_jsonl(inputs, rows)
    main(["calibrate", "--input", str(inputs), "--output", str(artifact)])
    assert json.loads(capsys.readouterr().out)["temperature"] > 1
    test = tmp_path / "test.jsonl"
    write_jsonl(test, [{**row, "id": row["id"].replace("cal-", "test-")} for row in rows])
    main(["evaluate", "--input", str(test), "--calibration", str(artifact), "--output", str(output)])
    assert json.loads(output.read_text())["accuracy"] == 0.5
    assert json.loads(output.read_text())["log_loss"] < 0.7
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--input", str(inputs), "--calibration", str(artifact), "--output", str(tmp_path / "leaked.json")])
    assert error.value.code == 2
