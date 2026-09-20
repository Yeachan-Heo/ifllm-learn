
import pytest

from ifllm_learn.calibration import apply_prediction_calibration, fit_prediction_calibration
from ifllm_learn.schema import assert_disjoint
from ifllm_learn.workflow import choose_subset


def test_document_subsets_keep_all_questions():
    rows = [{"id": f"{doc}:{q}", "metadata": {"group_id": doc}} for doc in ("a", "b", "c") for q in range(17)]
    picked = choose_subset(rows, 35)
    assert len(picked) == 34
    assert {row["metadata"]["group_id"] for row in picked} == {"a", "c"}
    with pytest.raises(ValueError, match="complete document"):
        choose_subset(rows, 16)
    with pytest.raises(ValueError, match="equal-sized"):
        choose_subset(rows[:-1], 34)


def test_cross_question_same_document_cannot_cross_splits():
    with pytest.raises(ValueError, match="group IDs overlap"):
        assert_disjoint([{"id": "a:1", "metadata": {"group_id": "a"}}],
                        [{"id": "a:2", "metadata": {"group_id": "a"}}])


def test_calibration_rejects_another_question_from_fit_document(tmp_path):
    from test_metrics import predictions
    rows = predictions()
    for row in rows:
        row["metadata"] = {"group_id": "same-contract"}
    artifact = fit_prediction_calibration(rows, tmp_path / "temperature.json")
    future = predictions("different-question")
    for row in future:
        row["metadata"] = {"group_id": "same-contract"}
    with pytest.raises(ValueError, match="fit groups"):
        apply_prediction_calibration(future, artifact)


def test_gradient_accumulation_counts_real_examples(tmp_path):
    pytest.importorskip("mlx.core")
    from test_training import ROW, tiny_bundle
    from ifllm_learn.training import train_adapter
    train = [{**ROW, "id": f"train-{i}"} for i in range(3)]
    valid = [{**ROW, "id": "valid"}]
    result = train_adapter(train, valid, tmp_path / "adapter", bundle=tiny_bundle(),
                           steps=2, batch_size=2, learning_rate=1e-4, num_layers=2, rank=2, max_tokens=1024)
    assert result["examples_seen"] == 4
    assert result["batch_size"] == 2
    assert all(len(item["example_ids"]) == 2 for item in result["history"])
    assert set(result["history"][0]["example_ids"] + result["history"][1]["example_ids"]) == {row["id"] for row in train}


def test_invalid_batch_fails_before_output(tmp_path):
    from test_training import ROW
    from ifllm_learn.training import train_adapter
    with pytest.raises(ValueError, match="batch_size"):
        train_adapter([ROW], [{**ROW, "id": "valid"}], tmp_path / "adapter", batch_size=0)
    assert not (tmp_path / "adapter").exists()


def test_export_rejects_incomplete_run_without_creating_output(tmp_path):
    from ifllm_learn.evidence import export_run
    with pytest.raises(FileNotFoundError):
        export_run(tmp_path / "run", tmp_path / "data", tmp_path / "published")
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("metadata", [None, [], "document", {"group_id": []}, {"question_id": ""}])
def test_bad_metadata_rejected_at_schema_boundary(metadata):
    from test_training import ROW
    from ifllm_learn.schema import validate_example
    with pytest.raises(ValueError, match="metadata"):
        validate_example({**ROW, "metadata": metadata})
