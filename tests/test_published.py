"""Public claims must stay synchronized with retained, recomputable results."""

import json
from pathlib import Path
import re
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("banking", "clinc", "contractnli", "bitcoin")


@pytest.mark.parametrize("name", DATASETS)
def test_published_metrics_and_hashes_recompute(name):
    verify = runpy.run_path(str(ROOT / "examples" / "verify_results.py"))["verify"]
    result = verify(ROOT / "results" / name)
    assert result["test_rows"] > 0
    receipt = json.loads((ROOT / "results" / name / "reload-verification.json").read_text())
    assert receipt["format"] == "ifllm-learn.reload.v2"
    assert receipt["rows"] == result["test_rows"]
    assert receipt["max_logit_difference"] <= 1e-4


@pytest.mark.parametrize("name", DATASETS)
def test_readme_headline_table_matches_report(name):
    text = (ROOT / "README.md").read_text()
    report = json.loads((ROOT / "results" / name / "report.json").read_text())
    line = next(line for line in text.splitlines() if line.startswith("| [") and f"results/{name}/report.json" in line)
    cells = [part.strip().replace("**", "").replace(",", "") for part in line.strip("|").split("|")]
    assert int(cells[1]) == report["base"]["count"]
    assert cells[3] == f"{report['base']['accuracy'] * 100:.2f}%"
    assert cells[4] == f"{report['trained']['accuracy'] * 100:.2f}%"
    assert cells[6] == f"{report['trained_calibrated']['log_loss']:.4f}"
    if "tfidf_baseline" in report:
        assert cells[2] == f"{report['tfidf_baseline']['accuracy'] * 100:.2f}%"
        assert cells[5] == f"{report['base_calibrated']['log_loss']:.4f}"
    else:
        assert cells[2] == cells[5] == "Not measured"


def test_readme_local_links_exist_and_examples_have_no_fake_output():
    text = (ROOT / "README.md").read_text()
    for link in re.findall(r"\]\(([^)]+)\)", text):
        if link.startswith(("http://", "https://", "#")):
            continue
        assert (ROOT / link.split("#", 1)[0]).exists(), link
    example = json.loads((ROOT / "results" / "banking-example.json").read_text())
    prediction = example["prediction"]
    assert prediction["answer"] == prediction["option_ids"][max(range(len(prediction["probabilities"])), key=prediction["probabilities"].__getitem__)]
    assert "0.9978" in text
    assert round(max(prediction["probabilities"]), 4) == 0.9978


def test_fixed_demo_budgets_and_group_counts_match_actual_runs():
    specification = json.loads((ROOT / "examples" / "demo-suite.json").read_text())
    for name, task in specification["datasets"].items():
        report = json.loads((ROOT / "results" / name / "report.json").read_text())
        assert report["run"]["seed"] == specification["seed"]
        for key, value in {**specification["shared"], **task}.items():
            if not key.endswith("_limit"):
                assert report["run"][key] == value
        if name == "contractnli":
            assert report["base"]["count"] == 170
            assert report["task_metrics"]["base"]["question_count"] == 17
            rows = [json.loads(line) for line in (ROOT / "results" / name / "base-test.jsonl").read_text().splitlines()]
            assert len({row["metadata"]["group_id"] for row in rows}) == 10
    assert not list((ROOT / "results").rglob("*.safetensors"))
