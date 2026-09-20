"""Visual demos must derive from retained predictions, including regressions."""

import hashlib
import json
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "examples" / "build_visual_demo.py"))


def test_visual_assets_are_deterministic_and_current():
    assets = BUILDER["rendered_assets"](ROOT)
    assert assets == BUILDER["rendered_assets"](ROOT)
    for path, content in assets.items():
        assert path.read_text() == content, f"Regenerate {path.name} from measured results"


def test_visual_data_source_hashes_match_actual_inputs():
    data = json.loads((ROOT / "demo" / "data.json").read_text())
    assert data["mode"] == "Recorded experiments, not live inference"
    for filename, expected in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    assert sum(dataset["test_count"] for dataset in data["datasets"]) == 1902


@pytest.mark.parametrize("name", ["banking", "clinc", "contractnli", "bitcoin"])
def test_visual_predictions_are_complete_and_match_measured_metrics(name):
    data = json.loads((ROOT / "demo" / "data.json").read_text())
    dataset = next(item for item in data["datasets"] if item["id"] == name)
    assert len(dataset["rows"]) == dataset["test_count"]
    assert len({row["id"] for row in dataset["rows"]}) == dataset["test_count"]
    report = json.loads((ROOT / "results" / name / "report.json").read_text())
    for model in dataset["models"]:
        assert model["accuracy"] == report[model["id"]]["accuracy"]
        accuracy = np.mean([row["models"][model["id"]]["answer"] == row["label"] for row in dataset["rows"]])
        assert accuracy == pytest.approx(model["accuracy"])
        assert sum(sum(row) for row in model["confusion"]) == dataset["test_count"]
        assert sum(item["count"] for item in model["reliability"]) == dataset["test_count"]
        for row in dataset["rows"]:
            prediction = row["models"][model["id"]]
            probabilities = prediction["probabilities"]
            assert len(probabilities) == len(dataset["choices"])
            assert sum(probabilities) == pytest.approx(1)
            assert prediction["answer"] == dataset["choices"][int(np.argmax(probabilities))]
    if name == "bitcoin":
        assert "base_calibrated" not in {model["id"] for model in dataset["models"]}
        assert dataset["models"][0]["accuracy"] == dataset["prior_accuracy"]
    if name == "clinc":
        models = {model["id"]: model for model in dataset["models"]}
        assert models["trained"]["accuracy"] < models["base"]["accuracy"]


def test_visual_data_does_not_invent_or_distribute_benchmark_input_text():
    data = json.loads((ROOT / "demo" / "data.json").read_text())
    for dataset in data["datasets"]:
        for row in dataset["rows"]:
            assert set(row) == {"id", "label", "models"}
            assert all(set(value) == {"answer", "probabilities"} for value in row["models"].values())


def test_readme_graphic_is_accessible_and_discloses_scope():
    svg = (ROOT / "docs" / "demo-overview.svg").read_text()
    root = ET.fromstring(svg)
    assert root.attrib["role"] == "img"
    assert root.find("{http://www.w3.org/2000/svg}title").text
    assert root.find("{http://www.w3.org/2000/svg}desc").text
    assert "not a robust gain" in svg and "Bitcoin did not beat" in svg
    assert "NO LIVE BROWSER INFERENCE" in svg


def test_browser_verification_receipt_matches_published_ui():
    receipt = json.loads((ROOT / "docs" / "browser-verification.json").read_text())
    assert receipt["status"] == "passed" and receipt["modelCases"] == 15
    assert receipt["externalRequests"] == receipt["runtimeErrors"] == []
    assert {row["dataset"] for row in receipt["mobileLayouts"]} == {item[0] for item in BUILDER["DATASETS"]}
    assert all(row["width"] == row["scrollWidth"] == 390 for row in receipt["mobileLayouts"])
    assert all(case["passed"] for case in receipt["failureCases"])
    for filename, expected in receipt["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    for name in ("demo-dashboard.png", "demo-mobile.png"):
        assert (ROOT / "docs" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_dashboard_resource_dependencies_are_local():
    from html.parser import HTMLParser
    from urllib.parse import urlsplit

    resources = []
    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "script" and "src" in values:
                resources.append(values["src"])
            if tag == "link" and values.get("rel") in ("stylesheet", "icon"):
                resources.append(values["href"])
    Resources().feed((ROOT / "demo" / "index.html").read_text())
    assert set(resources) == {"app.js", "styles.css", "favicon.svg"}
    for resource in resources:
        assert not urlsplit(resource).scheme and (ROOT / "demo" / resource).is_file()
    assert "@import" not in (ROOT / "demo" / "styles.css").read_text()
