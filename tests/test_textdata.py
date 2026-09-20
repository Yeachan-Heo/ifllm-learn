import csv
import hashlib
import io
import json
import zipfile

import pytest

from ifllm_learn import textdata
from ifllm_learn.schema import to_semif
from ifllm_learn.workflow import SPLITS, load_splits


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)


def banking_sources():
    sources = {}
    categories = list(textdata.BANKING_DESCRIPTIONS) + [f"unused_{index}" for index in range(67)]
    sources[textdata.BANKING_BASE + "categories.json"] = json.dumps(categories).encode()
    for split, count in (("train", 20), ("test", 3)):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["text", "category"])
        for label in textdata.BANKING_DESCRIPTIONS:
            writer.writerows((f"{split} {label} utterance {index}", label) for index in range(count))
        sources[textdata.BANKING_BASE + split + ".csv"] = stream.getvalue().encode()
    return sources


def contract_source():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        labels = {f"nda-{index}": {"hypothesis": f"Hypothesis {index}"} for index in range(17)}
        for split, count in (("train", 4), ("dev", 8), ("test", 4)):
            documents = []
            for index in range(count):
                documents.append({"id": f"{split}-{index}", "text": f"Full contract {split} {index}",
                                  "spans": [[0, 7]], "annotation_sets": [{"annotations": {
                                      key: {"choice": ("Entailment", "Contradiction", "NotMentioned")[index % 3],
                                            "spans": [987654]} for key in labels}}]})
            archive.writestr(f"contract-nli/{split}.json", json.dumps({"labels": labels, "documents": documents}))
    return stream.getvalue()


def clinc_source():
    data = {}
    for split, count in (("train", 10), ("val", 2), ("test", 3)):
        data[split] = [[f"{split} {label} {index}", label]
                       for label in list(textdata.CLINC_DESCRIPTIONS)[:-1] for index in range(count)]
        data["oos_" + split] = [[f"outside {split} {index}", "oos"] for index in range(1000 if split == "test" else 10)]
    return json.dumps(data).encode()


def patch_sources(monkeypatch, sources):
    def download(url, cache_dir, revision, license_name, attribution):
        content = sources[url]
        return content, {"url": url, "revision": revision, "license": license_name,
                         "sha256": hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(textdata, "_download", download)


@pytest.mark.parametrize("name", ["banking", "clinc", "contractnli"])
def test_deterministic_groups_and_fixed_contracts(tmp_path, monkeypatch, name):
    sources = banking_sources() if name == "banking" else {textdata.CLINC_URL: clinc_source()} if name == "clinc" else {textdata.CONTRACT_URL: contract_source()}
    patch_sources(monkeypatch, sources)
    first = textdata.prepare_text_dataset(name, tmp_path / "first", tmp_path / "cache", tokenizer=Tokenizer(), max_tokens=100000)
    second = textdata.prepare_text_dataset(name, tmp_path / "second", tmp_path / "cache", tokenizer=Tokenizer(), max_tokens=100000)
    assert first == second
    groups = load_splits(tmp_path / "first")
    seen = set()
    for split in SPLITS:
        membership = set(first["group_membership"][split])
        assert not seen & membership
        seen |= membership
        for row in groups[split]:
            assert isinstance(row["prompt"], str)
            assert "observed_at" not in row and "label_available_at" not in row
            assert set(to_semif(row)) == {"id", "state", "question", "options"}
    if name == "banking":
        assert first["benchmark"] == "BANKING77-10"
        assert first["classes"] == list(textdata.BANKING_DESCRIPTIONS)
        assert first["counts"] == {"train": 140, "valid": 30, "calibration": 30, "test": 30}
    elif name == "clinc":
        assert first["classes"] == list(textdata.CLINC_DESCRIPTIONS)
        assert first["class_counts"]["test"]["out_of_scope"] == 1000
        assert first["counts"]["test"] == 1030
    else:
        assert first["classes"] == list(textdata.NLI_DESCRIPTIONS)
        assert all(len(rows) == 68 for rows in groups.values())
        assert all("987654" not in json.dumps(to_semif(row)) for rows in groups.values() for row in rows)
        assert all(row["metadata"]["question_id"] == row["id"].rsplit(":", 1)[-1] for rows in groups.values() for row in rows)
    with pytest.raises(FileExistsError):
        textdata.prepare_text_dataset(name, tmp_path / "first", tmp_path / "cache", tokenizer=Tokenizer())


def test_contract_whole_document_filter_without_gold_input(tmp_path, monkeypatch):
    import semif_phase1.direct
    patch_sources(monkeypatch, {textdata.CONTRACT_URL: contract_source()})
    calls = []
    def encode(tokenizer, row, max_tokens):
        assert set(row) == {"id", "state", "question", "options"}
        assert "987654" not in json.dumps(row)
        calls.append(row["id"])
        too_long = row["id"] == "contractnli:train-0:nda-9"
        return list(range(101 if too_long else 5)), [1, 2, 3], "fixture"
    monkeypatch.setattr(semif_phase1.direct, "encode_prompt", encode)
    manifest = textdata.prepare_text_dataset("contractnli", tmp_path / "out", tmp_path / "cache", tokenizer=Tokenizer(), max_tokens=100)
    rows = load_splits(tmp_path / "out")["train"]
    assert len(rows) == 51
    assert not any(row["metadata"]["group_id"] == "contractnli:train-0" for row in rows)
    assert manifest["filter_coverage"]["train"]["token_excluded_groups"] == 1
    assert len(calls) == 16 * 17
    assert len(manifest["exclusions"][0]["ids"]) == 17


def test_dedup_priority_normalization_and_conflicts():
    groups = {split: [] for split in SPLITS}
    for split in SPLITS:
        text = "  SAME\nText " if split == "train" else "same text"
        groups[split] = [[textdata._row(text, "entailment", "fixture", split, "Q", textdata.NLI_DESCRIPTIONS)]]
    result, excluded = textdata._deduplicate(groups)
    assert len(result["test"]) == 1
    assert all(not result[split] for split in ("train", "valid", "calibration"))
    assert len(excluded) == 3
    groups["train"][0][0]["label"] = "contradiction"
    with pytest.raises(ValueError, match="Conflicting"):
        textdata._deduplicate(groups)


def test_contract_duplicate_compares_every_hypothesis():
    groups = textdata._contract(contract_source(), 17)
    duplicate = json.loads(json.dumps(groups["test"][0]))
    duplicate[-1]["label"] = "not_mentioned"
    groups["train"].append(duplicate)
    with pytest.raises(ValueError, match="Conflicting"):
        textdata._deduplicate(groups)


@pytest.mark.parametrize("name,content", [("contractnli", b"not a zip"), ("clinc", b"{}")])
def test_invalid_source_rejected_before_output(tmp_path, monkeypatch, name, content):
    patch_sources(monkeypatch, {textdata.CONTRACT_URL if name == "contractnli" else textdata.CLINC_URL: content})
    with pytest.raises(ValueError):
        textdata.prepare_text_dataset(name, tmp_path / "out", tmp_path / "cache", tokenizer=Tokenizer())
    assert not (tmp_path / "out").exists()


def test_invalid_banking_categories_and_empty_filter(tmp_path, monkeypatch):
    sources = banking_sources()
    sources[textdata.BANKING_BASE + "categories.json"] = b"[]"
    patch_sources(monkeypatch, sources)
    with pytest.raises(ValueError, match="categories"):
        textdata.prepare_text_dataset("banking", tmp_path / "bad", tmp_path / "cache", tokenizer=Tokenizer())
    patch_sources(monkeypatch, banking_sources())
    with pytest.raises(ValueError, match="Empty"):
        textdata.prepare_text_dataset("banking", tmp_path / "empty", tmp_path / "cache", tokenizer=Tokenizer(), max_tokens=1)
    assert not (tmp_path / "empty").exists()


def test_download_cache_hash_and_bounds(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def geturl(self):
            return textdata.CLINC_URL
    calls = []
    def open_url(request, timeout):
        calls.append(request.full_url)
        return Response(b"fixture")
    monkeypatch.setattr(textdata, "urlopen", open_url)
    args = (textdata.CLINC_URL, tmp_path, textdata.CLINC_REVISION, "CC-BY-3.0", "fixture")
    content, receipt = textdata._download(*args)
    assert content == b"fixture"
    assert receipt["sha256"] == hashlib.sha256(content).hexdigest()
    assert textdata._download(*args) == (content, receipt)
    assert len(calls) == 1
    next(tmp_path.glob("*.source")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        textdata._download(*args)
    monkeypatch.setattr(textdata, "MAX_DOWNLOAD_BYTES", 2)
    with pytest.raises(ValueError, match="bound"):
        textdata._download(textdata.CLINC_URL, tmp_path / "bounded", textdata.CLINC_REVISION, "CC-BY-3.0", "fixture")


def test_invalid_tokenizer_errors_are_not_silently_filtered(tmp_path, monkeypatch):
    patch_sources(monkeypatch, banking_sources())
    class BrokenTokenizer(Tokenizer):
        def decode(self, ids):
            return "wrong"
    with pytest.raises(ValueError, match="round-trip"):
        textdata.prepare_text_dataset("banking", tmp_path / "out", tmp_path / "cache", tokenizer=BrokenTokenizer())
    assert not (tmp_path / "out").exists()


def test_contract_conflict_component_excludes_every_member_and_audits_manifest(tmp_path, monkeypatch):
    source = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(contract_source())) as original, zipfile.ZipFile(source, "w") as modified:
        documents = {name: json.loads(original.read(name)) for name in original.namelist()}
        test_doc = documents["contract-nli/test.json"]["documents"][0]
        train_doc = documents["contract-nli/train.json"]["documents"][0]
        train_doc["text"] = test_doc["text"]
        train_doc["annotation_sets"][0]["annotations"]["nda-12"]["choice"] = "Contradiction"
        # A third member agrees with test but must also be excluded, not retained by priority.
        valid_doc = documents["contract-nli/dev.json"]["documents"][0]
        valid_doc["text"] = test_doc["text"]
        for name, value in documents.items():
            modified.writestr(name, json.dumps(value))
    content = source.getvalue()
    groups = textdata._contract(content, 17)
    with pytest.raises(ValueError, match="Conflicting"):
        textdata._deduplicate(groups)
    patch_sources(monkeypatch, {textdata.CONTRACT_URL: content})
    manifest = textdata.prepare_text_dataset("contractnli", tmp_path / "out", tmp_path / "cache",
                                             tokenizer=Tokenizer(), max_tokens=100000)
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == manifest
    conflicts = [item for item in manifest["exclusions"] if item["reason"] == "conflicting_label_maps"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["action"] == "exclude_entire_component"
    assert conflict["excluded_groups"] == 3
    assert conflict["excluded_rows"] == 51
    assert "retained_id" not in conflict
    expected = {rows[0]["metadata"]["group_id"]: (split, rows)
                for split, items in groups.items() for rows in items
                if rows[0]["metadata"]["group_id"] in {"contractnli:test-0", "contractnli:train-0", "contractnli:dev-0"}}
    assert {member["group_id"] for member in conflict["members"]} == set(expected)
    assert len({member["label_map_sha256"] for member in conflict["members"]}) == 2
    for member in conflict["members"]:
        split, rows = expected[member["group_id"]]
        assert member["split"] == split
        assert member["ids"] == [row["id"] for row in rows]
        signature = {row["question"]: row["label"] for row in rows}
        digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert member["label_map_sha256"] == digest
    retained = load_splits(tmp_path / "out")
    assert all(row["metadata"]["group_id"] not in expected for rows in retained.values() for row in rows)
    assert sum(manifest["counts"].values()) == 13 * 17
    assert "exclude entire conflicting component" in manifest["deduplication_policy"]


def test_transitive_conflict_exclusion_drops_agreeing_bridge():
    groups = {split: [] for split in SPLITS}
    for split, text, group, label in (("test", "first text", "a", "entailment"),
                                     ("valid", "first text", "b", "entailment"),
                                     ("train", "second text", "b", "contradiction")):
        groups[split].append([textdata._row(text, label, "fixture", split, "Q", textdata.NLI_DESCRIPTIONS, group)])
    with pytest.raises(ValueError, match="Conflicting"):
        textdata._deduplicate(groups)
    retained, exclusions = textdata._deduplicate(groups, exclude_conflicts=True)
    assert all(not rows for rows in retained.values())
    assert len(exclusions) == 1
    assert exclusions[0]["excluded_groups"] == 3
    assert {member["split"] for member in exclusions[0]["members"]} == {"test", "valid", "train"}
