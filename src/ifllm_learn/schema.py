"""JSONL contracts shared by datasets, supervised training, and evaluation.

An example has id, prompt, question, options [{id, description}], and a ground-truth
label (option ID). Predictions use answer for the inferred option ID; label is
never part of the model input. Optional observed_at / label_available_at and
metadata fields carry provenance, never input features.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEMIF_REVISION = "ca3ba65f142967030ecb453346e94d6f476a69df"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def validate_example(row, require_label=True):
    if not isinstance(row, dict):
        raise ValueError("Each example must be a JSON object")
    required = {"id", "prompt", "question", "options"}
    if require_label:
        required.add("label")
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    for key in ("id", "question"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(row["prompt"], (str, dict, list)) or not row["prompt"]:
        raise ValueError("prompt must be a nonempty string, object, or array")
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 16:
        raise ValueError("options must contain 2-16 choices")
    ids = []
    for option in options:
        if not isinstance(option, dict) or set(option) != {"id", "description"}:
            raise ValueError("Each option must have exactly id and description")
        if any(not isinstance(option[key], str) or not option[key].strip() for key in ("id", "description")):
            raise ValueError("Option IDs and descriptions must be nonempty strings")
        ids.append(option["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Option IDs must be unique")
    if "label" in row and (not isinstance(row["label"], str) or row["label"] not in ids):
        raise ValueError("label must identify a declared option")
    if "metadata" in row:
        if not isinstance(row["metadata"], dict):
            raise ValueError("metadata must be a JSON object")
        for key in ("group_id", "question_id"):
            if key in row["metadata"] and (not isinstance(row["metadata"][key], str) or not row["metadata"][key].strip()):
                raise ValueError(f"metadata.{key} must be a nonempty string")
    for key in ("observed_at", "label_available_at"):
        if key in row:
            if not isinstance(row[key], str):
                raise ValueError(f"{key} must be an ISO-8601 timestamp")
            timestamp = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError(f"{key} must include a timezone")
    if "observed_at" in row and "label_available_at" in row:
        observed = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
        available = datetime.fromisoformat(row["label_available_at"].replace("Z", "+00:00"))
        if available <= observed:
            raise ValueError("label_available_at must be after observed_at")
    canonical_json(row)
    return row


def to_semif(row):
    validate_example(row, require_label=False)
    return {"id": row["id"], "state": row["prompt"], "question": row["question"], "options": row["options"]}


def label_index(row):
    validate_example(row)
    return [option["id"] for option in row["options"]].index(row["label"])


def load_jsonl(path, require_label=True):
    rows = []
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = validate_example(json.loads(line), require_label=require_label)
                if row["id"] in ids:
                    raise ValueError(f"Duplicate example ID: {row['id']}")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def contract_hash(rows):
    """Identify task/question/ordered-choice contracts, independent of ground truth."""
    contracts = set()
    for row in rows:
        validate_example(row, require_label=False)
        contracts.add(canonical_json({"question": row["question"], "options": row["options"]}))
    if not contracts:
        raise ValueError("Cannot identify an empty task contract")
    return hashlib.sha256(canonical_json(sorted(contracts)).encode()).hexdigest()


def assert_disjoint(*groups):
    seen = set()
    for rows in groups:
        ids = {row["id"] for row in rows}
        if len(ids) != len(rows):
            raise ValueError("Duplicate IDs within a split")
        if seen & ids:
            raise ValueError("Example IDs overlap across dataset splits")
        seen.update(ids)
    seen_groups = set()
    for rows in groups:
        group_ids = {row.get("metadata", {}).get("group_id") for row in rows} - {None}
        if seen_groups & group_ids:
            raise ValueError("Document/group IDs overlap across dataset splits")
        seen_groups.update(group_ids)


def assert_temporal_separation(*groups):
    rows = [row for group in groups for row in group]
    if not any("observed_at" in row or "label_available_at" in row for row in rows):
        return
    if not all("observed_at" in row and "label_available_at" in row for row in rows):
        raise ValueError("Temporal experiments require both timestamps on every row")
    previous_labels = None
    for group in groups:
        observed = [datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00")) for row in group]
        available = [datetime.fromisoformat(row["label_available_at"].replace("Z", "+00:00")) for row in group]
        if not observed or any(right <= left for left, right in zip(observed, observed[1:])):
            raise ValueError("Temporal splits must be nonempty and strictly chronological")
        if previous_labels is not None and previous_labels >= min(observed):
            raise ValueError("Future labels cross a split boundary; purge overlapping horizons")
        previous_labels = max(available)
