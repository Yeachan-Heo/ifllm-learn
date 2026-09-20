"""Reproducible, leakage-controlled English text benchmark preparation."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time
import unicodedata
from urllib.request import Request, urlopen
import zipfile

from .schema import DEFAULT_MODEL, DEFAULT_REVISION, SEMIF_REVISION, file_sha256, to_semif, write_json, write_jsonl
from .workflow import SPLITS, validate_splits

BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CONTRACT_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
BANKING_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING_REVISION}/banking_data/"
CLINC_URL = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/data_full.json"
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024

BANKING_DESCRIPTIONS = {
    "card_not_working": "The payment card is not working.",
    "contactless_not_working": "Contactless card payments are not working.",
    "declined_card_payment": "A card payment was declined.",
    "pending_card_payment": "A card payment is still pending.",
    "card_payment_not_recognised": "The customer does not recognise a card payment.",
    "transaction_charged_twice": "A transaction was charged twice.",
    "card_payment_fee_charged": "A fee was charged for a card payment.",
    "card_payment_wrong_exchange_rate": "A card payment used the wrong exchange rate.",
    "reverted_card_payment?": "A card payment was reverted or reversed.",
    "Refund_not_showing_up": "An expected refund has not appeared.",
}
CLINC_DESCRIPTIONS = {
    "freeze_account": "Freeze or suspend an account.",
    "routing": "Find a bank routing number.",
    "pin_change": "Change an account or card PIN.",
    "bill_due": "Find when a bill is due.",
    "pay_bill": "Make a bill payment.",
    "account_blocked": "Resolve a blocked account.",
    "interest_rate": "Ask about an interest rate.",
    "min_payment": "Find the minimum required payment.",
    "bill_balance": "Find the outstanding bill balance.",
    "transfer": "Transfer money between accounts.",
    "out_of_scope": "The request is outside the ten listed banking intents.",
}
NLI_DESCRIPTIONS = {
    "entailment": "The contract entails the hypothesis.",
    "contradiction": "The contract contradicts the hypothesis.",
    "not_mentioned": "The contract neither entails nor contradicts the hypothesis.",
}
BANKING_QUESTION = "Which card-payment support intent best describes this customer message?"
CLINC_QUESTION = "Which listed banking intent describes this request, or is it out of scope?"


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Source text must be a nonempty string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _download(url, cache_dir, revision, license_name, attribution):
    """Cache bytes with a URL-bound receipt; verify every reuse, never silently repair."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (_digest(url) + ".source")
    receipt_path = path.with_suffix(".json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError("Incomplete source cache")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if path.stat().st_size > MAX_DOWNLOAD_BYTES or receipt.get("url") != url or receipt.get("sha256") != file_sha256(path):
            raise ValueError("Source cache checksum mismatch")
        content = path.read_bytes()
    else:
        started = time.monotonic()
        chunks, size = [], 0
        with urlopen(Request(url, headers={"User-Agent": "ifllm-learn/1"}), timeout=30) as response:
            if response.geturl() != url:
                raise ValueError("Unexpected source redirect")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES or time.monotonic() - started > 120:
                    raise ValueError("Source download exceeds size or time bound")
                chunks.append(chunk)
        content = b"".join(chunks)
        if not content:
            raise ValueError("Empty source download")
        with path.open("xb") as stream:
            stream.write(content)
        receipt = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
        write_json(receipt_path, receipt)
    return content, {**receipt, "bytes": len(content), "revision": revision,
                     "license": license_name, "attribution": attribution,
                     "license_url": "https://creativecommons.org/licenses/by/3.0/" if license_name == "CC-BY-3.0" else "https://creativecommons.org/licenses/by/4.0/"}


def _json(content):
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("Source JSON exceeds size bound")
    return json.loads(content)


def _row(text, label, source, source_id, question, descriptions, group_id=None):
    normalized = _normalized(text)
    group_id = group_id or _digest(normalized)
    return {"id": f"{source}:{source_id}", "prompt": text, "question": question,
            "options": [{"id": key, "description": value} for key, value in descriptions.items()],
            "label": label, "metadata": {"group_id": group_id, "source": source}}


def _partition(items, seed, fractions):
    shuffled = sorted(items, key=lambda item: item[0]["id"])
    random.Random(seed).shuffle(shuffled)
    boundaries = [0] + [int(len(shuffled) * fraction) for fraction in fractions] + [len(shuffled)]
    return [shuffled[left:right] for left, right in zip(boundaries, boundaries[1:])]


def _stratify(items, seed, fractions):
    classes = defaultdict(list)
    for item in items:
        classes[item[0]["label"]].append(item)
    parts = [[] for _ in range(len(fractions) + 1)]
    for label in sorted(classes):
        for destination, values in zip(parts, _partition(classes[label], f"{seed}:{label}", fractions)):
            destination.extend(values)
    return parts


def _contract(content, seed):
    groups = {name: [] for name in SPLITS}
    reference = None
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for source_split in ("train", "dev", "test"):
            members = [info for info in archive.infolist() if Path(info.filename).name == f"{source_split}.json"]
            if len(members) != 1 or members[0].file_size > MAX_JSON_BYTES:
                raise ValueError(f"Invalid ContractNLI {source_split} archive member")
            data = _json(archive.read(members[0]))
            labels = data["labels"]
            if not isinstance(labels, dict) or len(labels) != 17:
                raise ValueError("ContractNLI requires exactly 17 hypotheses")
            hypotheses = {key: value["hypothesis"] for key, value in labels.items()}
            if any(not isinstance(value, str) or not value.strip() for value in hypotheses.values()):
                raise ValueError("Invalid ContractNLI hypothesis")
            if reference is not None and hypotheses != reference:
                raise ValueError("ContractNLI hypotheses differ across source splits")
            reference = hypotheses
            documents = []
            for doc in data["documents"]:
                annotations = doc["annotation_sets"]
                if len(annotations) != 1 or set(annotations[0]["annotations"]) != set(labels):
                    raise ValueError("Invalid ContractNLI annotations")
                choices = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}
                rows = []
                for key in sorted(labels):
                    choice = annotations[0]["annotations"][key]["choice"]
                    if choice not in choices:
                        raise ValueError("Invalid ContractNLI choice")
                    rows.append(_row(doc["text"], choices[choice], "contractnli", f"{doc['id']}:{key}",
                                     "Does the contract support this hypothesis?\n" + hypotheses[key],
                                     NLI_DESCRIPTIONS, f"contractnli:{doc['id']}"))
                    rows[-1]["metadata"]["question_id"] = key
                documents.append(rows)
            if source_split == "dev":
                groups["valid"], groups["calibration"] = _partition(documents, seed, [0.5])
            else:
                groups[source_split] = documents
    return groups


def _banking(contents, seed):
    categories = _json(contents["categories.json"])
    if not isinstance(categories, list) or len(categories) != 77 or len(set(categories)) != 77 or not set(BANKING_DESCRIPTIONS) <= set(categories):
        raise ValueError("Invalid BANKING77 categories")
    groups = {name: [] for name in SPLITS}
    for split in ("train", "test"):
        reader = csv.DictReader(io.StringIO(contents[f"{split}.csv"].decode("utf-8-sig")))
        if reader.fieldnames != ["text", "category"]:
            raise ValueError("Invalid BANKING77 CSV header")
        for index, item in enumerate(reader):
            if set(item) != {"text", "category"} or item["category"] not in categories:
                raise ValueError("Invalid BANKING77 source row")
            _normalized(item["text"])
            if item["category"] in BANKING_DESCRIPTIONS:
                groups[split].append([_row(item["text"], item["category"], "banking", f"{split}:{index}", BANKING_QUESTION, BANKING_DESCRIPTIONS)])
    groups["train"], groups["valid"], groups["calibration"] = _stratify(groups["train"], seed, [0.70, 0.85])
    return groups


def _clinc(content, seed):
    data = _json(content)
    expected = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Invalid CLINC source splits")
    if len(data["oos_test"]) != 1000:
        raise ValueError("CLINC requires all 1000 official oos_test examples")
    groups = {name: [] for name in SPLITS}
    for split in sorted(expected):
        destination = "valid" if split.endswith("val") else "test" if split.endswith("test") else "train"
        for index, pair in enumerate(data[split]):
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[1], str):
                raise ValueError("Invalid CLINC source row")
            text, label = pair
            _normalized(text)
            if split.startswith("oos_"):
                if label != "oos":
                    raise ValueError("Invalid CLINC OOS label")
                label = "out_of_scope"
            elif label in ("oos", "out_of_scope"):
                raise ValueError("Unexpected OOS label in in-domain CLINC split")
            if label in CLINC_DESCRIPTIONS:
                groups[destination].append([_row(text, label, "clinc", f"{split}:{index}", CLINC_QUESTION, CLINC_DESCRIPTIONS)])
    groups["train"], groups["calibration"] = _stratify(groups["train"], seed, [0.8])
    return groups


def _deduplicate(groups, *, exclude_conflicts=False):
    # Union both identities before choosing a winner, including transitive matches.
    items = [(split, rows) for split in ("test", "calibration", "valid", "train") for rows in groups[split]]
    parents = list(range(len(items)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen = {}
    for index, (_, rows) in enumerate(items):
        for key in (("group", rows[0]["metadata"]["group_id"]), ("text", _normalized(rows[0]["prompt"]))):
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    components = defaultdict(list)
    for index in range(len(items)):
        components[root(index)].append(index)
    result = {name: [] for name in SPLITS}
    exclusions = []
    for indices in components.values():
        first = indices[0]
        split, rows = items[first]
        signatures = [{row["question"]: row["label"] for row in items[index][1]} for index in indices]
        if any(signature != signatures[0] for signature in signatures[1:]):
            if not exclude_conflicts:
                raise ValueError("Conflicting labels for duplicate text or group")
            members = []
            for index, signature in zip(indices, signatures):
                other_split, other = items[index]
                members.append({"split": other_split, "group_id": other[0]["metadata"]["group_id"],
                                "ids": [row["id"] for row in other],
                                "label_map_sha256": _digest(json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                                "normalized_text_sha256": _digest(_normalized(other[0]["prompt"]))})
            exclusions.append({"reason": "conflicting_label_maps", "action": "exclude_entire_component",
                               "members": members, "excluded_groups": len(members),
                               "excluded_rows": sum(len(member["ids"]) for member in members)})
            continue
        for index in indices:
            other_split, other = items[index]
            if index != first:
                exclusions.append({"split": other_split, "group_id": other[0]["metadata"]["group_id"],
                                   "ids": [row["id"] for row in other], "reason": "duplicate", "retained_id": rows[0]["id"]})
        result[split].append(rows)
    return result, exclusions


def prepare_text_dataset(name, output_dir, cache_dir, *, seed=17, max_tokens=2048, tokenizer=None):
    """Create four JSONL splits and return their manifest; never load model weights."""
    if name not in ("contractnli", "banking", "clinc"):
        raise ValueError("Unknown text dataset; choose contractnli, banking, or clinc")
    if type(seed) is not int or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("seed must be an integer and max_tokens a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Dataset output must be new: {output_dir}")
    sources = []
    def fetch(url, revision, license_name, attribution):
        content, receipt = _download(url, cache_dir, revision, license_name, attribution)
        sources.append(receipt)
        return content
    try:
        if name == "contractnli":
            groups = _contract(fetch(CONTRACT_URL, "official-2021-release-url; first-use SHA256 cache pin", "CC-BY-4.0", "Koreeda and Manning, ContractNLI (2021)"), seed)
            policy = "Official train/test; seeded document-level 50/50 official dev into valid/calibration; all 17 hypotheses, full document, no evidence spans."
        elif name == "banking":
            contents = {file: fetch(BANKING_BASE + file, BANKING_REVISION, "CC-BY-4.0", "PolyAI, BANKING77") for file in ("train.csv", "test.csv", "categories.json")}
            groups = _banking(contents, seed)
            policy = "BANKING77-10 fixed card-payment intents; original train stratified 70/15/15 train/valid/calibration; original test restricted to the same ten intents."
        else:
            groups = _clinc(fetch(CLINC_URL, CLINC_REVISION, "CC-BY-3.0", "Larson et al., CLINC out-of-scope evaluation (2019)"), seed)
            policy = "Fixed ten banking intents plus source oos; official train stratified 80/20 train/calibration; official val as valid; official test selected ten plus all 1000 oos_test, without OOS resampling."
    except (KeyError, TypeError, UnicodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid {name} source format: {error}") from error
    selected = {split: {"groups": len(items), "rows": sum(map(len, items))} for split, items in groups.items()}
    groups, exclusions = _deduplicate(groups, exclude_conflicts=name == "contractnli")
    supplied_tokenizer = tokenizer is not None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=False)
    from semif_phase1.direct import encode_prompt
    retained = {split: [] for split in SPLITS}
    coverage = {}
    for split, items in groups.items():
        token_exclusions = 0
        for rows in items:
            lengths = []
            for row in rows:
                # A large ceiling measures the complete SemIf input, not a truncated surrogate.
                ids, _, _ = encode_prompt(tokenizer, to_semif(row), 2**63 - 1)
                lengths.append(len(ids))
            if max(lengths) > max_tokens:
                token_exclusions += 1
                exclusions.append({"split": split, "group_id": rows[0]["metadata"]["group_id"],
                                   "ids": [row["id"] for row in rows], "reason": "max_tokens",
                                   "max_input_tokens": max(lengths)})
            else:
                retained[split].extend(rows)
        coverage[split] = {"selected_groups": selected[split]["groups"], "selected_rows": selected[split]["rows"],
                           "deduplicated_groups": len(items), "token_excluded_groups": token_exclusions,
                           "retained_groups": len(items) - token_exclusions, "retained_rows": len(retained[split]),
                           "group_retention_fraction": (len(items) - token_exclusions) / selected[split]["groups"] if selected[split]["groups"] else 0}
        if not retained[split]:
            raise ValueError(f"Empty {split} split after selection, deduplication, and token filtering")
        retained[split].sort(key=lambda row: row["id"])
    validate_splits(retained)
    manifest = {"format": "ifllm-learn.textdata.v1", "name": name,
                "benchmark": {"banking": "BANKING77-10", "clinc": "CLINC-banking-10+OOS", "contractnli": "ContractNLI-full-document-17"}[name],
                "sources": sources, "seed": seed, "max_tokens": max_tokens,
                "tokenizer": {"source": DEFAULT_MODEL if not supplied_tokenizer else "caller-supplied",
                              "revision": DEFAULT_REVISION if not supplied_tokenizer else None,
                              "class": type(tokenizer).__name__, "semif_revision": SEMIF_REVISION},
                "selection_policy": policy,
                "deduplication_policy": "NFKC, casefold, whitespace collapse; transitive text/group identity; test > calibration > valid > train for agreeing complete label maps; " + ("exclude entire conflicting component with per-member label-map digests." if name == "contractnli" else "reject conflicting complete label maps."),
                "partition_rounding": "Floor cumulative boundaries per class (per document for ContractNLI dev); remainder in last partition.",
                "counts": {split: len(rows) for split, rows in retained.items()},
                "classes": [option["id"] for option in retained["train"][0]["options"]],
                "class_counts": {split: dict(sorted(Counter(row["label"] for row in rows).items())) for split, rows in retained.items()},
                "group_membership": {split: sorted({row["metadata"]["group_id"] for row in rows}) for split, rows in retained.items()},
                "filter_coverage": coverage, "exclusions": exclusions,
                "limitations": ["Derived English benchmark subsets, not evidence of production performance or calibrated confidence.",
                                "Deduplication and whole-group token exclusion can change class prevalence; counts and exclusions describe the retained population.",
                                "No gold evidence spans, labels, or metadata are included in model input; documents are never truncated.",
                                "ContractNLI's official release URL is not an immutable revision; its first download is trust-on-first-use SHA256 pinned in this cache.",
                                "Normalized exact matching does not detect paraphrases or pretraining contamination."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in retained.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest["sha256"] = {f"{split}.jsonl": file_sha256(output_dir / f"{split}.jsonl") for split in SPLITS}
    write_json(output_dir / "manifest.json", manifest)
    return manifest
