"""Build offline demo assets from retained results; --check rejects stale assets."""

import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ("banking", "Banking", "Ten payment intents. One learned decision.", "400 held-out messages · selected 10 of 77 intents"),
    ("clinc", "Out of scope", "A better router is not always a better gatekeeper.", "1,300 requests · 10 intents + official OOS population"),
    ("contractnli", "Contracts", "Seventeen questions. Ten complete contracts.", "170 decisions · 10 short documents · no evidence extraction"),
    ("bitcoin", "Bitcoin", "The hook that did not beat the baseline.", "32 exploratory observations · reused test set · no trading"),
]
MODELS = [
    ("base", "Frozen", "base-test.jsonl"),
    ("base_calibrated", "Frozen + calibration", "base-calibrated-test.jsonl"),
    ("trained", "LoRA", "trained-test.jsonl"),
    ("trained_calibrated", "LoRA + calibration", "calibrated-test.jsonl"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_data(root=ROOT):
    datasets, hashes = [], {}
    for name, label, title, subtitle in DATASETS:
        directory = root / "results" / name
        report = json.loads((directory / "report.json").read_text())
        training = json.loads((directory / "training.json").read_text())
        provenance = json.loads((directory / "dataset-manifest.json").read_text())
        hashes[f"results/{name}/report.json"] = sha256(directory / "report.json")
        hashes[f"results/{name}/training.json"] = sha256(directory / "training.json")
        hashes[f"results/{name}/dataset-manifest.json"] = sha256(directory / "dataset-manifest.json")
        models, predictions = [], {}
        for key, model_label, filename in MODELS:
            if key not in report:
                continue
            metric = report[key]
            rows = read_rows(directory / filename)
            hashes[f"results/{name}/{filename}"] = sha256(directory / filename)
            predictions[key] = {row["id"]: row for row in rows}
            models.append({
                "id": key, "label": model_label,
                "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
                "nll": metric["log_loss"], "brier": metric["brier"], "ece": metric["ece_10_bins"],
                "reliability": metric["reliability_bins"], "confusion": metric["confusion_matrix"],
                "task_metrics": report.get("task_metrics", {}).get(key, {}),
            })
        base_rows = read_rows(directory / "base-test.jsonl")
        selected = []
        for row in base_rows:
            answers = {}
            for key in predictions:
                candidate = predictions[key][row["id"]]
                if candidate["label"] != row["label"] or candidate["option_ids"] != row["option_ids"]:
                    raise ValueError("Recorded prediction identities do not align")
                answers[key] = {"answer": candidate["answer"], "probabilities": candidate["probabilities"]}
            selected.append({"id": row["id"], "label": row["label"], "models": answers})
        datasets.append({
            "id": name, "label": label, "title": title, "subtitle": subtitle,
            "test_count": report["base"]["count"], "choices": report["option_ids"],
            "counts": report["run"]["selected_counts"], "models": models,
            "prior_accuracy": report["train_frequency_baseline"]["accuracy"],
            "tfidf_accuracy": report.get("tfidf_baseline", {}).get("accuracy"),
            "temperature": report["temperature"], "validation_improved": report["validation_improved"],
            "validation_before": report["validation_loss_before"], "validation_after": report["validation_loss_after"],
            "training_steps": training["steps"], "history": [{"step": r["step"], "loss": r["loss"]} for r in training["history"]],
            "rows": selected, "report_url": f"../results/{name}/report.json",
            "coverage": provenance.get("filter_coverage", {}),
            "notes": report["limitations"],
        })
    return {"format": "ifllm-learn.visual-demo.v1", "mode": "Recorded experiments, not live inference",
            "source_sha256": hashes, "datasets": datasets}


def build_svg(data):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="690" viewBox="0 0 1200 690" role="img" aria-labelledby="title desc">',
             '<title id="title">ifllm-learn: Train the if. Keep the receipts.</title>',
             '<desc id="desc">Measured frozen versus LoRA accuracy on four explicitly scoped demonstrations. Banking improves; CLINC declines; ContractNLI gains one correct decision; Bitcoin is unchanged.</desc>',
             '<rect width="1200" height="690" rx="24" fill="#111713"/>',
             '<g font-family="Arial, Helvetica, sans-serif">',
             '<text x="56" y="60" fill="#c4f47b" font-size="18" letter-spacing="3">IFLLM-LEARN / THE DECISION LAB</text>',
             '<text x="56" y="137" fill="#f2f6ee" font-size="58" font-weight="700">Train the if.</text>',
             '<text x="56" y="199" fill="#f2f6ee" font-size="58" font-weight="700">Keep the receipts.</text>',
             '<text x="58" y="244" fill="#a7b5a8" font-size="20">Real labels. Local LoRA. Measured results—not four cherry-picked wins.</text>',
             '<rect x="820" y="110" width="14" height="14" rx="3" fill="#7e9892"/><text x="843" y="123" fill="#c4cdc2" font-size="17">Frozen</text>',
             '<rect x="940" y="110" width="14" height="14" rx="3" fill="#c4f47b"/><text x="963" y="123" fill="#c4cdc2" font-size="17">LoRA</text>']
    for i, dataset in enumerate(data["datasets"]):
        x = 56 + i * 282
        models = {m["id"]: m for m in dataset["models"]}
        before, after = models["base"]["accuracy"], models["trained"]["accuracy"]
        delta = (after - before) * 100
        color = '#c4f47b' if delta > 0 else '#eeb882' if delta < 0 else '#a7b5a8'
        scope = {'banking': '10 intents / 400 messages', 'clinc': '10 intents + OOS / 1,300', 'contractnli': '10 documents / 170 decisions', 'bitcoin': '32 exploratory observations'}[dataset['id']]
        parts += [f'<text x="{x}" y="318" fill="#f2f6ee" font-size="23" font-weight="700">{html.escape(dataset["label"])}</text>',
                  f'<text x="{x}" y="346" fill="#a7b5a8" font-size="14">{html.escape(scope)}</text>']
        for j, (value, fill) in enumerate(((before, '#7e9892'), (after, '#c4f47b'))):
            y = 375 + j * 43
            parts += [f'<rect x="{x}" y="{y}" width="226" height="21" rx="5" fill="#26332a"/>',
                      f'<rect x="{x}" y="{y}" width="{226 * value:.3f}" height="21" rx="5" fill="{fill}"/>',
                      f'<text x="{x}" y="{y + 38}" fill="#dfe7db" font-size="15">{value * 100:.2f}% accuracy</text>']
        parts.append(f'<text x="{x}" y="516" fill="{color}" font-size="29" font-weight="700">{delta:+.2f} pp</text>')
    parts += ['<path d="M56 550H1144" stroke="#334036"/>',
              '<text x="56" y="591" fill="#c4cdc2" font-size="17">All numbers are recorded runs. ContractNLI: one more correct decision, not a robust gain.</text>',
              '<text x="56" y="621" fill="#a7b5a8" font-size="17">Bitcoin did not beat its accuracy baseline. Open the interactive lab to inspect errors and calibration.</text>',
              '<text x="56" y="656" fill="#c4f47b" font-size="14" letter-spacing="1">NO LIVE BROWSER INFERENCE • NO TRADING • NO HIDDEN REGRESSIONS</text>', '</g></svg>']
    return '\n'.join(parts) + '\n'


def rendered_assets(root=ROOT):
    data = build_data(root)
    return {root / 'demo' / 'data.json': json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n',
            root / 'docs' / 'demo-overview.svg': build_svg(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, content in rendered_assets().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f'Stale visualization asset: {path.relative_to(ROOT)}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        print(('Verified ' if args.check else 'Built ') + str(path.relative_to(ROOT)))


if __name__ == '__main__':
    main()
