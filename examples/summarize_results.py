"""Render measured demo tables from retained reports; no model loading."""

import json
from pathlib import Path


ORDER = (("banking", "BANKING77-10"), ("clinc", "CLINC banking-10 + OOS"),
         ("contractnli", "ContractNLI short-doc subset"), ("bitcoin", "Bitcoin exploratory toy"))


def main():
    print("| Demo | Test rows | TF-IDF accuracy | Frozen LLM | LoRA LLM | Frozen calibrated NLL | LoRA calibrated NLL |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, label in ORDER:
        report = json.loads((Path("results") / name / "report.json").read_text())
        classical = report.get("tfidf_baseline")
        tfidf = f"{classical['accuracy'] * 100:.2f}%" if classical else "Not measured"
        frozen_cal = report.get("base_calibrated")
        nll = f"{frozen_cal['log_loss']:.4f}" if frozen_cal else "Not measured"
        print(f"| [{label}](results/{name}/report.json) | {report['base']['count']} | {tfidf} | "
              f"{report['base']['accuracy'] * 100:.2f}% | {report['trained']['accuracy'] * 100:.2f}% | "
              f"{nll} | {report['trained_calibrated']['log_loss']:.4f} |")
    clinc = json.loads(Path("results/clinc/report.json").read_text())
    print("\nCLINC conditional metrics:")
    for model in ("base", "trained"):
        values = clinc["task_metrics"][model]
        print(model, json.dumps({key: values[key] for key in (
            "macro_f1", "out_of_scope_recall", "in_scope_false_rejection_rate", "supported_intent_accuracy")}))


if __name__ == "__main__":
    main()
