"""ifllm-learn command line: prepare, train, predict, calibrate, evaluate, run."""

import argparse
import json
from pathlib import Path

from .schema import load_jsonl, write_json, write_jsonl


def training_arguments(parser):
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Gradient accumulation examples per optimizer update")
    parser.add_argument("--bits", type=int, choices=(4, 8), help="Optional in-memory base quantization; omitted preserves original precision")


def training_kwargs(args):
    return {name: getattr(args, name) for name in ("steps", "learning_rate", "num_layers", "rank", "max_tokens", "seed", "bits", "batch_size")}


def read_predictions(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected nonempty prediction JSONL")
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ifllm-learn",
        description="Supervised semantic decisions on MLX: prompt + question + options -> answer. label is ground truth, never input.",
        epilog="Outputs are create-only. The Bitcoin example is offline classification, not an investment strategy.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    text = commands.add_parser("prepare-text", help="Prepare an explicit ContractNLI / CLINC / BANKING demo with source provenance")
    text.add_argument("dataset", choices=("contractnli", "clinc", "banking"))
    text.add_argument("--output", type=Path, required=True)
    text.add_argument("--cache-dir", type=Path, default=Path("data/cache/text"))
    text.add_argument("--seed", type=int, default=17)
    text.add_argument("--max-tokens", type=int, default=2048)
    verify = commands.add_parser("verify-run", help="Reload a saved adapter and compare actual test logits and calibrated probabilities")
    verify.add_argument("--run-dir", type=Path, required=True)
    export = commands.add_parser("export-run", help="Export a compact evidence bundle without source documents or model weights")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-btc", help="Download/checksum Binance data and build four chronological JSONL splits")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, default=Path("data/cache/binance"))
    prepare.add_argument("--start-month", default="2023-01")
    prepare.add_argument("--end-month", default="2025-12")
    train = commands.add_parser("train", help="Actually optimize LoRA using categorical choice cross entropy")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--valid", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    training_arguments(train)
    predict = commands.add_parser("predict", help="Read native option logits, optionally using an adapter and calibration artifact")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--adapter", type=Path)
    predict.add_argument("--calibration", type=Path)
    predict.add_argument("--bits", type=int, choices=(4, 8))
    predict.add_argument("--max-tokens", type=int, default=2048)
    calibrate = commands.add_parser("calibrate", help="Fit a separate positive scalar temperature on labeled calibration predictions")
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate labeled native predictions; no model loading")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path)
    run = commands.add_parser("run", help="Base evaluation -> actual LoRA -> calibration -> held-out evaluation")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    for name in ("train", "valid", "calibration", "test"):
        run.add_argument(f"--{name}-limit", type=int, help="Optional deterministic time-spread toy subset, chosen before scoring")
    training_arguments(run)
    return parser


def dispatch(args):
    if args.command == "prepare-text":
        from .textdata import prepare_text_dataset
        result = prepare_text_dataset(args.dataset, args.output, args.cache_dir, seed=args.seed, max_tokens=args.max_tokens)
        return {"output": str(args.output), "manifest": str(args.output / "manifest.json"),
                "benchmark": result["benchmark"], "counts": result["counts"], "filter_coverage": result["filter_coverage"]}
    if args.command == "verify-run":
        from .evidence import verify_run
        return verify_run(args.run_dir)
    if args.command == "export-run":
        from .evidence import export_run
        return export_run(args.run_dir, args.data_dir, args.output)
    if args.command == "prepare-btc":
        from .bitcoin import prepare_bitcoin
        result = prepare_bitcoin(args.output, args.cache_dir, args.start_month, args.end_month)
        return {"output": str(args.output), "counts": result["counts"], "candle_count": result["candle_count"]}
    if args.command == "train":
        from .training import train_adapter
        result = train_adapter(load_jsonl(args.train), load_jsonl(args.valid), args.output, **training_kwargs(args))
        return {"output": str(args.output), "adapter_sha256": result["adapter_sha256"], "steps": result["steps"],
                "validation_loss_before": result["validation_loss_before"], "validation_loss_after": result["validation_loss_after"]}
    if args.command == "predict":
        from .inference import load_backend, predict_rows
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        rows = load_jsonl(args.input, require_label=False)
        predictions = predict_rows(load_backend(args.adapter, args.bits), rows, args.max_tokens)
        if args.calibration:
            from .calibration import apply_prediction_calibration
            predictions = apply_prediction_calibration(predictions, json.loads(args.calibration.read_text()))
        write_jsonl(args.output, predictions)
        return {"output": str(args.output), "count": len(predictions)}
    if args.command == "calibrate":
        from .calibration import fit_prediction_calibration
        result = fit_prediction_calibration(read_predictions(args.input), args.output)
        return {"output": str(args.output), "temperature": result["temperature"], "n": result["n"],
                "nll_before": result["nll_before"], "nll_after": result["nll_after"]}
    if args.command == "evaluate":
        from .metrics import metrics, prediction_arrays
        from .calibration import apply_prediction_calibration, prediction_identity
        predictions = read_predictions(args.input)
        prediction_identity(predictions)
        if args.calibration:
            predictions = apply_prediction_calibration(predictions, json.loads(args.calibration.read_text()))
        temperatures = {row.get("temperature", 1.0) for row in predictions}
        if len(temperatures) != 1:
            raise ValueError("Mixed prediction temperatures")
        logits, labels = prediction_arrays(predictions)
        result = metrics(logits, labels, temperature=temperatures.pop())
        write_json(args.output, result)
        return result
    if args.command == "run":
        from .workflow import run_workflow
        limits = {f"{name}_limit": getattr(args, f"{name}_limit") for name in ("train", "valid", "calibration", "test")}
        result = run_workflow(args.data_dir, args.output, **training_kwargs(args), **limits)
        return {"report": str(args.output / "report.json"), "adapter_sha256": result["adapter_sha256"],
                "counts": result["run"]["selected_counts"], "temperature": result["temperature"],
                "validation_loss_before": result["validation_loss_before"],
                "validation_loss_after": result["validation_loss_after"],
                "validation_improved": result["validation_improved"],
                "log_loss": {key: result[key]["log_loss"] for key in ("train_frequency_baseline", "base", "trained", "trained_calibrated")},
                "accuracy": {key: result[key]["accuracy"] for key in ("train_frequency_baseline", "base", "trained", "trained_calibrated")}}
    raise ValueError(f"Unknown command: {args.command}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
