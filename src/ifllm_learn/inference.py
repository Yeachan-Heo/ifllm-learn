"""Native MLX option scoring with exact model and adapter provenance."""

import json
from pathlib import Path

from .schema import DEFAULT_MODEL, DEFAULT_REVISION, contract_hash, file_sha256, to_semif


def load_backend(adapter=None, bits=None):
    from semif_phase1 import mlx_backend

    manifest = None
    if adapter is not None:
        adapter = Path(adapter)
        manifest = json.loads((adapter / "training.json").read_text())
        if manifest.get("format") != "ifllm-learn.lora.v1" or manifest["model_revision"] != DEFAULT_REVISION or manifest["model_source"] != DEFAULT_MODEL:
            raise ValueError("Adapter model identity is incompatible")
        if bits is not None and bits != manifest["bits"]:
            raise ValueError("Adapter precision does not match requested precision")
        bits = manifest["bits"]
        if file_sha256(adapter / "adapters.safetensors") != manifest["adapter_sha256"]:
            raise ValueError("Adapter weights checksum mismatch")
        if file_sha256(adapter / "adapter_config.json") != manifest["adapter_config_sha256"]:
            raise ValueError("Adapter configuration checksum mismatch")
    model, tokenizer, metadata = mlx_backend.load_model(DEFAULT_MODEL, DEFAULT_REVISION, bits=bits, cache_limit_mib=256)
    metadata["adapter_sha256"] = None
    metadata["task_contract_sha256"] = None
    if manifest is not None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.utils import linear_to_lora_layers

        if metadata["source_artifact_sha256"] != manifest["base_artifact_sha256"]:
            raise ValueError("Adapter was trained on different source artifacts")
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config["fine_tune_type"] != "lora" or not 1 <= config["num_layers"] <= len(model.layers):
            raise ValueError("Invalid adapter layer configuration")
        model.freeze()
        linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
        expected = dict(tree_flatten(model.trainable_parameters()))
        weights = mx.load(str(adapter / "adapters.safetensors"))
        if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in expected):
            raise ValueError("Adapter tensors do not exactly match model trainable tensors")
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        mx.synchronize()
        metadata["adapter_sha256"] = manifest["adapter_sha256"]
        metadata["task_contract_sha256"] = manifest["contract_sha256"]
    model.eval()
    return model, tokenizer, metadata


def predict_rows(bundle, rows, max_tokens=2048):
    from semif_phase1 import mlx_backend

    model, tokenizer, metadata = bundle
    contract = contract_hash(rows)
    if metadata.get("task_contract_sha256") not in (None, contract):
        raise ValueError("Prediction task differs from the trained adapter task contract")
    model.eval()
    predictions = []
    for row in rows:
        result = mlx_backend.score(model, tokenizer, to_semif(row), metadata, max_tokens)
        probabilities = result["probabilities"]
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        prediction = {
            "id": row["id"], "answer": result["option_ids"][winner],
            "option_ids": result["option_ids"], "logits": result["option_logits"],
            "probabilities": probabilities, "input_tokens": result["input_tokens"],
            "model_revision": metadata["revision"], "adapter_sha256": metadata.get("adapter_sha256"),
            "precision": {"dtype": metadata["dtype"], "quantization": metadata["quantization"]},
            "prompt_sha256": result["prompt_sha256"], "contract_sha256": contract,
            "total_seconds": result["total_seconds"],
        }
        for key in ("label", "observed_at", "label_available_at"):
            if key in row:
                prediction[key] = row[key]
        provenance = row.get("metadata", {})
        prediction["metadata"] = {key: provenance[key] for key in ("group_id", "question_id") if key in provenance}
        predictions.append(prediction)
    return predictions
