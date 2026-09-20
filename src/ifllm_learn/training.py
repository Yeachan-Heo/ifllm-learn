"""Supervised LoRA: optimize ground-truth choices, not generated text or temperature."""

import math
from pathlib import Path
import hashlib
import time
import numpy as np

from .schema import (
    DEFAULT_MODEL, DEFAULT_REVISION, SEMIF_REVISION, assert_disjoint, assert_temporal_separation, canonical_json,
    contract_hash, file_sha256, label_index, to_semif, validate_example, write_json,
)


def validate_training(train_rows, valid_rows, steps, learning_rate, num_layers, rank, max_tokens, bits):
    if not train_rows or not valid_rows:
        raise ValueError("Nonempty separate train and validation sets are required")
    for name, value in (("steps", steps), ("num_layers", num_layers), ("rank", rank), ("max_tokens", max_tokens)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if bits not in (None, 4, 8):
        raise ValueError("bits must be 4, 8, or None")
    for row in [*train_rows, *valid_rows]:
        validate_example(row)
    assert_disjoint(train_rows, valid_rows)
    assert_temporal_separation(train_rows, valid_rows)
    if contract_hash(train_rows) != contract_hash(valid_rows):
        raise ValueError("Training and validation task contracts differ")


def choice_loss(model, tokens, slots, target):
    import mlx.core as mx
    import mlx.nn as nn

    vocabulary = model(tokens)[0, -1].astype(mx.float32)
    choices = vocabulary[slots][None, :]
    return nn.losses.cross_entropy(choices, target, reduction="mean")


def encode_rows(tokenizer, rows, max_tokens):
    import mlx.core as mx
    from semif_phase1.direct import encode_prompt

    encoded = []
    for row in rows:
        ids, slots, _ = encode_prompt(tokenizer, to_semif(row), max_tokens)
        encoded.append((mx.array([ids]), mx.array(slots), mx.array([label_index(row)])))
    return encoded


def validation_loss(model, encoded):
    import mlx.core as mx

    model.eval()
    values = [float(choice_loss(model, *item).item()) for item in encoded]
    mx.synchronize()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite validation loss")
    return float(np.mean(values))


def train_adapter(train_rows, valid_rows, output_dir, *, steps=100, learning_rate=1e-5,
                  num_layers=2, rank=8, max_tokens=2048, seed=0, bits=None, bundle=None, batch_size=1):
    validate_training(train_rows, valid_rows, steps, learning_rate, num_layers, rank, max_tokens, bits)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Adapter output must be new: {output_dir}")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from .inference import load_backend

    rng = np.random.default_rng(seed)
    model, tokenizer, metadata = bundle if bundle is not None else load_backend(bits=bits)
    if metadata.get("adapter_sha256") is not None:
        raise ValueError("Train from the pinned base model, not an already adapted bundle")
    source_bits = (metadata.get("quantization") or {}).get("bits")
    if source_bits != bits:
        raise ValueError("Bundle precision differs from requested training precision")
    if metadata["source"] != DEFAULT_MODEL or metadata["revision"] != DEFAULT_REVISION:
        raise ValueError("Training requires the pinned base model identity")
    if num_layers > len(model.layers):
        raise ValueError("Requested more trainable layers than the model contains")
    encoded_train = encode_rows(tokenizer, train_rows, max_tokens)
    encoded_valid = encode_rows(tokenizer, valid_rows, max_tokens)
    # Reject bad data/token limits before touching adapters or creating output files.
    model.freeze()
    # Seed adapter initialization after loading; standalone and resident-model runs agree.
    mx.random.seed(seed)
    lora_parameters = {"rank": rank, "scale": 20.0, "dropout": 0.0}
    linear_to_lora_layers(model, num_layers, lora_parameters)
    initial = {name: np.array(value) for name, value in tree_flatten(model.trainable_parameters())}
    if not initial:
        raise RuntimeError("LoRA conversion produced no trainable parameters")
    config = {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora_parameters}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "adapter_config.json", config)
    before = validation_loss(model, encoded_valid)
    optimizer = optim.Adam(learning_rate=learning_rate)
    value_and_grad = nn.value_and_grad(model, choice_loss)
    history, order = [], []
    started = time.perf_counter()
    model.train()
    cursor = 0
    for step in range(steps):
        accumulated = None
        losses, example_ids = [], []
        for _ in range(batch_size):
            if cursor % len(encoded_train) == 0:
                order = rng.permutation(len(encoded_train)).tolist()
            index = order[cursor % len(encoded_train)]
            cursor += 1
            loss, gradients = value_and_grad(model, *encoded_train[index])
            flat = dict(tree_flatten(gradients))
            finite = mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in flat.values()]))
            mx.eval(loss, finite)
            if not bool(finite.item()) or not math.isfinite(float(loss.item())):
                raise RuntimeError(f"Non-finite loss/gradient at optimizer step {step + 1}")
            if accumulated is None:
                accumulated = {key: value / batch_size for key, value in flat.items()}
            else:
                accumulated = {key: accumulated[key] + value / batch_size for key, value in flat.items()}
            mx.eval(list(accumulated.values()))
            losses.append(float(loss.item()))
            example_ids.append(train_rows[index]["id"])
        from mlx.utils import tree_unflatten
        gradients, norm = optim.clip_grad_norm(tree_unflatten(list(accumulated.items())), max_norm=1.0)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, norm)
        mx.synchronize()
        history.append({"step": step + 1, "example_ids": example_ids,
                        "loss": float(np.mean(losses)), "gradient_norm": float(norm.item())})
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(f"LoRA step {step + 1}/{steps}: choice loss={history[-1]['loss']:.6f}", flush=True)
    after = validation_loss(model, encoded_valid)
    if after >= before:
        print(f"Warning: validation loss did not improve ({before:.6f} -> {after:.6f}); this adapter is experimental.", flush=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    changed = [name for name, value in weights.items() if not np.array_equal(initial[name], np.array(value))]
    if not changed:
        raise RuntimeError("No adapter tensors changed; refusing to report a trained model")
    if not all(np.isfinite(np.array(value)).all() for value in weights.values()):
        raise RuntimeError("Trained adapter contains non-finite weights")
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), weights)
    adapter_sha = file_sha256(output_dir / "adapters.safetensors")
    contract = contract_hash(train_rows)
    manifest = {
        "format": "ifllm-learn.lora.v1", "objective": "cross entropy over declared native option-token logits",
        "model_source": DEFAULT_MODEL, "model_revision": DEFAULT_REVISION, "semif_revision": SEMIF_REVISION,
        "base_artifact_sha256": metadata["source_artifact_sha256"],
        "runtime": {key: metadata.get(key) for key in ("mlx_version", "mlx_lm_version", "mlx_lm_source", "transformers_version")},
        "bits": bits, "seed": seed, "steps": steps, "learning_rate": learning_rate,
        "num_layers": num_layers, "rank": rank, "max_tokens": max_tokens,
        "batch_size": batch_size, "examples_seen": steps * batch_size,
        "trainable_parameters": sum(value.size for value in initial.values()),
        "changed_adapter_tensors": changed, "history": history,
        "validation_loss_before": before, "validation_loss_after": after,
        "validation_improved": after < before,
        "training_seconds": time.perf_counter() - started,
        "train_ids": [row["id"] for row in train_rows], "valid_ids": [row["id"] for row in valid_rows],
        "train_sha256": hashlib.sha256(canonical_json(train_rows).encode()).hexdigest(),
        "valid_sha256": hashlib.sha256(canonical_json(valid_rows).encode()).hexdigest(),
        "contract_sha256": contract, "adapter_sha256": adapter_sha,
        "adapter_config_sha256": file_sha256(output_dir / "adapter_config.json"),
    }
    write_json(output_dir / "training.json", manifest)
    metadata["adapter_sha256"] = adapter_sha
    metadata["task_contract_sha256"] = contract
    model.eval()
    return manifest
