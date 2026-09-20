"""Real tiny Qwen3.5 gradients and checkpoint roundtrip; no large model download."""

import platform

import numpy as np
import pytest

from ifllm_learn.schema import DEFAULT_MODEL, DEFAULT_REVISION
from ifllm_learn.training import train_adapter, validate_training

ROW = {"id": "train-a", "prompt": "A deployment succeeded.", "question": "Success?",
       "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}], "label": "yes"}


@pytest.mark.parametrize("parameter,value", [("steps", 0), ("learning_rate", -1), ("num_layers", 0), ("rank", 0), ("max_tokens", 0), ("bits", 3)])
def test_hyperparameters_rejected_without_model_load(parameter, value):
    kwargs = dict(steps=2, learning_rate=1e-3, num_layers=1, rank=2, max_tokens=512, bits=None)
    kwargs[parameter] = value
    with pytest.raises(ValueError):
        validate_training([ROW], [{**ROW, "id": "valid"}], **kwargs)


def test_overlap_rejected_before_mlx_import():
    with pytest.raises(ValueError, match="overlap"):
        validate_training([ROW], [ROW], 1, 1e-3, 1, 2, 512, None)


def test_standalone_train_rejects_label_horizon_crossing_validation():
    train = {**ROW, "observed_at": "2024-12-31T00:00:00+00:00", "label_available_at": "2025-01-01T00:00:00+00:00"}
    valid = {**ROW, "id": "valid", "observed_at": "2025-01-01T00:00:00+00:00", "label_available_at": "2025-01-02T00:00:00+00:00"}
    with pytest.raises(ValueError, match="boundary"):
        validate_training([train], [valid], 1, 1e-3, 1, 2, 512, None)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, turns, **kwargs):
        return "\n".join(turn["content"] for turn in turns) + "\nAssistant:"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def tiny_bundle():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    mx.random.seed(17)
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 256, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "full_attention_interval": 2,
    }))
    model.eval()
    mx.eval(model.parameters())
    metadata = {"source": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
                "source_artifact_sha256": {"fixture": "tiny native hybrid test model"},
                "dtype": ["mlx.core.float32"], "quantization": None,
                "adapter_sha256": None, "task_contract_sha256": None}
    return model, Tokenizer(), metadata


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Native MLX requires Apple Silicon")
def test_real_hybrid_gradients_adapter_roundtrip_and_tamper(tmp_path, monkeypatch):
    from semif_phase1 import mlx_backend
    from ifllm_learn.inference import load_backend, predict_rows

    bundle = tiny_bundle()
    train = [ROW, {**ROW, "id": "train-b", "prompt": "A second deployment succeeded."}]
    valid = [{**ROW, "id": "valid", "prompt": "Another deployment succeeded."}]
    heldout = [{**ROW, "id": "test", "prompt": "The service deployment succeeded."}]
    before = predict_rows(bundle, heldout, 1024)
    output = tmp_path / "adapter"
    manifest = train_adapter(train, valid, output, steps=3, learning_rate=1e-3,
                             num_layers=2, rank=2, max_tokens=1024, bundle=bundle)
    assert manifest["steps"] == 3
    assert manifest["changed_adapter_tensors"]
    assert all(np.isfinite(item["loss"]) for item in manifest["history"])
    assert manifest["validation_loss_after"] < manifest["validation_loss_before"]
    trained = predict_rows(bundle, heldout, 1024)
    assert not np.allclose(before[0]["logits"], trained[0]["logits"])
    monkeypatch.setattr(mlx_backend, "load_model", lambda *args, **kwargs: tiny_bundle())
    reloaded = predict_rows(load_backend(output), heldout, 1024)
    np.testing.assert_allclose(trained[0]["logits"], reloaded[0]["logits"], atol=1e-5)
    assert reloaded[0]["adapter_sha256"] == manifest["adapter_sha256"]
    with pytest.raises(ValueError, match="task"):
        predict_rows(bundle, [{**heldout[0], "question": "A different question?"}], 1024)
    with pytest.raises(ValueError, match="precision"):
        load_backend(output, bits=4)
    with (output / "adapters.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_backend(output)


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64", reason="Native MLX requires Apple Silicon")
def test_seed_is_independent_of_model_loading_path(tmp_path, monkeypatch):
    from ifllm_learn import inference
    monkeypatch.setattr(inference, "load_backend", lambda **kwargs: tiny_bundle())
    train = [ROW]
    valid = [{**ROW, "id": "valid"}]
    kwargs = dict(steps=1, learning_rate=1e-4, num_layers=2, rank=2, max_tokens=1024, seed=31)
    resident = train_adapter(train, valid, tmp_path / "resident", bundle=tiny_bundle(), **kwargs)
    standalone = train_adapter(train, valid, tmp_path / "standalone", **kwargs)
    assert resident["adapter_sha256"] == standalone["adapter_sha256"]
    assert resident["history"] == standalone["history"]
