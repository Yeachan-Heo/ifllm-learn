# ifllm-learn

## Train the `if`. Not the essay.

**Turn labeled examples into local, typed LLM decisions.**

```text
prompt + question + choices  →  answer + scores
                labels      →  LoRA training
       held-out outcomes    →  probability calibration
```

We started with the obvious bad idea: **can an LLM predict Bitcoin?**
It did not beat a class-frequency baseline on accuracy. We kept the receipts.

Then we gave it problems with evidence in the input: payment-support routing,
out-of-scope requests, and contract checklists. Same model. Same learning loop.
Different datasets. No task-specific classification head to design.

**ifllm-learn is a small research toolkit, not a claim that fine-tuning always helps.**
It makes the comparison runnable—and keeps the failures visible.

[Quick start](#quick-start) · [Measured demos](#measured-demos) · [Your own data](#your-own-data) · [How it works](#how-it-works) · [Limitations](#limitations)

## What you get

- **Actual supervised learning.** MLX LoRA updates on the correct choice's cross-entropy—not temperature-only tuning.
- **No answer-generation loop.** Read the native logits for declared choice tokens; do not generate prose or repair model-written JSON. Python still serializes the result.
- **A fair comparison.** Train-frequency prior, TF-IDF + logistic regression, frozen LLM, trained LLM, and separately calibrated variants.
- **Separate train / validation / calibration / test data.** Document-group checks for text; horizon purging for time series.
- **Reproducibility artifacts.** Source revisions, dataset hashes, selected IDs, adapter checksums, raw predictions, and fresh-process reload checks.
- **Runs locally on Apple Silicon.** Qwen3.5-4B + native MLX. No inference API key or hosted training service.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf) and [MLX-LM](https://github.com/ml-explore/mlx-lm).
Independent of Jev / TypeSafe; not an implementation of their undisclosed training system.

## Quick start

Requires **Apple Silicon macOS, Python 3.12+, Git, and working Metal** for training
and inference. The measured runs use an M5 Max with 128 GiB unified memory;
minimum-memory devices have not been qualified. The pinned source checkpoint
is approximately 9.3 GB on disk, plus runtime memory and dependencies.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mlx,test]'

# Download and prepare the explicitly selected ten-intent banking demo.
ifllm-learn prepare-text banking --output data/banking-demo

# Frozen base → actual LoRA → independent calibration → test comparison.
ifllm-learn run \
  --data-dir data/banking-demo \
  --output runs/my-banking-demo \
  --steps 128 --batch-size 4 --learning-rate 1e-5 \
  --num-layers 2 --rank 8 --max-tokens 2048 --seed 17

# Verify that the saved adapter reproduces the measured outputs.
ifllm-learn verify-run --run-dir runs/my-banking-demo
```

First use downloads public datasets and the pinned model. Subsequent model runs
can use the cache with `HF_HUB_OFFLINE=1`. Downloads are not an inference API.
Every output directory is **create-only**: choose a new name for another run;
partial and unsuccessful experiments are not silently overwritten.

### Ask your trained model

The included query is an original example, not a benchmark row:

> “I bought one train ticket, but my card statement shows two identical charges for it.”

```bash
ifllm-learn predict \
  --input examples/banking-query.jsonl \
  --adapter runs/my-banking-demo/adapter \
  --calibration runs/my-banking-demo/temperature.json \
  --output runs/my-banking-demo/my-query.jsonl
```

The output contains `answer`, ordered `option_ids`, `logits`, `probabilities`,
input-token counts, and model/adapter/prompt identities. No rationale is generated.
The measured answer for this original example was **`transaction_charged_twice`**
([full output](results/banking-example.json)). Its calibrated option score was
0.9978; that is not a guarantee of correctness or a benchmark accuracy estimate.

## Measured demos

These are **single-seed, fixed-budget demonstrations**, not leaderboard submissions.
The execution budgets and class selections are frozen in
[`examples/demo-suite.json`](examples/demo-suite.json). The same Qwen3.5-4B
checkpoint and unquantized source precision are used for the text demos.

| Demo | Test rows | TF-IDF accuracy | Frozen LLM | LoRA LLM | Frozen calibrated NLL | LoRA calibrated NLL |
|---|---:|---:|---:|---:|---:|---:|
| [BANKING77-10](results/banking/report.json) | 400 | 87.75% | 87.75% | **92.50%** | 0.4904 | **0.2820** |
| [CLINC banking-10 + OOS](results/clinc/report.json) | 1,300 | 77.77% | **94.54%** | 93.69% | **0.1969** | 0.2646 |
| [ContractNLI short-doc subset](results/contractnli/report.json) | 170 | 40.59% | 71.76% | 72.35% | 0.6780 | 0.6467 |
| [Bitcoin exploratory toy](results/bitcoin/report.json) | 32 | Not measured | 56.25% | 56.25% | Not measured | 0.9957 |

**The headline win:** +4.75 percentage points on the fixed ten-intent banking task,
with 128 optimizer updates and 512 example presentations. That is not a full epoch
over its 1,029-row training partition, nor a full BANKING77 result.

**The more interesting result:** CLINC traded fewer false rejections for weaker OOS
rejection. Overall accuracy went down, while supported-intent accuracy went up.
We report both instead of picking whichever number looks better:

| CLINC measure | Frozen LLM | LoRA LLM |
|---|---:|---:|
| Supported-intent accuracy: 300 requests | 77.33% | **95.00%** |
| In-scope false-rejection rate ↓ | 21.67% | **2.00%** |
| OOS recall: 1,000 requests | **99.70%** | 93.30% |
| Macro F1 across all 11 classes | 0.8378 | **0.8931** |

For class-balanced interpretation, macro F1 for TF-IDF / frozen / LoRA is
**0.8728 / 0.8789 / 0.9257** on BANKING and **0.1925 / 0.5301 / 0.5389** on
ContractNLI. ContractNLI's 170 decisions come from **10 documents**, and the change
in accuracy is just **one additional correct decision**—not evidence of a robust gain.

NLL is log loss; lower is better. The two calibrated columns use separate,
held-out temperatures. Calibration does **not** change accuracy. Before calibration,
trained NLL worsened from **0.2315 to 0.4104** on CLINC and **0.8181 to 1.1224** on
ContractNLI; both also worsened on validation loss. No failed fine-tune is hidden.

Text-demo partitions, in train / validation / calibration / test order:

- **BANKING77-10:** 1,029 / 221 / 228 / 400 messages.
- **CLINC banking-10 + OOS:** 880 / 300 / 220 / 1,300 requests.
- **ContractNLI:** 340 / 68 / 136 / 170 decisions from 20 / 4 / 8 / 10 complete documents.
  The 2,048-token filter retains 64 of 123 official test documents; this demo selects
  10 of those 64 by fixed label-blind indices. It is not the full test benchmark.

[Exact budgets](examples/demo-suite.json) · [Measured environment](results/environment.json)
· [Regenerate these numbers](examples/summarize_results.py)

### Three useful tasks, one interface

| Demo | Input → decision | What we actually evaluate |
|---|---|---|
| BANKING77-10 | Customer message → payment-support intent | Ten predefined card/payment intents, not all 77 |
| CLINC banking-10 + OOS | Request → supported intent or out-of-scope | Ten banking intents plus OOS, not all 150 intents |
| ContractNLI | Full NDA + hypothesis → entailment / contradiction / not mentioned | All 17 fixed hypotheses on a disclosed short-document subset; no evidence-span extraction |
| Bitcoin | Past OHLCV features → next-day up / flat / down | Historical toy with a negative accuracy result; no trading |

CLINC's OOS examples are the dataset's official `oos_*` population. The other
140 named intents are excluded, **not** relabeled as OOS; rejection of those
omitted intents is not evaluated. Its test population contains 1,000 OOS requests
and 300 supported requests, so conditional metrics matter more than accuracy alone.

The interesting ContractNLI mapping is deliberately ordinary:

```text
prompt   = full contract text
question = a review hypothesis
choices  = entailment | contradiction | not_mentioned
label    = annotated choice
```

ContractNLI's original research baseline combined document segmentation and
span-level modeling. This demo tests a simpler **classification-only** route.
It does not claim to replace legal review, outperform that baseline, or solve
the dataset's evidence-identification task.

### Reproduce all three text demos

```bash
python examples/run_demo_suite.py \
  --output-root runs/reproduced \
  --evidence-root results/reproduced
```

The script prepares missing datasets, verifies existing split checksums, runs
one frozen-budget experiment per task, reloads each adapter, and exports evidence.
It runs models sequentially to avoid competing training jobs on the GPU.
Use new output/evidence roots for every attempt.

For ContractNLI alone, the published budget is:

```bash
ifllm-learn prepare-text contractnli --output data/contractnli-demo
ifllm-learn run \
  --data-dir data/contractnli-demo --output runs/my-contract-demo \
  --steps 64 --batch-size 2 --learning-rate 1e-5 \
  --train-limit 340 --valid-limit 68 --calibration-limit 136 --test-limit 170 \
  --num-layers 2 --rank 8 --max-tokens 2048 --seed 17
```

Limits count decision rows. Contract groups stay intact: 170 decisions means
10 documents × 17 hypotheses, not 170 independent contracts.

### The Bitcoin hook, with the punchline left in

The earlier 32-test-row exploratory run obtained **56.25% accuracy** for the
class-frequency prior, the frozen model, and the trained model. Calibration
changed log loss, not that accuracy. The trained model's validation loss worsened.

That experiment is here to show the workflow and its limits—not to sell a trading
signal. Its test rows were reused during development, and there is no claim of
profitability, prospective prediction skill, or statistical significance.

[Bitcoin evidence](results/bitcoin/report.json) · [Experiment history](results/preparation-notes.json)

```bash
ifllm-learn prepare-btc --output data/btc-2023-2025
ifllm-learn run \
  --data-dir data/btc-2023-2025 --output runs/my-bitcoin-toy \
  --steps 32 --batch-size 1 --train-limit 64 --valid-limit 12 \
  --calibration-limit 32 --test-limit 32 --max-tokens 1024 --seed 7
```

This runs the current workflow; the preserved historical Bitcoin report predates
the additional TF-IDF and frozen-model-calibration comparisons.

## Your own data

Use JSONL: one example per line. This is the schema, shown formatted for readability:

```json
{
  "id": "ticket-001",
  "prompt": "The deployment failed after the database migration.",
  "question": "Which team should investigate first?",
  "options": [
    {"id": "database", "description": "Database and migration failures."},
    {"id": "network", "description": "Network connectivity failures."},
    {"id": "unknown", "description": "Insufficient evidence to route reliably."}
  ],
  "label": "database"
}
```

- `prompt`: a nonempty string, JSON object, or array.
- `question`: the decision criterion.
- `options`: **2–16** uniquely identified choices with descriptions.
- `label`: the ground-truth option ID, used for training/evaluation only.
- `answer`: the predicted option ID, written by inference—not an input field.
- `metadata.group_id`: use the same ID for questions from the same source document.
- `observed_at` and `label_available_at`: timezone-aware ISO timestamps for temporal tasks.

Put `train.jsonl`, `valid.jsonl`, `calibration.jsonl`, and `test.jsonl` in a dataset
directory, then use `ifllm-learn run`. Keep the same declared question/ordered-option
contracts across partitions. An adapter is bound to its task contract: changing
questions, choices, or precision requires deliberate retraining/evaluation rather
than silently reusing calibration.

You own the truth labels and the feature boundary. The framework keeps top-level
labels/metadata out of model inputs; it cannot detect a future answer manually
embedded inside your `prompt`.

### Individual stages

```bash
ifllm-learn train --train data/custom/train.jsonl --valid data/custom/valid.jsonl \
  --output runs/custom-adapter --steps 100

ifllm-learn predict --input data/custom/calibration.jsonl \
  --adapter runs/custom-adapter --output runs/custom-calibration.jsonl

ifllm-learn calibrate --input runs/custom-calibration.jsonl \
  --output runs/custom-temperature.json

ifllm-learn predict --input data/custom/test.jsonl \
  --adapter runs/custom-adapter --calibration runs/custom-temperature.json \
  --output runs/custom-test.jsonl

ifllm-learn evaluate --input runs/custom-test.jsonl --output runs/custom-metrics.json
```

## How it works

```text
Full input + natural-language choices
                  │
            Qwen3.5-4B + LoRA
                  │
       last-position logits for A…P
                  │
     cross-entropy during training
                  │
    softmax over the declared choices
                  │
       optional held-out temperature
                  │
           typed decision scores
```

The loss targets the correct **choice**, not every token in the input. Base weights
are frozen; selected layers receive MLX-LM LoRA modules. `--batch-size` controls
sequential gradient accumulation, so variable-length inputs are not padded into
one training batch. The final optimizer state is evaluated; there is no hidden
test-set checkpoint search.

Temperature scaling learns a positive scalar `T` on a separate calibration set:

```text
p = softmax(choice_logits / T)
```

**It can soften overconfidence. It cannot change the winning class.** Both the
frozen and trained models get their own calibration comparison in the text demos.
An `unknown` option is a learned category, not a mathematical guarantee of abstention.

## Evidence, not screenshots of lucky answers

Each run retains:

```text
run.json                        # settings, selected IDs, dataset/source hashes
adapter/training.json           # real optimizer steps, losses, changed tensors
adapter/adapters.safetensors    # local learned weights
base-test.jsonl                 # frozen model outputs
trained-test.jsonl              # learned model outputs
base-temperature.json           # frozen model calibration
temperature.json                 # learned model calibration
report.json                     # metrics, baselines, coverage, limitations
reload-verification.json        # fresh model/adapter reload comparison
```

The compact [`results/`](results/) bundles exclude source document text and model
weights. Dataset source URLs, licenses, selection/filtering coverage, and hashes
are retained. Structured export filtering is not a general-purpose secret scanner:
IDs and approved provenance descriptions are retained. Inspect custom identifiers
before publishing private-data experiments.

To verify checksums and recompute LLM headline metrics **without loading a model**:

```bash
python examples/verify_results.py
```

To export a new verified run:

```bash
ifllm-learn export-run --run-dir runs/my-banking-demo \
  --data-dir data/banking-demo --output results/my-banking-demo
```

## Limitations

- **Classification toolkit, not a general agent.** No tool execution, generated rationales, extraction, or OpenAI-compatible chat endpoint.
- **One supported native model family/backend here.** The training path uses the pinned Qwen3.5-4B / MLX integration. CPU-only training, CUDA training, and arbitrary model architectures are not implemented by this package.
- **Explicit subsets.** We do not fit 77 or 150 labels into a 16-choice interface and call it a full benchmark.
- **Long documents are not silently truncated.** ContractNLI excludes an entire document if any of its 17 complete prompts exceeds 2,048 tokens; all coverage is disclosed. Results do not establish long-contract performance.
- **Source conflicts are audited.** Four ContractNLI documents with conflicting annotations on normalized-identical text are excluded in full before scoring. No preferred gold label is selected.
- **Hashes are provenance, not authentication.** BANKING/CLINC source revisions are pinned. ContractNLI's official ZIP URL is mutable; its first download is SHA-256 pinned in the local cache, not verified against an independently authenticated checksum.
- **No leakage immunity.** Group/time checks and exact normalized deduplication do not detect all paraphrases, user-embedded answers, or public benchmark contamination in pretraining.
- **No confidence guarantee.** Finite, normalized scores can still be confidently wrong. Threshold coverage curves are observed summaries, not safety guarantees.
- **No universal improvement claim.** Keep the frozen model and small classical baseline in the comparison. A failed fine-tune is a result, not something to hide.

## Development

```bash
python -m pytest -q
```

Tests cover real tiny-Qwen MLX gradient updates, adapter reload/tampering,
training/inference prompt alignment, group and temporal leakage, dataset source
validation, calibration identity, and metric edge cases. Tiny-model tests do not
download the 4B checkpoint. MLX-specific tests require Apple Silicon; CPU-only
preparation/evaluation tests remain separate from native inference.

## Sources and attribution

| Component | Source / terms |
|---|---|
| Semantic option scoring | [SemIf](https://github.com/TheoLeeCJ/SemIf), MIT; revision pinned in `pyproject.toml` |
| Training/runtime | [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm), upstream terms apply |
| Base model | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), upstream model license applies |
| ContractNLI | [Koreeda & Manning, 2021](https://stanfordnlp.github.io/contract-nli/), CC BY 4.0; classification-only derived subset |
| BANKING77 | [Casanueva et al., 2020 / PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), CC BY 4.0; ten-intent derived subset |
| CLINC150 / OOS | [Larson et al., 2019](https://github.com/clinc/oos-eval), CC BY 3.0; ten banking intents + OOS derived task |
| Bitcoin candles | [Binance Public Data](https://github.com/binance/binance-public-data), source archives verified against supplied SHA-256 checksums |

Raw datasets and pretrained weights are downloaded from their sources, not vendored
here. Retain the relevant dataset attribution and terms when redistributing derived
material. No affiliation or endorsement by the cited projects is implied.

---

**Bring labels. Train a decision. Keep the receipts.**
