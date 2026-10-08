# Evidence-Aware Proactive Consultation

This repository contains the implementation, training data, and evaluation entry points for **Evidence-Aware Proactive Consultation**. The method combines parameter-efficient LoRA adaptation with **Evidence-Aware Serial Conditioning (EASC)**. EASC adds a response-route head and a missing-information slot head to the shared backbone. The predicted route and slot then condition response generation.

The repository is available at:

`https://github.com/Rayz3rk/Evidence-Aware-Proactive-Consultation`

## Repository contents

```text
src/proactive_consultation/
  evidence_aware.py              # Data encoding, route and slot supervision, control tokens, and model heads
  evidence_aware_runtime.py      # Checkpoint loading, route prediction, and conditional generation
  scope3000.py                   # Dataset loading, grouped splitting, and task-mixture construction
  scope3000_evaluation.py        # Evaluation helpers and rule-based metrics

scripts/
  audit_evidence_aware_data.py   # Data and supervision audit
  train_evidence_aware.py        # LoRA plus route and slot head training
  run_evidence_aware.py          # Local response generation
  evaluate_evidence_aware.py     # Route, slot, and data checks
  evaluate_serial_main.py        # Main test-set generation
  evaluate_serial_checkpoints.py # Checkpoint evaluation
  evaluate_lightweight_matched_200.py
  run_lightweight_matched_200.ps1
  run_easc_auxiliary.py          # Auxiliary benchmark generation
  sync_lightweight_comparison.py
  sync_main_baselines_1000.py

tests/
  test_evidence_aware.py

data/
  EviGap-Med/scope2000_repaired_1k/  # 1,000 ordinary-response views and 1,000 evidence-missing inquiry views

example/
  data/fused_medical/general/test.jsonl  # Fixed main-evaluation test set
```

The repository does **not** contain private experiment configuration, manuscript sources, third-party papers, local checkpoints, or the remaining large example datasets.

## Dataset

`data/EviGap-Med/scope2000_repaired_1k` is the release used for the reported 1:1 LoRA+EASC training mixture. The principal files are:

| File | Purpose | SHA256 |
| --- | --- | --- |
| `sft_candidates.auto.jsonl` | Training views | `11c0d3dec0f9224065b71e467790aa52b309c030d992b1e4730873badb05c75f` |
| `paired_contexts.auto.jsonl` | Audit and provenance records | `22ca55a11411ae3ffc85127765002d2fc5b617be0422b2f81c026000682c784b` |

The release contains 1,000 ordinary-response views and 1,000 evidence-missing inquiry views with a 1:1 mixture and seed 42. Audit metadata is used for supervision and verification and is not inserted into model-visible prompts.

The source medical-dialogue datasets retain their original terms. Access and reuse require compliance with those terms. Inclusion in this repository does not grant a new license over underlying dialogue content and does not imply clinical validation.

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for the local CUDA environment first when GPU execution is required.

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The main implementation requires `torch`, `transformers`, `accelerate`, `peft`, `PyYAML`, `numpy`, and model-processing dependencies. `pyarrow` is required by the auxiliary benchmark script. `pytest` is required for tests.

## Prepare a base model

Training and inference require a locally available base model. The reported experiments use **Qwen3.5-0.8B**, but it is not redistributed here. Obtain the base model from its official provider under the applicable license and set its local path in your own YAML configuration.

A private configuration file is intentionally excluded. Create your own configuration with at least the following fields.

```yaml
local:
  model_path: models/Qwen3.5-0.8B
  torch_dtype: auto
  device_map: auto
  max_gpu_memory: 10GiB
  max_cpu_memory: 48GiB
  offload_folder: .cache/model_offload
  load_in_4bit: false
  max_input_tokens: 2048
  max_output_tokens: 256
  strip_reasoning: true

training:
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - in_proj_qkv
      - out_proj
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: false
  learning_rate: 0.0002
  epochs: 3
```

## Audit the released data

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/audit_evidence_aware_data.py `
  --data-file data/EviGap-Med/scope2000_repaired_1k/sft_candidates.auto.jsonl `
  --audit-file data/EviGap-Med/scope2000_repaired_1k/paired_contexts.auto.jsonl
```

## Train

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/train_evidence_aware.py `
  --config path/to/your_experiment.yaml `
  --data-file data/EviGap-Med/scope2000_repaired_1k/sft_candidates.auto.jsonl `
  --audit-file data/EviGap-Med/scope2000_repaired_1k/paired_contexts.auto.jsonl `
  --task-mode joint `
  --mix-ratio 1:1 `
  --seed 42 `
  --epochs 3 `
  --quality-status released
  --output-dir outputs/checkpoints/easc_run
```

The published release declares `quality_status: released`. The release record documents automatic screening and does not imply clinical validation.

## Generate responses

Create a JSON file containing a dialogue message list.

```json
[
  {"role": "user", "content": "最近几天头痛。"},
  {"role": "assistant", "content": "头痛主要在什么位置？"},
  {"role": "user", "content": "主要在额头。"}
]
```

Then run local generation.

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_evidence_aware.py `
  --checkpoint outputs/checkpoints/easc_run `
  --messages-json messages.json
```

## Evaluate

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/evaluate_evidence_aware.py `
  --checkpoint outputs/checkpoints/easc_run `
  --data-file data/EviGap-Med/scope2000_repaired_1k/sft_candidates.auto.jsonl `
  --audit-file data/EviGap-Med/scope2000_repaired_1k/paired_contexts.auto.jsonl
```

The fixed main-evaluation input is provided at `example/data/fused_medical/general/test.jsonl`.

## Run tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_evidence_aware.py -q
```

## Checkpoint availability

The trained **E3 checkpoint is not included** because its `pytorch_model.bin` is approximately 1.60 GiB. The repository therefore provides code and training data, but not a directly runnable trained model. Reproducing inference requires the base model plus a compatible trained checkpoint.

## Reproducibility notes

- The reported primary checkpoint is epoch 3 of the reported LoRA+EASC run.
- The published data release uses seed 42 and a 1:1 task mixture.
- Exact results can vary with the base-model release, Transformers version, GPU type, and decoding configuration.
- This is research code for offline experiments and is not a medical device.

