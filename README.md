# Federated large-language model for depression detection

Anonymous code release accompanying the manuscript: centralized and federated QLoRA
adaptation of Qwen2-Audio-7B-Instruct on DAIC-WOZ, with participant-level evaluation.

## What is included

| Files | Contents |
| --- | --- |
| `preprocess.py`, `visual_clean.py`, `windows.py` | Cleaning and fixed-length window generation |
| `partitions.py` | Four participant-disjoint client partitions |
| `data.py`, `model.py` | Audio-text inputs, frozen backbone with LoRA adapters |
| `algorithms.py` | Weighted FedAvg, adapter-level FedProx, SCAFFOLD control correction |
| `schedule.py`, `engine.py`, `train.py` | Training plans, checkpoint recovery, CLI |
| `metrics.py`, `evaluate.py` | Participant-level evaluation |
| `config.py`, `settings.json`, `storage.py` | Shared configuration and atomic artifacts |
| `test_release.py` | Synthetic CPU self-test |

Covered settings: centralized, local-only, FedAvg, FedProx and SCAFFOLD,
each over the `iid`, `mild`, `severe` and `quantity` partitions.

## Evaluation protocol

- Client partitions are participant-disjoint and generated from the training split only.
- Every adapter gets a positive-F1 threshold fitted on Dev and frozen before Test;
  Test is never used for threshold selection.
- Reported at window and participant level: positive F1, ROC-AUC, average precision.

## Environment

Linux, Python 3.10, `torch==2.7.0` (CUDA 11.8 wheels), one 24 GB GPU with bfloat16.
Install with `pip install -r requirements.txt`. Shared defaults — learning rate 3e-6,
effective batch 21, three epochs, 15 federated rounds — are in `settings.json`.

## Pipeline

`preprocess.py` → `partitions.py` → `train.py` → `evaluate.py`.
Exact settings are documented in `PROTOCOL.md`; the release self-check is recorded in
`VALIDATION.md`.

## Not included

DAIC-WOZ audio, transcripts and labels are access-controlled and must be obtained under
their own licence. No trained weights, result tables, credentials or machine-specific
paths are released.
