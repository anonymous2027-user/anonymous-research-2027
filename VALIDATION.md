# Validation record

Release self-check, 2026-11-13, Python 3.10 and PyTorch 2.7.0+cu118.

- The 14 synthetic unit tests in `test_release.py` pass. They cover the four partitions and
  participant disjointness, duplicate rejection, window coverage, weighted aggregation,
  FedProx gradients, SCAFFOLD corrections and controls, positive F1 / ROC-AUC / average
  precision, Dev-only thresholding, audio interval and text formatting, and optimizer and
  RNG recovery in all five training modes.
- `--plan-only` checks for centralized and quantity-FedAvg training reproduce 64,200 window
  presentations over three epochs, with 3,060 and 3,069 planned updates respectively and 15
  federated aggregations.
- Nine windows sampled across Train, Dev and Test match the original loader exactly in both
  concatenated waveform and conversational text.
- The pinned processor for the Qwen revision was exercised on a synthetic waveform and
  conversation: chat-prefix alignment, answer-token masking, audio feature dimensions and
  the 30-second cap all hold.
- A source scan found no account identifiers, machine-specific paths, SSH configuration or
  keys. No experiment outputs, licensed data, credentials, checkpoints or manuscript files
  are distributed.
