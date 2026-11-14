# Protocol

Reference configuration for the audio-text depression detection experiments reported in
the manuscript: Qwen2-Audio-7B-Instruct adapted with LoRA under centralized and federated
training on DAIC-WOZ.

## Model and adaptation

Backbone `Qwen/Qwen2-Audio-7B-Instruct`. Weights are loaded in 4-bit NF4 with double
quantization and computed in bfloat16. LoRA is applied to the language model's `q_proj`
and `v_proj` projections and to the linear audio-to-language projector: rank 8, alpha 16,
dropout 0.05, no bias. This yields 65 target modules, 130 trainable tensors and 4,237,312
trainable parameters; the audio tower remains frozen.

## Federated setting

Three clients, 15 aggregation rounds, full participation. Every participant contributes
200 windows, and no participant appears in more than one client or more than one official
split. Client composition:

| Partition | Client 0 | Client 1 | Client 2 |
| --- | --- | --- | --- |
| iid | 36 (10) | 36 (10) | 35 (10) |
| mild | 36 (7) | 36 (10) | 35 (13) |
| severe | 36 (1) | 36 (9) | 35 (20) |
| quantity | 54 (15) | 32 (9) | 21 (6) |

Counts are participants, with depressed participants in parentheses. All clients share one
optimization configuration: learning rate 3e-6, effective batch 21, three epochs, AdamW
with weight decay 0.001 and gradient clipping at norm 0.5, linear warmup over 10% of the
steps followed by cosine decay.

## Algorithms

**FedAvg** averages the LoRA `A` and `B` tensors separately in float32, weighted by each
client's number of windows, then casts back to the parameter dtype. The product `BA` is
never averaged and adapters are never merged into the backbone.

**FedProx** adds `(mu/2) * sum(||adapter - round_start_adapter||^2)` with `mu = 0.01` over
all trainable adapter elements, once per optimizer step. The proximal term acts on the
adapters only.

**SCAFFOLD** uses the AdamW adaptation: clients and server keep float32 control variates
initialized to zero, `c - c_i` is added to the task gradient before each update, and
client controls are refreshed at the end of a segment as

```
c_i_new = c_i - c + (round_start_adapter - local_adapter) / sum(segment_LRs)
```

after which the server takes the window-weighted mean of the client controls.

Centralized and independent local-only training use the same adapter configuration, the
latter starting each client from the common initial adapter without communication.

## Evaluation

A window score is the difference in mean assistant-token negative log likelihood between
the two candidate responses, `NLL(not depressed) - NLL(depressed)`, averaged over a
participant's 200 windows. A positive-F1 threshold is fitted on Dev for each adapter and
frozen before Test is scored. Reported metrics are positive F1, ROC-AUC and
average precision, at window and participant level; three-client results are unweighted
means of the per-client metrics.

## Data

DAIC-WOZ is access-controlled: 107 / 35 / 47 participants for Train / Dev / Test, of whom
30 / 12 / 14 are depressed. Recordings, transcripts, labels, manifests and trained weights
are not redistributed in this repository.
