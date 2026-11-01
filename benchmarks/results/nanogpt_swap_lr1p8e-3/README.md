# GPT-2 small, peak lr 1.8e-3 (stopped after the GPT-2 baseline)

Same tokenizer-swap job as `benchmarks/results/nanogpt_swap/` (llm.c GPT-2 124M recipe, 600 steps,
524,288 tokens/step, warmup 22, cosine to 0, Colab A100) with one change: `--lr 1.8e-3` (3x llm.c's
6e-4, the AdamW lr from modded-nanogpt's early records) instead of 6e-4, identical for every tokenizer.

The sweep was stopped after its first tokenizer (GPT-2) because the higher lr was worse at every
evaluation, without instability (no loss spikes; max grad norm after warmup 0.81):

| step | val BPB, lr 6e-4 | val BPB, lr 1.8e-3 |
| ---: | ---: | ---: |
| 0 | 3.5710 | 3.5710 |
| 100 | 2.0912 | 2.1412 |
| 200 | 1.9380 | 2.0234 |
| 300 | 1.8249 | 1.9265 |
| 400 | 1.7391 | 1.8610 |
| 500 | 1.6945 | 1.8184 |
| 600 | 1.6851 | 1.8076 |

modded-nanogpt raised the lr together with architecture changes (rotary embeddings etc.); on plain
llm.c GPT-2 at this 600-step budget 3x is too high. `gpt2_run.json` is the raw run file.
