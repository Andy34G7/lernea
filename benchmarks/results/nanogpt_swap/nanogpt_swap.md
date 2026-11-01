- NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-4c6b4d48`; commit `ceb11bac38`, tokenizers prepared at commit `a04d3c3ff3`; generated 2026-11-01T02:01:33Z
- recipe: `karpathy/llm.c@f1e2ace651495b74ae22d45d1723443fd00ecd3a` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr 0.0006, wd 0.1, 524288 tokens/step, bf16, compiled) with 600 steps, warmup 22, val every 100 steps
- val: 14982 FineWeb documents (46.5 MB, the speedrun's first 10,485,760 GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits over every val token (one separator per document included) / UTF-8 bytes
- our tokenizers: trained on the first 16,157 train documents (50.0 MB) at 50,256 tokens + 1 separator; TokenMonster: every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents
- target: the speedrun's 3.28 nats per GPT-2 token = 1.0667 BPB on this val text

| Tokenizer | \|V\| | Params | Val B/token | Train B/token | Final BPB | GPT-2-equivalent steps | Step at target | ms/step | tok/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 50,257 | 124.5 M | 4.436 | 4.452 | **1.6851** | 600 | not reached | 2665 | 196,731 |
| `tm_englishcode_50256` | 50,257 | 124.5 M | 5.322 | 5.357 | **1.8678** | 253 | not reached | 2670 | 196,359 |
| `tm_englishcode_32000` | 32,001 | 110.5 M | 4.954 | 4.986 | **1.9518** | 183 | not reached | 2434 | 215,379 |
| `wp_bpe_ppig` | 50,257 | 124.5 M | 2.736 | 2.740 | **1.9561** | 181 | not reached | 2679 | 195,680 |
| `mdl_bits` | 50,257 | 124.5 M | 2.718 | 2.722 | **1.9684** | 172 | not reached | 2680 | 195,611 |
| `morpheme_bpe` | 50,257 | 124.5 M | 2.718 | 2.721 | **1.9695** | 171 | not reached | 2681 | 195,561 |
| `wp_bpe_npmi` | 50,257 | 124.5 M | 2.702 | 2.706 | **1.9835** | 160 | not reached | 2679 | 195,693 |
| `wp_bpe_pig` | 50,257 | 124.5 M | 2.733 | 2.738 | **2.0077** | 144 | not reached | 2679 | 195,678 |
| `unigram_lm` | 50,257 | 124.5 M | 2.725 | 2.729 | **2.0239** | 135 | not reached | 2684 | 195,351 |
| `bpe_frequency` | 50,257 | 124.5 M | 2.750 | 2.754 | **2.0259** | 133 | not reached | 2683 | 195,414 |
| `super_bpe` | 50,257 | 124.5 M | 5.031 | 5.054 | **2.0301** | 130 | not reached | 2671 | 196,261 |
| `length_penalized` | 50,257 | 124.5 M | 2.749 | 2.752 | **2.0393** | 125 | not reached | 2680 | 195,659 |

- GPT-2-equivalent steps: the first step at which GPT-2's train loss, in bits per byte of its train text (nats / ln 2 / 4.452 bytes per token) and smoothed by a centred 21-step rolling mean, reaches the tokenizer's final val BPB: how many of GPT-2's 600 steps buy the same quality (GPT-2: 600 by definition). The train text is one pass over unseen documents; at GPT-2's val passes (steps 100-600) the smoothed train BPB is within 0.0074 of val BPB (largest at step 100).
- train: each tokenizer's first 600 x 524,288 tokens of the same FineWeb train documents in the same order, one pass; one run per tokenizer, llm.c's fixed seed

Findings (one seed per tokenizer: the seed-to-seed spread is not measured here):

- Lowest final BPB: `gpt2` (1.6851), GPT-2's own tokenizer; next `tm_englishcode_50256` (1.8678, +0.1828).
- More val bytes per token than GPT-2's 4.436: `tm_englishcode_50256` 5.322, `tm_englishcode_32000` 4.954, `super_bpe` 5.031. Their final BPB is 1.8678-2.0301, all above GPT-2's: more text per step did not give a lower BPB.
- Our 8 whitespace-isolating tokenizers (the repo's pre-tokenizer makes every whitespace run its own token) get 2.70-2.75 val bytes per token, so each 524,288-token step holds 38-39% less text than GPT-2's (train bytes per token). The comparison is confounded by that design choice: they train on less text, not only on different tokens.
- Early training: after 600 steps GPT-2 is at 5.18 nats per token on val, far from the speedrun's target of 3.28; no tokenizer reaches the target. The ranking at this budget need not hold for longer training.
