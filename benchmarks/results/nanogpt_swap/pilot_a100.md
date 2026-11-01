- PILOT (timing only, not a result): NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-f09133da`; commit `5d437be6cb`, tokenizers prepared at commit `a04d3c3ff3`; generated 2026-10-31T18:41:59Z
- recipe: `karpathy/llm.c@f1e2ace651495b74ae22d45d1723443fd00ecd3a` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr 0.0006, wd 0.1, 524288 tokens/step, bf16, compiled) with 20 steps, warmup 1, val every 20 steps
- val: 14982 FineWeb documents (46.5 MB, the speedrun's first 10,485,760 GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits over every val token (one separator per document included) / UTF-8 bytes
- our tokenizers: trained on the first 16,157 train documents (50.0 MB) at 50,256 tokens + 1 separator; TokenMonster: every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents
- target: the speedrun's 3.28 nats per GPT-2 token = 1.0667 BPB on this val text

| Tokenizer | \|V\| | Val B/token | Encode MB/s | ms/step | tok/s | Peak GiB | Warm eval s | Projected h |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 50,257 | 4.436 | 30.7 | 2594 | 202,104 | 17.2 | 5 | 0.63 |
| `tm_englishcode_50256` | 50,257 | 5.322 | 65.8 | (2594) | | | | 0.57 |
| `tm_englishcode_32000` | 32,001 | 4.954 | 64.4 | (2341) | | | | 0.56 |
| `bpe_frequency` | 50,257 | 2.750 | 33.4 | 2613 | 200,632 | 17.2 | 9 | 0.68 |
| `wp_bpe_pig` | 50,257 | 2.733 | 33.9 | (2594) | | | | 0.68 |
| `wp_bpe_ppig` | 50,257 | 2.736 | 33.7 | (2594) | | | | 0.68 |
| `wp_bpe_npmi` | 50,257 | 2.702 | 32.1 | (2594) | | | | 0.68 |
| `super_bpe` | 50,257 | 5.031 | 26.9 | (2594) | | | | 0.58 |
| `unigram_lm` | 50,257 | 2.725 | 29.1 | (2594) | | | | 0.68 |
| `length_penalized` | 50,257 | 2.749 | 31.4 | (2594) | | | | 0.68 |
| `morpheme_bpe` | 50,257 | 2.718 | 31.1 | (2594) | | | | 0.68 |
| `mdl_bits` | 50,257 | 2.718 | 37.2 | (2594) | | | | 0.68 |

- setup: pip 5.0 s, extension build 41.0 s, downloads (s) {'fineweb_val_000000.bin': 2.1, 'fineweb_train_000001.bin': 1.9}, decode (s) {'val_decode': 3.4, 'train_decode': 10.3}
- projection, one session, sequential: 7.8 h for 12 tokenizers x 600 steps (13 evals each, source text 1.69 GB); ms/step in parentheses: estimated from a timed run by FLOPs per token
