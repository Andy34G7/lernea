- PILOT (timing only, not a result): NVIDIA L4, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-690089c7`; commit `c83ba0da10`, tokenizers prepared at commit `a04d3c3ff3`; generated 2026-10-31T18:43:38Z
- recipe: `karpathy/llm.c@f1e2ace651495b74ae22d45d1723443fd00ecd3a` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr 0.0006, wd 0.1, 524288 tokens/step, bf16, compiled) with 20 steps, warmup 1, val every 20 steps
- val: 14982 FineWeb documents (46.5 MB, the speedrun's first 10,485,760 GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits over every val token (one separator per document included) / UTF-8 bytes
- our tokenizers: trained on the first 16,157 train documents (50.0 MB) at 50,256 tokens + 1 separator; TokenMonster: every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents
- target: the speedrun's 3.28 nats per GPT-2 token = 1.0667 BPB on this val text
- note: the L4 session reached the job's 1,200 s timeout about a minute after its job had finished, and was stopped before the run files of bpe_frequency (trained) and 8 val-only tokenizers were downloaded. Their val statistics and encode MB/s here come from the A100 pilot (same tokenizers and documents; encode measured on that VM's CPU). bpe_frequency's L4 ms/step is estimated from GPT-2's (identical model and padded vocabulary; on the A100 they differ by 0.7%). L4 setup (pip, extension build) was not recorded (A100 VM: 5 s + 41 s); download and decode times are from the L4 job log

| Tokenizer | \|V\| | Val B/token | Encode MB/s | ms/step | tok/s | Peak GiB | Warm eval s | Projected h |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 50,257 | 4.436 | 31.4 | 13441 | 39,007 | 9.0 | 26 | 3.01 |
| `tm_englishcode_50256` | 50,257 | 5.322 | 67.3 | (13441) | | | | 2.89 |
| `tm_englishcode_32000` | 32,001 | 4.954 | 69.7 | (12131) | | | | 2.65 |
| `bpe_frequency` | 50,257 | 2.750 | 33.4 | (13441) | | | | 3.42 |
| `wp_bpe_pig` | 50,257 | 2.733 | 33.9 | (13441) | | | | 3.43 |
| `wp_bpe_ppig` | 50,257 | 2.736 | 33.7 | (13441) | | | | 3.43 |
| `wp_bpe_npmi` | 50,257 | 2.702 | 32.1 | (13441) | | | | 3.44 |
| `super_bpe` | 50,257 | 5.031 | 26.9 | (13441) | | | | 2.93 |
| `unigram_lm` | 50,257 | 2.725 | 29.1 | (13441) | | | | 3.43 |
| `length_penalized` | 50,257 | 2.749 | 31.4 | (13441) | | | | 3.42 |
| `morpheme_bpe` | 50,257 | 2.718 | 31.1 | (13441) | | | | 3.43 |
| `mdl_bits` | 50,257 | 2.718 | 37.2 | (13441) | | | | 3.43 |

- setup: pip ? s, extension build ? s, downloads (s) {'fineweb_val_000000.bin': 3.4, 'fineweb_train_000001.bin': 2.4}, decode (s) {'train_decode': 11.0}
- projection, one session, sequential: 38.9 h for 12 tokenizers x 600 steps (13 evals each, source text 1.69 GB); ms/step in parentheses: estimated from a timed run by FLOPs per token
