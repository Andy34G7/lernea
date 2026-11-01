- NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-7ff01406`; commit `950506beba`; generated 2026-11-01T16:28:54Z
- recipe: `karpathy/llm.c@f1e2ace651495b74ae22d45d1723443fd00ecd3a` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr 0.0006, wd 0.1, 524288 tokens/step, bf16, compiled) with 600 steps, warmup 22, val every 100 steps
- val: 14982 FineWeb documents (46.5 MB, the speedrun's first 10,485,760 GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits over every val token (one separator per document included) / UTF-8 bytes
- our tokenizers: trained on the first 16,157 train documents (50.0 MB) / 97,402 train documents (300.0 MB) at 50,256 tokens + 1 separator, pre-tokenizer gpt2 / space_prefix; TokenMonster: every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents
- target: the speedrun's 3.28 nats per GPT-2 token = ? (GPT-2 did not run)

| Tokenizer | \|V\| | Params | Val B/token | Train B/token | Final BPB | GPT-2-equivalent steps | Step at target | ms/step | tok/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unigram_lm_g2` | 50,257 | 124.5 M | 4.421 | 4.441 | **1.6817** | ? | not reached | 2657 | 197,295 |
| `mdl_bits_g2` | 50,257 | 124.5 M | 4.454 | 4.474 | **1.6910** | ? | not reached | 2655 | 197,489 |
| `wp_bpe_ppig_g2` | 50,257 | 124.5 M | 4.472 | 4.492 | **1.6946** | ? | not reached | 2654 | 197,515 |
| `unigram_lm_sp300` | 50,257 | 124.5 M | 4.704 | 4.721 | **1.7026** | ? | not reached | 2655 | 197,471 |
| `wp_bpe_pig_g2` | 50,257 | 124.5 M | 4.467 | 4.487 | **1.7029** | ? | not reached | 2655 | 197,460 |
| `wp_bpe_ppig_sp300` | 50,257 | 124.5 M | 4.807 | 4.827 | **1.7090** | ? | not reached | 2655 | 197,447 |
| `wp_bpe_npmi_g2` | 50,257 | 124.5 M | 4.429 | 4.450 | **1.7093** | ? | not reached | 2655 | 197,457 |
| `wp_bpe_npmi_sp300` | 50,257 | 124.5 M | 4.693 | 4.713 | **1.7207** | ? | not reached | 2654 | 197,563 |
| `mdl_bits_sp300` | 50,257 | 124.5 M | 4.719 | 4.739 | **1.7226** | ? | not reached | 2654 | 197,522 |
| `wp_bpe_pig_sp300` | 50,257 | 124.5 M | 4.800 | 4.821 | **1.7644** | ? | not reached | 2658 | 197,255 |
