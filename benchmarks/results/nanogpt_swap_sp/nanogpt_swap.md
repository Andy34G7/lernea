- NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-55b11b60`; commit `2df09a7847`; generated 2026-11-01T10:53:38Z
- recipe: `karpathy/llm.c@f1e2ace651495b74ae22d45d1723443fd00ecd3a` scripts/pyrun_gpt2_124M.sh (GPT-2 124M, tied embeddings, AdamW lr 0.0006, wd 0.1, 524288 tokens/step, bf16, compiled) with 600 steps, warmup 22, val every 100 steps
- val: 14982 FineWeb documents (46.5 MB, the speedrun's first 10,485,760 GPT-2 val tokens), the same bytes for every tokenizer; BPB = NLL bits over every val token (one separator per document included) / UTF-8 bytes
- our tokenizers: trained on the first 16,157 train documents (50.0 MB) at 50,256 tokens + 1 separator, pre-tokenizer space_prefix; TokenMonster: every prebuilt englishcode vocabulary NFD-normalizes its input (lossy on 1,053 of 14,982 val documents); capcode also folds U+0130 to I (2 documents). Used: the nocapcode vocabulary of the same size and mode, normalization flag cleared (same tokens and ids), lossless on all val documents
- target: the speedrun's 3.28 nats per GPT-2 token = ? (GPT-2 did not run)

| Tokenizer | \|V\| | Params | Val B/token | Train B/token | Final BPB | GPT-2-equivalent steps | Step at target | ms/step | tok/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mdl_bits_sp` | 50,257 | 124.5 M | 4.689 | 4.707 | **1.7067** | ? | not reached | 2648 | 197,980 |
| `unigram_lm_sp` | 50,257 | 124.5 M | 4.692 | 4.709 | **1.7088** | ? | not reached | 2647 | 198,084 |
| `wp_bpe_npmi_sp` | 50,257 | 124.5 M | 4.658 | 4.676 | **1.7436** | ? | not reached | 2649 | 197,901 |
| `wp_bpe_ppig_sp` | 50,257 | 124.5 M | 4.777 | 4.796 | **1.7552** | ? | not reached | 2647 | 198,065 |
| `morpheme_bpe_sp` | 50,257 | 124.5 M | 4.710 | 4.728 | **1.7742** | ? | not reached | 2648 | 197,968 |
| `wp_bpe_pig_sp` | 50,257 | 124.5 M | 4.769 | 4.788 | **1.7749** | ? | not reached | 2648 | 197,989 |
| `bpe_frequency_sp` | 50,257 | 124.5 M | 4.831 | 4.851 | **1.8000** | ? | not reached | 2650 | 197,820 |
| `length_penalized_sp` | 50,257 | 124.5 M | 4.822 | 4.841 | **1.9008** | ? | not reached | 2648 | 197,978 |
