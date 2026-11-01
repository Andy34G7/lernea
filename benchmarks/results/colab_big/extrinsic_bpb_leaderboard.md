- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out val split (3.1 MB); commit `6d260c23b6`, resuming a job (tokenizers, LM text and its ids) prepared at commit `3186c1ba81` from a dirty tree (uncommitted changes: not reproducible from any commit); generated 2026-10-31T20:24:38Z
- arguments: `--preset big --job-dir /home/andey/.cache/tokenisers/big-lm-job --resume --concurrency 1`
- backend: Colab GPU `NVIDIA L4`
- proxy LM: pre-LayerNorm causal Transformer (torch TransformerEncoderLayer: ReLU FFN, biases), no positional encoding of any kind (NoPE: order comes only from the causal mask), tied input/output embeddings; d_model=512, n_layers=8, n_heads=8, d_ffn=2048, max_seq_len=512
- fixed budget 100,000,000 UTF-8 bytes of LM training text per tokenizer, drawn from the train split the tokenizers were trained on plus an LM text pool with no val or test document (each document used once; 62.1 MB of the 110.0 MB of LM text came from the pool, `meta.lm_text`); seeds [0, 1, 2] (mean ± sample sd; the seed sets only the LM's initialization: every seed trains on the same LM text in the same order, so the sd leaves out data-sampling variance)

| Tokenizer | \|V\| | Train bytes/token | Tokens trained | english BPB (↓) | multilingual BPB (↓) | code BPB (↓) | Macro BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp_bpe_ppig` | 16384 | 2.599 | 38,502,400 | 2.318 ± 0.005 | 2.715 ± 0.005 | 2.533 ± 0.006 | **2.522 ± 0.005** |
| `wp_bpe_pig` | 16384 | 2.591 | 38,600,704 | 2.318 ± 0.001 | 2.723 ± 0.006 | 2.542 ± 0.005 | **2.528 ± 0.001** |
| `wp_bpe_npmi` | 16384 | 2.529 | 39,550,976 | 2.326 ± 0.001 | 2.718 ± 0.002 | 2.549 ± 0.008 | **2.531 ± 0.003** |
| `super_bpe` | 16384 | 3.670 | 27,262,976 | 2.253 ± 0.003 | 2.806 ± 0.006 | 2.553 ± 0.001 | **2.537 ± 0.003** |
| `bpe_frequency` | 16384 | 2.634 | 37,978,112 | 2.341 ± 0.002 | 2.752 ± 0.002 | 2.555 ± 0.003 | **2.549 ± 0.001** |
| `morpheme_bpe` | 16384 | 2.519 | 39,714,816 | 2.289 ± 0.005 | 2.773 ± 0.005 | 2.590 ± 0.005 | **2.551 ± 0.005** |
| `unigram_lm` | 16384 | 2.578 | 38,830,080 | 2.308 ± 0.003 | 2.776 ± 0.005 | 2.570 ± 0.004 | **2.551 ± 0.004** |
| `bpe_dropout_p0.1` | 16384 | 2.494 | 40,108,032 | 2.356 ± 0.006 | 2.750 ± 0.009 | 2.564 ± 0.007 | **2.557 ± 0.007** |
| `length_penalized` | 16384 | 2.628 | 38,076,416 | 2.353 ± 0.004 | 2.760 ± 0.001 | 2.585 ± 0.004 | **2.566 ± 0.003** |
| `mdl_bits` | 16384 | 2.398 | 41,746,432 | 2.325 ± 0.002 | 2.739 ± 0.008 | 2.664 ± 0.006 | **2.576 ± 0.005** |
| `wp_bpe_scp` | 16384 | 1.621 | 61,702,144 | 3.015 ± 0.008 | 2.802 ± 0.009 | 3.161 ± 0.029 | **2.993 ± 0.015** |
