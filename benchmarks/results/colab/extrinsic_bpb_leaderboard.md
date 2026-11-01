- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out val split (3.1 MB); commit `603a983557`; generated 2026-10-30T14:06:27Z
- arguments: `--skip-memory-check --gpu T4 --timeout 10800`
- backend: Colab GPU `Tesla T4`
- proxy LM: pre-LayerNorm causal Transformer (torch TransformerEncoderLayer: ReLU FFN, biases), no positional encoding of any kind (NoPE: order comes only from the causal mask), tied input/output embeddings; d_model=128, n_layers=2, n_heads=4, d_ffn=512, max_seq_len=128
- fixed budget 10,000,000 UTF-8 bytes of LM training text per tokenizer, drawn from the same train split the tokenizers were trained on (each document used once); seeds [0, 1, 2] (mean ± sample sd; the seed sets only the LM's initialization: every seed trains on the same LM text in the same order, so the sd leaves out data-sampling variance)

| Tokenizer | \|V\| | Train bytes/token | Tokens trained | english BPB (↓) | multilingual BPB (↓) | code BPB (↓) | Macro BPB (↓) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp_bpe_pig` | 16384 | 2.597 | 3,858,432 | 2.465 ± 0.009 | 3.174 ± 0.048 | 2.922 ± 0.031 | **2.854 ± 0.029** |
| `wp_bpe_ppig` | 16384 | 2.605 | 3,846,144 | 2.467 ± 0.005 | 3.229 ± 0.018 | 2.925 ± 0.029 | **2.874 ± 0.015** |
| `super_bpe` | 16384 | 3.696 | 2,711,552 | 2.503 ± 0.009 | 3.229 ± 0.028 | 2.920 ± 0.016 | **2.884 ± 0.015** |
| `wp_bpe_npmi` | 16384 | 2.539 | 3,948,544 | 2.467 ± 0.002 | 3.215 ± 0.039 | 2.974 ± 0.011 | **2.886 ± 0.017** |
| `bpe_frequency` | 16384 | 2.639 | 3,796,992 | 2.494 ± 0.005 | 3.244 ± 0.019 | 2.938 ± 0.003 | **2.892 ± 0.004** |
| `length_penalized` | 16384 | 2.631 | 3,807,232 | 2.511 ± 0.012 | 3.242 ± 0.021 | 2.974 ± 0.026 | **2.909 ± 0.014** |
| `bpe_dropout_p0.1` | 16384 | 2.500 | 4,007,936 | 2.520 ± 0.012 | 3.255 ± 0.025 | 2.983 ± 0.046 | **2.919 ± 0.013** |
| `unigram_lm` | 16384 | 2.581 | 3,878,912 | 2.480 ± 0.009 | 3.362 ± 0.023 | 3.011 ± 0.033 | **2.951 ± 0.021** |
| `morpheme_bpe` | 16384 | 2.521 | 3,973,120 | 2.489 ± 0.004 | 3.348 ± 0.032 | 3.068 ± 0.010 | **2.968 ± 0.015** |
| `mdl_bits` | 16384 | 2.434 | 4,126,720 | 2.495 ± 0.014 | 3.466 ± 0.015 | 3.304 ± 0.022 | **3.088 ± 0.016** |
| `wp_bpe_scp` | 16384 | 1.617 | 6,195,200 | 3.386 ± 0.021 | 3.350 ± 0.032 | 3.837 ± 0.059 | **3.524 ± 0.037** |
| `wordpiece_pmi` | 16384 | 1.013 | 9,875,456 | 3.708 ± 0.007 | 3.719 ± 0.024 | 4.137 ± 0.033 | **3.855 ± 0.018** |
| `wp_bpe_ft` | 16384 | 1.013 | 9,877,504 | 3.703 ± 0.033 | 3.710 ± 0.077 | 4.150 ± 0.098 | **3.855 ± 0.069** |
| `wp_bpe_base` | 16384 | 1.006 | 9,945,088 | 3.734 ± 0.019 | 3.829 ± 0.021 | 4.198 ± 0.044 | **3.920 ± 0.018** |
