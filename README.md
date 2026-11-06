# Lernea

Rust (PyO3/maturin) + Python research platform for subword tokenization algorithms. Trains, evaluates and compares 13+ tokenizer variants across compression, intrinsic quality, proxy-LM BPB, GPT-2 swap and cross-lingual parity (FLORES+).

<details><summary>Repository layout</summary>

```text
├── Cargo.toml                      # Rust workspace manifest
├── pyproject.toml                  # Maturin + uv build & CLI configuration
├── crates/
│   ├── tokenisers_core/            # Pure Rust core library
│   │   └── src/
│   │       ├── arena.rs            # Doubly-linked symbol arena & inverted pair index
│   │       ├── vocab.rs            # Byte-level lossless base alphabet & vocabulary mapping
│   │       ├── scoring/            # Pluggable scorers: Frequency, WordPiece/PMI, MDL, Length-Penalized
│   │       ├── engine/             # Bottom-up merge engine & Top-Down Unigram EM + Viterbi pruning
│   │       ├── curricula/          # Two-Stage SuperBPE & Morpheme-Guided boundary priors
│   │       ├── inference/          # Greedy, BPE-Dropout, Unigram FFBS/k-best, Byte-Entropy patching
│   │       └── export/             # HuggingFace tokenizer.json serializer
│   └── tokenisers_py/              # PyO3 bindings with GIL release
├── python/tokenisers/
│   ├── models.py                   # Python API (BpeTokenizer, UnigramTokenizer, etc.)
│   ├── wrapper.py                  # HuggingFace PreTrainedTokenizer subclass
│   ├── data/                       # Dataset pipeline (FineWeb-Edu, FineWeb-2, StarCoder, SIGMORPHON)
│   ├── dashboard/                  # Interactive web dashboard & REST API
│   └── benchmarks/                 # Tokka-Bench, intrinsic, proxy-LM, FLORES+, throughput, GPT-2 swap
├── vendor/tokka-bench/             # Vendored reference suite
├── benchmarks/results/             # Generated JSON, CSV and Markdown leaderboards
└── tests/e2e/                      # E2E & adversarial tests
```

</details>

---

## Tokenizer Algorithms

| Family | Variant | Objective / Mechanism |
| :--- | :--- | :--- |
| **Bottom-Up Merge** | `bpe_frequency` | Standard Byte-Level BPE: $\text{score}(u, v) = \text{count}(uv)$ |
| **Bottom-Up Merge** | `wordpiece_pmi` | Laplace-smoothed WordPiece likelihood: $\frac{\text{count}(uv)}{(\text{count}(u)+\alpha)(\text{count}(v)+\alpha)}$, $\alpha = 1$ by default ($\alpha = 0$ with no cutoff is exactly `wp_bpe_base`) |
| **WP-BPE (Gen 1)** | `wp_bpe_base` | **wp-bpe-base**: Uncalibrated PMI BPE merges without frequency cutoff: $\frac{\text{count}(uv)}{\text{count}(u)\text{count}(v)}$ |
| **WP-BPE (Gen 1)** | `wp_bpe_ft` | **wp-bpe-ft**: Frequency-Thresholded PMI BPE merges: same ratio, only for $\text{freq}(uv) \ge 2$ |
| **WP-BPE (Gen 1)** | `wp_bpe_pig` | **wp-bpe-pig**: Pointwise Information Gain BPE: $\text{freq}(uv) \cdot \text{PMI}(u, v)$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_scp` | **wp-bpe-scp**: Symmetric Conditional Probability / Geometric-Mean Calibrated PMI with Kneser-Ney discounting: $\frac{(\text{freq}(uv) - d)_+}{\sqrt{\text{freq}(u)\text{freq}(v)}}$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_npmi` | **wp-bpe-npmi**: Bouma Normalized PMI ($\text{NPMI} \in [-1, 1]$) weighted by sub-linear frequency support $\text{freq}(uv)^\alpha$ |
| **WP-BPE (Gen 2 Improved)** | `wp_bpe_ppig` | **wp-bpe-ppig**: Positive Length-Regularized Pointwise Information Gain: $\frac{\text{freq}(uv)\cdot\max(0, \text{PMI})}{(|u|+|v|)^\eta}$ |
| **Bottom-Up Merge** | `mdl_bits` | Minimum Description Length ($\Delta\text{Bits}$): corpus compression gain minus codebook overhead |
| **Bottom-Up Merge** | `length_penalized` | Length-Penalized BPE: $\text{count}(uv) / (|u| + |v|)^\beta$ |
| **Curriculum BPE** | `super_bpe` | **Two-Stage SuperBPE**: enforces whitespace pre-tokenization up to $V_{\text{switch}}$, then lifts whitespace barriers up to $V_{\text{final}}$ |
| **Morphological Prior** | `morpheme_bpe` | **Morpheme-Guided BPE**: scales each candidate's score by $e^{\lambda M(u,v)}$, where $M \in [-1,1]$ is the share of the pair's current occurrences in lexicon words whose merged token stays inside one morpheme minus the share that would span a boundary; pairs with $\text{count}(uv)\,e^{\lambda M} < 1$ are not merged. The caller supplies the lexicon (training annotations or unsupervised boundaries) |
| **Top-Down Pruning** | `unigram_lm` | **Unigram Language Model**: suffix-seeded trie + log-space forward-backward EM and Viterbi loss pruning |
| **Stochastic / Dynamic** | `bpe_dropout` | **BPE-Dropout**: randomly skips eligible merges with probability $p$ during LM training |
| **Stochastic / Dynamic** | `unigram_sampling` | **Subword Regularization**: Forward-Filtering Backward-Sampling (FFBS) & $k$-best Viterbi lattice sampling |
| **Stochastic / Dynamic** | `byte_entropy` | **Byte-Entropy Dynamic Patching**: places patch boundaries where Markov byte transition entropy exceeds threshold $\tau$ |

**Default parameters.** `min_frequency`: 2 for `wordpiece_pmi`, `wp_bpe_ft`, `wp_bpe_scp`, `wp_bpe_npmi`, `wp_bpe_ppig` and `mdl_bits`; 1 for others. $\alpha = 1$ (`wordpiece_pmi`), $d = 0.75$ (`wp_bpe_scp`), $\alpha = 0.5$ (`wp_bpe_npmi`), $\eta = 0.35$ (`wp_bpe_ppig`), 24 bits per entry (`mdl_bits`), $\beta = 0.5$ (`length_penalized`), 10 EM iterations with shrinking 0.75 (`unigram_lm`), $\lambda = 2$ (`morpheme_bpe`). Each result file records all training parameters under `meta.variants`.

**Pre-tokenizer.** Four modes, applied identically for training, encoding and `tokenizer.json` export: `"whitespace"` (default: isolates whitespace runs), `"space_prefix"` (GPT-2-style word-leading space), `"gpt2"` (tiktoken's exact regex; exports as HF `ByteLevel(use_regex=true)`), and `{"hf": config}` (any lossless HF `pre_tokenizer` config: `Split`, `Punctuation`, `Digits`, `FixedLength`, `ByteLevel` without `add_prefix_space`, and `Sequence`s of these).

```python
from tokenisers import BpeTokenizer

llama_digits = {"type": "Sequence", "pretokenizers": [
    {"type": "Digits", "individual_digits": True},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True}]}
tok = BpeTokenizer.train(corpus, vocab_size=32000, pretokenizer={"hf": llama_digits})
tok.encode("abc 2024")  # segments "abc", " ", "2", "0", "2", "4": no token holds two digits
```

---

## Quickstart

```bash
# 1. Build & test
uv sync --extra dev
cargo test --workspace
uv run maturin develop --release
uv run pytest tests/e2e/ -v

# 2. Dashboard (http://127.0.0.1:8765)
uv run python -m tokenisers.dashboard --port 8765

# 3. Prepare corpus (FineWeb-Edu + FineWeb-2 + code, document-disjoint train/val/test)
uv run python -m tokenisers.data --preset small --sigmorphon eng fra hun rus spa
uv run python -m tokenisers.data --preset smoke --offline  # synthetic offline corpus

# 4. Run benchmarks
uv run python -m tokenisers.benchmarks.tokka               # Tokka-Bench (test split)
uv run python -m tokenisers.benchmarks.intrinsic            # Rényi, morph F1, throughput
uv run python -m tokenisers.benchmarks.colab --gpu T4       # Proxy-LM BPB (Colab GPU)
uv run python scripts/gen_readme_tables.py                  # Update tables below
```

**Data access.** The code domain reads the gated `bigcode/starcoderdata` (needs an HF token via `HF_TOKEN` or `hf auth login`). Without it, `codeparrot/github-code-clean` is used as fallback (recorded in `meta.dataset.fallbacks`).

**Colab.** Needs `google-colab-cli` >= 0.7.5. The benchmark uploads jobs and polls them detached; `--resume` continues after drops.

**Memory.** The CLI checks available memory before training (~4.2 GiB for 50 MiB with `super_bpe`). Use `--skip-memory-check` inside a `systemd-run -p MemoryMax=` scope if needed.

---

## Benchmark Results

Tables generated from `benchmarks/results/*/` by `scripts/gen_readme_tables.py`. Default dataset: 52.5 MB train (FineWeb-Edu 50%, FineWeb-2 30%, StarCoder 20%), held-out val/test 3.1 MB each. Tokenizer training ran locally (Intel Core Ultra 7 155H, 15 GB RAM); proxy LMs on Colab GPUs; the GPT-2 swap on Colab A100. Each section's generated header lists exact arguments, data sizes, backend and commit.

### 1. Tokka-Bench (natural languages + code, held-out test split)
<!-- BEGIN GENERATED: tokka -->
- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T12:59:35Z
- arguments: (defaults)
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.432 | 3.599 | 2.606 | 53.03 | 15867 |
| `bpe_frequency` | 16384 | 2.787 | 2.643 | 3.237 | 52.27 | 15735 |
| `length_penalized` | 16384 | 2.784 | 2.640 | 3.239 | 52.15 | 15808 |
| `wp_bpe_ppig` | 16384 | 2.759 | 2.608 | 3.300 | 52.56 | 15319 |
| `wp_bpe_pig` | 16384 | 2.752 | 2.599 | 3.311 | 52.98 | 15151 |
| `unigram_lm` | 16384 | 2.710 | 2.588 | 3.308 | 53.11 | 15834 |
| `wp_bpe_npmi` | 16384 | 2.682 | 2.530 | 3.435 | 53.94 | 14635 |
| `morpheme_bpe` | 16384 | 2.633 | 2.529 | 3.395 | 54.94 | 15089 |
| `mdl_bits` | 16384 | 2.439 | 2.370 | 3.991 | 53.56 | 14819 |
| `wp_bpe_scp` | 16384 | 1.995 | 1.641 | 5.184 | 67.43 | 6914 |
| `wordpiece_pmi` | 16384 | 1.026 | 1.017 | 8.900 | 83.23 | 1050 |
| `wp_bpe_ft` | 16384 | 1.026 | 1.017 | 8.900 | 83.23 | 990 |
| `wp_bpe_base` | 16384 | 1.009 | 1.007 | 9.005 | 83.55 | 282 |

Held-out text per language:

| Language | Docs | Bytes | In macro |
| :--- | ---: | ---: | :---: |
| en | 316 | 1,459,019 | yes |
| de | 38 | 92,973 | yes |
| fr | 20 | 88,540 | yes |
| es | 34 | 87,428 | yes |
| ru | 18 | 97,904 | yes |
| zh | 23 | 132,462 | yes |
| ja | 26 | 90,374 | yes |
| ar | 19 | 93,303 | yes |
| hi | 21 | 116,166 | yes |
| hu | 23 | 93,410 | yes |
| fi | 39 | 88,122 | yes |
| python | 18 | 113,050 | yes |
| javascript | 15 | 118,916 | yes |
| java | 24 | 115,673 | yes |
| cpp | 8 | 128,232 | yes |
| go | 11 | 104,232 | yes |
| rust | 13 | 123,740 | yes |
<!-- END GENERATED: tokka -->

### 2. Extended Intrinsic Evaluation
<!-- BEGIN GENERATED: intrinsic -->
- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:09:07Z
- arguments: `--skip-memory-check`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 16384 | 3.600 | 2.101 | 0.860 | 0.453 | 0.277 | 0.780 | 8.14 | 81.10 |
| `wp_bpe_base` | 16384 | 1.007 | 6.737 | 0.399 | 0.296 | 0.206 | 0.245 | 39.21 | 57.49 |
| `wp_bpe_ft` | 16384 | 1.017 | 6.671 | 0.398 | 0.294 | 0.206 | 0.245 | 38.00 | 57.48 |
| `wordpiece_pmi` | 16384 | 1.017 | 6.670 | 0.398 | 0.294 | 0.206 | 0.245 | 43.89 | 67.22 |
| `wp_bpe_scp` | 16384 | 1.641 | 3.927 | 0.497 | 0.284 | 0.248 | 0.349 | 23.31 | 51.23 |
| `morpheme_bpe` | 16384 | 2.529 | 2.203 | 0.645 | 0.214 | 0.420 | 0.737 | 17.52 | 63.98 |
| `wp_bpe_npmi` | 16384 | 2.531 | 2.175 | 0.619 | 0.208 | 0.306 | 0.774 | 18.04 | 65.42 |
| `wp_bpe_pig` | 16384 | 2.600 | 2.112 | 0.635 | 0.208 | 0.295 | 0.789 | 18.84 | 69.27 |
| `wp_bpe_ppig` | 16384 | 2.609 | 2.098 | 0.638 | 0.207 | 0.294 | 0.793 | 18.55 | 70.89 |
| `length_penalized` | 16384 | 2.641 | 2.069 | 0.659 | 0.207 | 0.273 | 0.780 | 15.06 | 59.51 |
| `bpe_frequency` | 16384 | 2.643 | 2.066 | 0.656 | 0.206 | 0.279 | 0.789 | 17.64 | 68.85 |
| `unigram_lm` | 16384 | 2.588 | 2.108 | 0.643 | 0.206 | 0.322 | 0.773 | 12.23 | 69.62 |
| `mdl_bits` | 16384 | 2.371 | 2.147 | 0.583 | 0.180 | 0.311 | 0.782 | 16.42 | 57.87 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.304 | 0.152 | 0.492 | 0.171 | 0.267 | 0.375 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.260 | 0.179 | 0.378 | 0.185 | 0.237 | 0.306 |
| `morpheme_bpe` | 0.483 | 0.334 | 0.514 | 0.404 | 0.363 | 0.476 |
| `wp_bpe_npmi` | 0.361 | 0.201 | 0.505 | 0.177 | 0.285 | 0.405 |
| `wp_bpe_pig` | 0.340 | 0.181 | 0.517 | 0.168 | 0.267 | 0.400 |
| `wp_bpe_ppig` | 0.340 | 0.178 | 0.514 | 0.171 | 0.267 | 0.399 |
| `length_penalized` | 0.300 | 0.147 | 0.482 | 0.171 | 0.265 | 0.369 |
| `bpe_frequency` | 0.306 | 0.153 | 0.493 | 0.172 | 0.270 | 0.377 |
| `unigram_lm` | 0.400 | 0.235 | 0.508 | 0.200 | 0.267 | 0.420 |
| `mdl_bits` | 0.391 | 0.224 | 0.490 | 0.169 | 0.283 | 0.407 |
<!-- END GENERATED: intrinsic -->

### 3. Extrinsic Proxy-LM Validation Bits per Byte
<!-- BEGIN GENERATED: colab -->
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
<!-- END GENERATED: colab -->

#### Big proxy LM (Colab L4)
<!-- BEGIN GENERATED: colab_big -->
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
<!-- END GENERATED: colab_big -->

##### Small vs big proxy LM
<!-- BEGIN GENERATED: colab_compare -->
- the 11 tokenizers both proxy-LM tiers trained; small LM: d_model=128, n_layers=2, 10,000,000-byte budget, seeds [0, 1, 2], `Tesla T4`; big LM: d_model=512, n_layers=8, 100,000,000-byte budget, seeds [0, 1, 2], `NVIDIA L4`
- rank: by mean macro val BPB (1 = best); Δ: macro val BPB minus `bpe_frequency`'s in the same tier (negative = better) ± its standard error √(sd²/n + sd_ref²/n) over the n seeds (sample sd). The seeds vary only the LM's initialization (same LM text, same order), so the standard errors measure initialization noise on this one text sample, not data-sampling variance
- small LM: 4 tokenizers beat `bpe_frequency`, by 0.006–0.038 BPB, i.e. 0.6–2.3 standard errors
- big LM: 4 tokenizers beat `bpe_frequency`, by 0.012–0.027 BPB, i.e. 6.4–23.8 standard errors

| Tokenizer | Small LM rank | Small LM Δ BPB | Big LM rank | Big LM Δ BPB |
| :--- | ---: | ---: | ---: | ---: |
| `wp_bpe_ppig` | 2 | -0.018 ± 0.009 | 1 | -0.027 ± 0.003 |
| `wp_bpe_pig` | 1 | -0.038 ± 0.017 | 2 | -0.022 ± 0.001 |
| `wp_bpe_npmi` | 4 | -0.006 ± 0.010 | 3 | -0.018 ± 0.002 |
| `super_bpe` | 3 | -0.008 ± 0.009 | 4 | -0.012 ± 0.002 |
| `bpe_frequency` | 5 | reference | 5 | reference |
| `morpheme_bpe` | 9 | +0.076 ± 0.009 | 6 | +0.002 ± 0.003 |
| `unigram_lm` | 8 | +0.059 ± 0.012 | 7 | +0.002 ± 0.002 |
| `bpe_dropout_p0.1` | 7 | +0.027 ± 0.008 | 8 | +0.008 ± 0.004 |
| `length_penalized` | 6 | +0.017 ± 0.008 | 9 | +0.017 ± 0.002 |
| `mdl_bits` | 10 | +0.197 ± 0.010 | 10 | +0.027 ± 0.003 |
| `wp_bpe_scp` | 11 | +0.632 ± 0.021 | 11 | +0.444 ± 0.009 |
<!-- END GENERATED: colab_compare -->

With the small LM, the top five are within about two standard errors of each other. With the big LM, `wp_bpe_ppig`, `wp_bpe_pig`, `wp_bpe_npmi` and `super_bpe` beat `bpe_frequency` by several standard errors, though the three wp_bpe variants' mutual order is not resolved.

<details><summary>Big proxy LM: how to run</summary>

`--preset big`: 8 layers, d_model 512, 8 heads, FFN 2048, seq 512 (~25M non-embedding params). AdamW lr 6e-4, wd 0.1, 2% warmup + cosine, 100 MB byte budget, seeds 0/1/2, default GPU L4, `--concurrency 1`. The 100 MB budget draws from the train split plus an LM text pool (document-disjoint from val/test).

```bash
uv run python -m tokenisers.benchmarks.colab --preset big --prepare-only --skip-memory-check
uv run python -m tokenisers.benchmarks.colab --preset big --resume
```

Use `--resume` to continue after interruptions. `--gpu none` for CPU smoke tests.

</details>

### 4. Larger Vocabularies (V = 32,768 and 65,536)

Same data as sections 1-2; only `--vocab-size` differs. No proxy LM at these sizes.

<details><summary>Tokka-Bench, V = 32,768</summary>

<!-- BEGIN GENERATED: tokka_v32768 -->
- requested V = 32768; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:02:20Z
- arguments: `--vocab-size 32768 --output-dir benchmarks/results/v32768/tokka`
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 3.891 | 4.062 | 2.306 | 48.11 | 29931 |
| `bpe_frequency` | 32768 | 3.047 | 2.838 | 2.980 | 47.34 | 29164 |
| `length_penalized` | 32768 | 3.040 | 2.836 | 2.985 | 47.27 | 29291 |
| `wp_bpe_ppig` | 32768 | 3.017 | 2.804 | 3.032 | 47.79 | 27605 |
| `wp_bpe_pig` | 32768 | 3.005 | 2.790 | 3.049 | 48.17 | 27032 |
| `wp_bpe_npmi` | 32768 | 2.932 | 2.719 | 3.159 | 49.41 | 25491 |
| `morpheme_bpe` | 32768 | 2.925 | 2.747 | 3.088 | 49.34 | 26359 |
| `unigram_lm` | 32768 | 2.921 | 2.752 | 3.089 | 48.82 | 28094 |
| `mdl_bits` | 32768 | 2.662 | 2.535 | 3.717 | 48.78 | 26396 |
| `wp_bpe_scp` | 32768 | 2.256 | 1.884 | 4.548 | 63.70 | 10181 |
| `wordpiece_pmi` | 32768 | 1.032 | 1.020 | 8.889 | 83.13 | 1784 |
| `wp_bpe_ft` | 32768 | 1.031 | 1.020 | 8.890 | 83.15 | 1693 |
| `wp_bpe_base` | 32768 | 1.015 | 1.011 | 8.949 | 83.53 | 383 |

Held-out text per language:

| Language | Docs | Bytes | In macro |
| :--- | ---: | ---: | :---: |
| en | 316 | 1,459,019 | yes |
| de | 38 | 92,973 | yes |
| fr | 20 | 88,540 | yes |
| es | 34 | 87,428 | yes |
| ru | 18 | 97,904 | yes |
| zh | 23 | 132,462 | yes |
| ja | 26 | 90,374 | yes |
| ar | 19 | 93,303 | yes |
| hi | 21 | 116,166 | yes |
| hu | 23 | 93,410 | yes |
| fi | 39 | 88,122 | yes |
| python | 18 | 113,050 | yes |
| javascript | 15 | 118,916 | yes |
| java | 24 | 115,673 | yes |
| cpp | 8 | 128,232 | yes |
| go | 11 | 104,232 | yes |
| rust | 13 | 123,740 | yes |
<!-- END GENERATED: tokka_v32768 -->

</details>

<details><summary>Extended intrinsic evaluation, V = 32,768</summary>

<!-- BEGIN GENERATED: intrinsic_v32768 -->
- requested V = 32768; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:12:14Z
- arguments: `--skip-memory-check --vocab-size 32768 --output-dir benchmarks/results/v32768/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 32768 | 4.062 | 1.886 | 0.848 | 0.427 | 0.283 | 0.828 | 7.41 | 84.08 |
| `wp_bpe_base` | 32768 | 1.011 | 6.723 | 0.372 | 0.276 | 0.206 | 0.245 | 42.65 | 63.01 |
| `wp_bpe_ft` | 32768 | 1.020 | 6.652 | 0.371 | 0.274 | 0.206 | 0.245 | 38.84 | 57.86 |
| `wordpiece_pmi` | 32768 | 1.020 | 6.650 | 0.371 | 0.274 | 0.206 | 0.245 | 43.03 | 64.32 |
| `wp_bpe_scp` | 32768 | 1.884 | 3.296 | 0.497 | 0.245 | 0.260 | 0.436 | 17.89 | 45.19 |
| `morpheme_bpe` | 32768 | 2.748 | 1.950 | 0.613 | 0.186 | 0.438 | 0.805 | 14.24 | 56.51 |
| `wp_bpe_npmi` | 32768 | 2.720 | 1.958 | 0.587 | 0.184 | 0.316 | 0.816 | 16.04 | 62.68 |
| `wp_bpe_pig` | 32768 | 2.791 | 1.900 | 0.602 | 0.183 | 0.302 | 0.825 | 15.06 | 62.67 |
| `unigram_lm` | 32768 | 2.753 | 1.921 | 0.606 | 0.182 | 0.327 | 0.823 | 10.41 | 67.15 |
| `wp_bpe_ppig` | 32768 | 2.804 | 1.885 | 0.605 | 0.182 | 0.299 | 0.835 | 15.67 | 63.82 |
| `length_penalized` | 32768 | 2.836 | 1.858 | 0.623 | 0.181 | 0.278 | 0.831 | 16.95 | 69.13 |
| `bpe_frequency` | 32768 | 2.838 | 1.857 | 0.620 | 0.181 | 0.282 | 0.833 | 14.13 | 60.01 |
| `mdl_bits` | 32768 | 2.536 | 1.926 | 0.550 | 0.158 | 0.321 | 0.825 | 17.28 | 63.21 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.313 | 0.158 | 0.507 | 0.170 | 0.266 | 0.386 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.288 | 0.193 | 0.391 | 0.175 | 0.255 | 0.323 |
| `morpheme_bpe` | 0.492 | 0.346 | 0.539 | 0.433 | 0.379 | 0.495 |
| `wp_bpe_npmi` | 0.379 | 0.215 | 0.525 | 0.174 | 0.289 | 0.420 |
| `wp_bpe_pig` | 0.350 | 0.190 | 0.535 | 0.165 | 0.271 | 0.413 |
| `unigram_lm` | 0.408 | 0.238 | 0.525 | 0.213 | 0.250 | 0.430 |
| `wp_bpe_ppig` | 0.344 | 0.180 | 0.537 | 0.168 | 0.265 | 0.411 |
| `length_penalized` | 0.308 | 0.155 | 0.501 | 0.169 | 0.255 | 0.380 |
| `bpe_frequency` | 0.314 | 0.158 | 0.510 | 0.168 | 0.259 | 0.387 |
| `mdl_bits` | 0.402 | 0.232 | 0.507 | 0.167 | 0.297 | 0.420 |
<!-- END GENERATED: intrinsic_v32768 -->

</details>

<details><summary>Tokka-Bench, V = 65,536</summary>

<!-- BEGIN GENERATED: tokka_v65536 -->
- requested V = 65536; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:06:59Z
- arguments: `--skip-memory-check --vocab-size 65536 --output-dir benchmarks/results/v65536/tokka`
- macro = unweighted mean over the 17 languages with >= 3 held-out documents and >= 10,000 bytes (en, de, fr, es, ru, zh, ja, ar, hi, hu, fi, python, javascript, java, cpp, go, rust); micro = totals summed over all 17 languages (token-weighted); unique tokens = union over languages

| Tokenizer | \|V\| | Macro bytes/token (↑) | Micro bytes/token (↑) | Macro fertility (↓) | Macro continued-token % (↓) | Unique tokens used |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.369 | 4.537 | 2.058 | 43.42 | 50752 |
| `bpe_frequency` | 65536 | 3.291 | 3.007 | 2.777 | 42.71 | 47483 |
| `length_penalized` | 65536 | 3.285 | 3.006 | 2.780 | 42.81 | 48001 |
| `wp_bpe_ppig` | 65536 | 3.266 | 2.975 | 2.818 | 43.26 | 44082 |
| `wp_bpe_pig` | 65536 | 3.258 | 2.963 | 2.828 | 43.62 | 42903 |
| `morpheme_bpe` | 65536 | 3.228 | 2.956 | 2.826 | 43.91 | 43824 |
| `wp_bpe_npmi` | 65536 | 3.170 | 2.887 | 2.938 | 44.94 | 39469 |
| `unigram_lm` | 65536 | 3.122 | 2.897 | 2.912 | 44.55 | 42909 |
| `mdl_bits` | 65536 | 2.882 | 2.686 | 3.485 | 43.87 | 42098 |
| `wp_bpe_scp` | 65536 | 2.652 | 2.253 | 3.805 | 57.18 | 17737 |
| `wordpiece_pmi` | 65536 | 1.043 | 1.025 | 8.865 | 83.06 | 3609 |
| `wp_bpe_ft` | 65536 | 1.037 | 1.022 | 8.874 | 83.08 | 2649 |
| `wp_bpe_base` | 65536 | 1.015 | 1.011 | 8.948 | 83.53 | 439 |

Held-out text per language:

| Language | Docs | Bytes | In macro |
| :--- | ---: | ---: | :---: |
| en | 316 | 1,459,019 | yes |
| de | 38 | 92,973 | yes |
| fr | 20 | 88,540 | yes |
| es | 34 | 87,428 | yes |
| ru | 18 | 97,904 | yes |
| zh | 23 | 132,462 | yes |
| ja | 26 | 90,374 | yes |
| ar | 19 | 93,303 | yes |
| hi | 21 | 116,166 | yes |
| hu | 23 | 93,410 | yes |
| fi | 39 | 88,122 | yes |
| python | 18 | 113,050 | yes |
| javascript | 15 | 118,916 | yes |
| java | 24 | 115,673 | yes |
| cpp | 8 | 128,232 | yes |
| go | 11 | 104,232 | yes |
| rust | 13 | 123,740 | yes |
<!-- END GENERATED: tokka_v65536 -->

</details>

<details><summary>Extended intrinsic evaluation, V = 65,536</summary>

<!-- BEGIN GENERATED: intrinsic_v65536 -->
- requested V = 65536; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out test split (3.1 MB); commit `603a983557`; generated 2026-10-30T13:35:36Z
- arguments: `--skip-memory-check --vocab-size 65536 --output-dir benchmarks/results/v65536/intrinsic`
- morph F1: per-word boundary F1 on 126622 gold words (SIGMORPHON 2022 test split, surface entries (eng, fra, hun, rus, spa)), averaged per language, then over the 5 languages unweighted (per language below); identifier F1: 2000 identifiers mined from held-out code; throughput: median of 5 passes over 3.1 MB (666 documents, one encode call each) on Intel(R) Core(TM) Ultra 7 155H

| Tokenizer | \|V\| | Bytes/token (↑) | Fertility (↓) | Rényi E1 | Rényi E2.5 | Morph F1 (↑) | Ident F1 (↑) | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 65536 | 4.537 | 1.718 | 0.832 | 0.405 | 0.283 | 0.847 | 5.26 | 58.37 |
| `wp_bpe_base` | 65536 | 1.011 | 6.722 | 0.349 | 0.258 | 0.206 | 0.245 | 27.40 | 40.99 |
| `wp_bpe_ft` | 65536 | 1.022 | 6.632 | 0.348 | 0.257 | 0.206 | 0.245 | 30.80 | 43.40 |
| `wordpiece_pmi` | 65536 | 1.025 | 6.612 | 0.348 | 0.256 | 0.206 | 0.245 | 29.38 | 43.09 |
| `wp_bpe_scp` | 65536 | 2.254 | 2.593 | 0.501 | 0.204 | 0.278 | 0.605 | 15.05 | 43.77 |
| `wp_bpe_npmi` | 65536 | 2.888 | 1.789 | 0.556 | 0.164 | 0.320 | 0.840 | 11.78 | 45.31 |
| `morpheme_bpe` | 65536 | 2.957 | 1.742 | 0.580 | 0.164 | 0.445 | 0.839 | 11.96 | 47.62 |
| `wp_bpe_pig` | 65536 | 2.964 | 1.732 | 0.570 | 0.163 | 0.303 | 0.850 | 10.96 | 42.68 |
| `unigram_lm` | 65536 | 2.898 | 1.770 | 0.570 | 0.162 | 0.333 | 0.849 | 6.65 | 48.68 |
| `wp_bpe_ppig` | 65536 | 2.976 | 1.720 | 0.572 | 0.162 | 0.303 | 0.857 | 11.58 | 47.66 |
| `length_penalized` | 65536 | 3.007 | 1.696 | 0.587 | 0.161 | 0.282 | 0.848 | 12.28 | 50.72 |
| `bpe_frequency` | 65536 | 3.008 | 1.696 | 0.585 | 0.161 | 0.285 | 0.849 | 11.51 | 46.70 |
| `mdl_bits` | 65536 | 2.687 | 1.746 | 0.519 | 0.139 | 0.322 | 0.854 | 11.67 | 44.10 |

Morph F1 by language:

| Tokenizer | eng (40,455 words) | fra (6,299 words) | hun (61,213 words) | rus (8,526 words) | spa (10,129 words) | Word-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `super_bpe` | 0.319 | 0.169 | 0.516 | 0.167 | 0.247 | 0.391 |
| `wp_bpe_base` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_ft` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wordpiece_pmi` | 0.212 | 0.187 | 0.312 | 0.126 | 0.193 | 0.252 |
| `wp_bpe_scp` | 0.324 | 0.200 | 0.436 | 0.178 | 0.252 | 0.356 |
| `wp_bpe_npmi` | 0.385 | 0.219 | 0.540 | 0.176 | 0.279 | 0.429 |
| `morpheme_bpe` | 0.488 | 0.336 | 0.557 | 0.448 | 0.397 | 0.504 |
| `wp_bpe_pig` | 0.350 | 0.189 | 0.546 | 0.165 | 0.266 | 0.418 |
| `unigram_lm` | 0.420 | 0.251 | 0.537 | 0.228 | 0.226 | 0.440 |
| `wp_bpe_ppig` | 0.350 | 0.189 | 0.545 | 0.169 | 0.264 | 0.417 |
| `length_penalized` | 0.317 | 0.166 | 0.507 | 0.170 | 0.251 | 0.386 |
| `bpe_frequency` | 0.320 | 0.171 | 0.518 | 0.167 | 0.248 | 0.392 |
| `mdl_bits` | 0.402 | 0.236 | 0.523 | 0.169 | 0.281 | 0.427 |
<!-- END GENERATED: intrinsic_v65536 -->

</details>

### 5. Throughput Re-measured (Colab CPU and a pinned laptop core)

Same tokenizers as sections 2 and 4, re-timed on a Colab CPU runtime and on one pinned laptop core (`taskset -c 3`). Two full passes measure run-to-run spread. Compare MB/s within one table, not across tables.

```bash
uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp --prepare-only
uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp --colab   # Colab CPU
taskset -c 3 uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp  # pinned laptop core
```

**Colab CPU runtime:**

<!-- BEGIN GENERATED: throughput_colab -->
- Colab CPU runtime (no accelerator requested), session `tokenisers-throughput-d8a4e8da`; 2 vCPUs (1 core(s) x 2 threads, KVM VM): AMD EPYC 7B12, 12.7 GiB RAM, Python 3.13.18, glibc 2.39; extension built on the VM from commit `893cb9b516` (rustc 1.99.1 (58cf8f9c3 2026-10-13), 66.4 s); commit `893cb9b516`; generated 2026-10-30T15:11:02Z
- 39 tokenizers: the suite's variants at V = 16384, 32768, 65536, retrained from the real train split (52.5 MB, dataset `hf-fb5b8df7af00b10f`), saved, reloaded on the measuring machine and checked there: identical token ids (sha256) on every throughput document
- 666 documents, 3.1 MB, one encode/decode call per document; each cell is the median of 5 timed passes after 1 warm-up (`IntrinsicEvaluator.benchmark_throughput`), measured in 2 full runs over all 39 tokenizers: run 1 / run 2
- run-to-run spread (max/min over the runs, per tokenizer): encode median 1.03x, max 1.50x; decode median 1.02x, max 1.52x
- 1-min load average (the benchmark's own process adds about 1) and MemAvailable before -> after each run: run 1: 1.37 -> 1.26, 11.3 GiB -> 11.3 GiB (66 s); run 2: 1.26 -> 1.25, 11.3 GiB -> 11.3 GiB (66 s)

| Tokenizer | \|V\| | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: |
| `bpe_frequency` | 16384 | 14.39 / 14.15 | 65.90 / 66.52 |
| `wordpiece_pmi` | 16384 | 35.48 / 36.88 | 48.04 / 47.73 |
| `wp_bpe_base` | 16384 | 38.69 / 39.08 | 46.82 / 48.48 |
| `wp_bpe_ft` | 16384 | 24.55 / 36.79 | 30.62 / 46.61 |
| `wp_bpe_pig` | 16384 | 14.97 / 14.55 | 66.32 / 63.00 |
| `wp_bpe_scp` | 16384 | 20.80 / 20.34 | 47.43 / 47.23 |
| `wp_bpe_npmi` | 16384 | 15.63 / 11.33 | 64.84 / 46.72 |
| `wp_bpe_ppig` | 16384 | 14.66 / 14.97 | 65.57 / 66.12 |
| `mdl_bits` | 16384 | 15.05 / 14.21 | 62.72 / 60.63 |
| `length_penalized` | 16384 | 14.16 / 14.47 | 67.87 / 68.87 |
| `super_bpe` | 16384 | 5.44 / 7.30 | 92.39 / 90.95 |
| `morpheme_bpe` | 16384 | 14.09 / 14.58 | 62.14 / 62.77 |
| `unigram_lm` | 16384 | 9.28 / 6.54 | 64.88 / 47.82 |
| `bpe_frequency` | 32768 | 12.78 / 12.97 | 65.32 / 59.56 |
| `wordpiece_pmi` | 32768 | 36.02 / 36.15 | 47.24 / 46.26 |
| `wp_bpe_base` | 32768 | 38.13 / 37.78 | 47.07 / 47.52 |
| `wp_bpe_ft` | 32768 | 36.15 / 37.04 | 47.72 / 46.54 |
| `wp_bpe_pig` | 32768 | 13.56 / 13.45 | 66.57 / 63.72 |
| `wp_bpe_scp` | 32768 | 12.92 / 18.14 | 38.80 / 50.04 |
| `wp_bpe_npmi` | 32768 | 14.25 / 14.01 | 63.76 / 65.25 |
| `wp_bpe_ppig` | 32768 | 13.48 / 13.48 | 64.17 / 65.21 |
| `mdl_bits` | 32768 | 13.99 / 14.14 | 63.58 / 64.44 |
| `length_penalized` | 32768 | 13.67 / 12.99 | 70.47 / 62.74 |
| `super_bpe` | 32768 | 6.24 / 6.41 | 93.52 / 96.24 |
| `morpheme_bpe` | 32768 | 10.06 / 13.54 | 49.54 / 66.11 |
| `unigram_lm` | 32768 | 8.17 / 8.17 | 64.73 / 63.98 |
| `bpe_frequency` | 65536 | 11.43 / 8.80 | 66.36 / 45.52 |
| `wordpiece_pmi` | 65536 | 34.74 / 34.77 | 47.86 / 46.00 |
| `wp_bpe_base` | 65536 | 36.69 / 38.01 | 46.20 / 47.92 |
| `wp_bpe_ft` | 65536 | 35.50 / 35.92 | 46.47 / 46.12 |
| `wp_bpe_pig` | 65536 | 12.38 / 12.11 | 63.29 / 64.46 |
| `wp_bpe_scp` | 65536 | 16.16 / 15.60 | 57.87 / 57.56 |
| `wp_bpe_npmi` | 65536 | 9.66 / 13.08 | 63.66 / 63.99 |
| `wp_bpe_ppig` | 65536 | 11.75 / 11.57 | 62.24 / 61.81 |
| `mdl_bits` | 65536 | 12.53 / 9.10 | 61.22 / 45.02 |
| `length_penalized` | 65536 | 12.00 / 11.72 | 64.44 / 63.10 |
| `super_bpe` | 65536 | 5.46 / 5.25 | 85.93 / 87.81 |
| `morpheme_bpe` | 65536 | 9.08 / 11.69 | 46.09 / 62.81 |
| `unigram_lm` | 65536 | 7.06 / 6.73 | 66.36 / 65.75 |
<!-- END GENERATED: throughput_colab -->

**Laptop, pinned to one performance core:**

<!-- BEGIN GENERATED: throughput_local -->
- local, pinned to CPU 3 (max 4800 MHz, governor powersave, SMT sibling(s) 3-4): Intel(R) Core(TM) Ultra 7 155H, 15.1 GiB RAM, Python 3.13.17, glibc 2.44; commit `893cb9b516`; generated 2026-10-30T15:06:11Z
- 39 tokenizers: the suite's variants at V = 16384, 32768, 65536, retrained from the real train split (52.5 MB, dataset `hf-fb5b8df7af00b10f`), saved, reloaded on the measuring machine and checked there: identical token ids (sha256) on every throughput document
- 666 documents, 3.1 MB, one encode/decode call per document; each cell is the median of 5 timed passes after 1 warm-up (`IntrinsicEvaluator.benchmark_throughput`), measured in 2 full runs over all 39 tokenizers: run 1 / run 2
- run-to-run spread (max/min over the runs, per tokenizer): encode median 1.02x, max 1.15x; decode median 1.02x, max 1.13x
- 1-min load average (the benchmark's own process adds about 1) and MemAvailable before -> after each run: run 1: 1.86 -> 1.57, 6.0 GiB -> 5.9 GiB (53 s); run 2: 1.57 -> 1.56, 5.9 GiB -> 5.8 GiB (53 s)

| Tokenizer | \|V\| | Encode MB/s | Decode MB/s |
| :--- | ---: | ---: | ---: |
| `bpe_frequency` | 16384 | 16.08 / 16.18 | 75.57 / 78.14 |
| `wordpiece_pmi` | 16384 | 41.71 / 43.01 | 63.88 / 65.44 |
| `wp_bpe_base` | 16384 | 43.28 / 44.77 | 64.86 / 65.34 |
| `wp_bpe_ft` | 16384 | 41.31 / 41.12 | 63.51 / 63.65 |
| `wp_bpe_pig` | 16384 | 16.78 / 16.33 | 73.35 / 69.56 |
| `wp_bpe_scp` | 16384 | 21.60 / 22.21 | 48.54 / 49.26 |
| `wp_bpe_npmi` | 16384 | 16.30 / 17.48 | 67.04 / 72.23 |
| `wp_bpe_ppig` | 16384 | 16.55 / 16.82 | 75.11 / 73.96 |
| `mdl_bits` | 16384 | 17.02 / 16.70 | 69.64 / 67.06 |
| `length_penalized` | 16384 | 15.98 / 16.16 | 75.95 / 75.41 |
| `super_bpe` | 16384 | 8.55 / 8.56 | 108.64 / 102.47 |
| `morpheme_bpe` | 16384 | 16.44 / 16.30 | 69.63 / 69.21 |
| `unigram_lm` | 16384 | 10.88 / 11.29 | 67.48 / 70.35 |
| `bpe_frequency` | 32768 | 15.52 / 15.71 | 78.63 / 77.56 |
| `wordpiece_pmi` | 32768 | 42.25 / 43.24 | 64.62 / 65.47 |
| `wp_bpe_base` | 32768 | 43.80 / 42.64 | 66.14 / 65.07 |
| `wp_bpe_ft` | 32768 | 42.25 / 41.89 | 64.73 / 65.12 |
| `wp_bpe_pig` | 32768 | 16.00 / 16.33 | 75.20 / 76.97 |
| `wp_bpe_scp` | 32768 | 19.82 / 20.07 | 51.93 / 52.44 |
| `wp_bpe_npmi` | 32768 | 16.47 / 16.56 | 74.23 / 72.79 |
| `wp_bpe_ppig` | 32768 | 15.77 / 15.75 | 75.78 / 75.06 |
| `mdl_bits` | 32768 | 15.67 / 14.70 | 70.51 / 64.90 |
| `length_penalized` | 32768 | 15.41 / 15.10 | 77.03 / 75.58 |
| `super_bpe` | 32768 | 7.69 / 7.81 | 108.73 / 113.49 |
| `morpheme_bpe` | 32768 | 15.58 / 15.94 | 75.68 / 77.21 |
| `unigram_lm` | 32768 | 9.79 / 10.00 | 70.34 / 70.32 |
| `bpe_frequency` | 65536 | 13.04 / 14.89 | 70.60 / 78.10 |
| `wordpiece_pmi` | 65536 | 38.56 / 41.05 | 61.80 / 64.37 |
| `wp_bpe_base` | 65536 | 42.22 / 39.35 | 63.49 / 59.32 |
| `wp_bpe_ft` | 65536 | 41.04 / 36.91 | 63.06 / 58.94 |
| `wp_bpe_pig` | 65536 | 15.40 / 15.41 | 77.58 / 75.13 |
| `wp_bpe_scp` | 65536 | 18.21 / 18.22 | 61.59 / 61.27 |
| `wp_bpe_npmi` | 65536 | 15.36 / 16.04 | 75.50 / 75.75 |
| `wp_bpe_ppig` | 65536 | 15.03 / 15.10 | 77.64 / 78.24 |
| `mdl_bits` | 65536 | 15.60 / 13.52 | 73.84 / 65.48 |
| `length_penalized` | 65536 | 14.57 / 12.87 | 76.68 / 70.16 |
| `super_bpe` | 65536 | 7.17 / 7.21 | 113.90 / 117.59 |
| `morpheme_bpe` | 65536 | 15.01 / 14.88 | 78.21 / 76.85 |
| `unigram_lm` | 65536 | 8.86 / 8.73 | 74.76 / 74.15 |
<!-- END GENERATED: throughput_local -->

### 6. Tokenizer Swap on llm.c's GPT-2 (124M) Recipe

Trains GPT-2 small (124M params, llm.c's recipe at `f1e2ace`) once per tokenizer on the speedrun's FineWeb shards (`kjj0/fineweb10B-gpt2`), 600 steps on a Colab A100. Hyperparameters identical across tokenizers (AdamW lr 6e-4, 524,288 tokens/step, bf16, `torch.compile`); only the vocabulary differs. Twelve tokenizers: GPT-2 (tiktoken), TokenMonster 50k/32k, and our 9 variants at V = 50,256 trained on the first 50 MB of FineWeb train documents.

```bash
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --prepare-only
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --gpu A100
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --gpu A100 --resume  # after a drop
```

<!-- BEGIN GENERATED: nanogpt_swap -->
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
<!-- END GENERATED: nanogpt_swap -->

**Space-prefix retrain.** The 8 whitespace-isolating variants retrained with `pretokenizer="space_prefix"` — same text, parameters and size, but a word carries the space before it. Separate job, same recipe and val documents.

<!-- BEGIN GENERATED: nanogpt_swap_sp -->
- space-prefix run: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, commit `2df09a7847`, generated 2026-11-01T10:53:38Z; whitespace run: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07
- same recipe, val documents and tokenizer training text as the whitespace run

| Variant | Val B/token whitespace | Val B/token space_prefix | Final BPB whitespace | Final BPB space_prefix | Change |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `bpe_frequency` | 2.750 | 4.831 | 2.0259 | 1.8000 | -0.2259 |
| `wp_bpe_pig` | 2.733 | 4.769 | 2.0077 | 1.7749 | -0.2328 |
| `wp_bpe_ppig` | 2.736 | 4.777 | 1.9561 | 1.7552 | -0.2009 |
| `wp_bpe_npmi` | 2.702 | 4.658 | 1.9835 | 1.7436 | -0.2399 |
| `unigram_lm` | 2.725 | 4.692 | 2.0239 | 1.7088 | -0.3151 |
| `length_penalized` | 2.749 | 4.822 | 2.0393 | 1.9008 | -0.1385 |
| `morpheme_bpe` | 2.718 | 4.710 | 1.9695 | 1.7742 | -0.1953 |
| `mdl_bits` | 2.718 | 4.689 | 1.9684 | 1.7067 | -0.2617 |
| `gpt2` (reference) | 4.436 | | 1.6851 | | |
<!-- END GENERATED: nanogpt_swap_sp -->

**Pre-split and data-scaling ablation.** Three variants with `pretokenizer="gpt2"` (50 MB) and three space-prefix variants trained on 300 MB instead of 50. Same recipe, same val documents.

<!-- BEGIN GENERATED: nanogpt_swap_ablation -->
- ablation run: NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, commit `950506beba`, generated 2026-11-01T16:28:54Z
- same recipe and val documents as the whitespace run
- training text: `mdl_bits_g2` 16,157 docs, 50.0 MB; `unigram_lm_g2` 16,157 docs, 50.0 MB; `wp_bpe_npmi_g2` 16,157 docs, 50.0 MB; `wp_bpe_pig_g2` 16,157 docs, 50.0 MB; `wp_bpe_ppig_g2` 16,157 docs, 50.0 MB; `mdl_bits_sp300` 97,402 docs, 300.0 MB; `unigram_lm_sp300` 97,402 docs, 300.0 MB; `wp_bpe_npmi_sp300` 97,402 docs, 300.0 MB; `wp_bpe_pig_sp300` 97,402 docs, 300.0 MB; `wp_bpe_ppig_sp300` 97,402 docs, 300.0 MB
- cells: final val BPB (val bytes per token)

| Variant | whitespace, 50 MB | space_prefix, 50 MB | gpt2, 50 MB | space_prefix, 300 MB |
| :--- | ---: | ---: | ---: | ---: |
| `mdl_bits` | **1.9684** (2.72) | **1.7067** (4.69) | **1.6910** (4.45) | **1.7226** (4.72) |
| `unigram_lm` | **2.0239** (2.73) | **1.7088** (4.69) | **1.6817** (4.42) | **1.7026** (4.70) |
| `wp_bpe_npmi` | **1.9835** (2.70) | **1.7436** (4.66) | **1.7093** (4.43) | **1.7207** (4.69) |
| `wp_bpe_pig` | **2.0077** (2.73) | **1.7749** (4.77) | **1.7029** (4.47) | **1.7644** (4.80) |
| `wp_bpe_ppig` | **1.9561** (2.74) | **1.7552** (4.78) | **1.6946** (4.47) | **1.7090** (4.81) |
| `gpt2` (reference) | **1.6851** (4.44) | | | |
<!-- END GENERATED: nanogpt_swap_ablation -->

### 7. Cross-lingual Tokenization Parity (FLORES+)

Encodes parallel FLORES+ devtest sentences (1,012 per language) and reports the **premium**: each language's token count divided by English's (Petrov et al. 2023). Languages: English, 10 in-mix FineWeb-2 languages, 7 out-of-mix. Two tokenizer tiers: intrinsic (13 variants at V = 16k/32k/64k, 30% multilingual mix) and GPT-2 swap (English-centric).

```bash
uv run --with tokenmonster==1.1.12 python -m tokenisers.benchmarks.flores --skip-memory-check
```

<!-- BEGIN GENERATED: flores -->
- FLORES+ `openlanguagedata/flores_plus` at `1147e2dc59f3` (CC BY-SA 4.0), devtest split only: 1,012 parallel sentences per language, each encoded on its own with no separator; never used for training
- in-mix: the 10 FineWeb-2 languages of the data pipeline's training mix (FineWeb-2 config -> FLORES+ file): deu_Latn, fra_Latn, spa_Latn, rus_Cyrl, cmn_Hani -> cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva, hun_Latn, fin_Latn; out-of-mix (in no tokenizer's training mix): kor_Hang, tha_Thai, tam_Taml, amh_Ethi, ell_Grek, heb_Hebr, ben_Beng
- intrinsic tier: 13 variants x V = 16,384 / 32,768 / 65,536, retrained as the committed intrinsic results (real dataset `hf-fb5b8df7af00b10f`, 52.5 MB train split); 39 of 39 reproduce the committed test-split bytes/token exactly
- swap tier: the GPT-2-swap tokenizers from the prepared jobs `nanogpt-swap-job`, `nanogpt-swap-job-sp`, `nanogpt-swap-pool-g2`, `nanogpt-swap-pool-sp300`: `gpt2` (tiktoken's `gpt2` encoding: GPT-2's own BPE, trained by OpenAI on WebText); `tm_englishcode_50256` (TokenMonster's prebuilt `englishcode-50256-consistent-nocapcode-v1` vocabulary, 50,256 tokens); `tm_englishcode_32000` (TokenMonster's prebuilt `englishcode-32000-consistent-nocapcode-v1` vocabulary, 32,000 tokens); 33 of ours, trained on English FineWeb text at 50,256 tokens
- premium = tokens(language) / tokens(English) on the same sentences; σ ln = standard deviation of ln(premium) over the group's languages; <= 1.5 = share of them with premium <= 1.5; byte fallback = share of tokens that are one byte >= 0x80, a piece of a multi-byte character
- commit `d9b38ad6e9`; generated 2026-11-01T12:21:34Z; per-language tokens, bytes/token, characters/token and one-byte shares of every tokenizer are in `flores_by_language.csv`

How to read: premium 1.0 is parity with English; 2.0 means the same content costs twice the tokens (so half the context and twice the price). Bytes or characters per token are not comparable across scripts: UTF-8 spends 1 byte on a Latin letter, 2 on Cyrillic, Greek, Hebrew or Arabic, 3 on Devanagari, Han, Kana, Hangul or Thai, so a script's bytes per token can rise while its text still costs more tokens. The premium counts tokens for the same parallel sentences, which is comparable.

Intrinsic tier (30% multilingual mix), sorted by mean in-mix premium:

| Tokenizer | \|V\| | EN tokens/sentence | In-mix mean | In-mix max | In-mix σ ln | In-mix <= 1.5 | Out-of-mix mean | Out-of-mix max | Out-of-mix σ ln | Out-of-mix <= 1.5 | Out-of-mix byte fallback |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp_bpe_scp@16384` | 16,384 | 86.3 | 1.04 | 1.35 hun_Latn | 0.43 | 100% | 2.75 | 4.81 tam_Taml | 0.41 | 14% | 77.8% |
| `wp_bpe_scp@32768` | 32,768 | 73.9 | 1.10 | 1.46 hun_Latn | 0.42 | 100% | 2.99 | 5.62 tam_Taml | 0.50 | 14% | 73.6% |
| `wp_bpe_scp@65536` | 65,536 | 60.8 | 1.15 | 1.51 spa_Latn | 0.40 | 80% | 3.59 | 6.83 tam_Taml | 0.52 | 14% | 72.9% |
| `wp_bpe_npmi@65536` | 65,536 | 47.8 | 1.16 | 1.39 spa_Latn | 0.25 | 100% | 4.93 | 8.70 tam_Taml | 0.43 | 0% | 74.8% |
| `mdl_bits@65536` | 65,536 | 47.2 | 1.17 | 1.42 hin_Deva | 0.25 | 100% | 4.98 | 8.81 tam_Taml | 0.44 | 0% | 74.4% |
| `wp_bpe_pig@65536` | 65,536 | 46.9 | 1.17 | 1.41 hin_Deva | 0.25 | 100% | 5.07 | 8.86 tam_Taml | 0.41 | 0% | 75.8% |
| `wp_bpe_ppig@65536` | 65,536 | 46.8 | 1.17 | 1.44 hin_Deva | 0.25 | 100% | 5.01 | 8.88 tam_Taml | 0.44 | 0% | 74.0% |
| `bpe_frequency@65536` | 65,536 | 46.3 | 1.19 | 1.46 hin_Deva | 0.23 | 100% | 5.22 | 8.97 tam_Taml | 0.41 | 0% | 77.7% |
| `morpheme_bpe@65536` | 65,536 | 47.1 | 1.19 | 1.47 hin_Deva | 0.23 | 100% | 5.20 | 8.82 tam_Taml | 0.38 | 0% | 79.4% |
| `length_penalized@65536` | 65,536 | 46.3 | 1.19 | 1.50 hin_Deva | 0.24 | 100% | 5.08 | 8.97 tam_Taml | 0.44 | 0% | 74.1% |
| `unigram_lm@65536` | 65,536 | 47.5 | 1.20 | 1.51 hin_Deva | 0.20 | 90% | 5.13 | 7.25 tam_Taml | 0.29 | 0% | 78.3% |
| `wp_bpe_npmi@32768` | 32,768 | 49.8 | 1.22 | 1.44 hin_Deva | 0.24 | 100% | 4.93 | 8.35 tam_Taml | 0.38 | 0% | 79.5% |
| `mdl_bits@32768` | 32,768 | 49.2 | 1.23 | 1.48 hin_Deva | 0.24 | 100% | 5.00 | 8.45 tam_Taml | 0.38 | 0% | 80.0% |
| `wp_bpe_ppig@32768` | 32,768 | 48.7 | 1.23 | 1.51 hin_Deva | 0.23 | 90% | 5.02 | 8.53 tam_Taml | 0.39 | 0% | 79.1% |
| `wp_bpe_pig@32768` | 32,768 | 48.8 | 1.24 | 1.48 hin_Deva | 0.23 | 100% | 5.16 | 8.52 tam_Taml | 0.36 | 0% | 82.6% |
| `unigram_lm@32768` | 32,768 | 49.2 | 1.25 | 1.59 hin_Deva | 0.18 | 90% | 5.37 | 7.78 tam_Taml | 0.32 | 0% | 87.7% |
| `bpe_frequency@32768` | 32,768 | 48.1 | 1.25 | 1.54 hin_Deva | 0.21 | 90% | 5.36 | 8.64 tam_Taml | 0.34 | 0% | 85.3% |
| `length_penalized@32768` | 32,768 | 48.2 | 1.25 | 1.58 hin_Deva | 0.22 | 90% | 5.11 | 8.62 tam_Taml | 0.38 | 0% | 79.6% |
| `morpheme_bpe@32768` | 32,768 | 49.4 | 1.26 | 1.59 hin_Deva | 0.19 | 90% | 5.60 | 8.41 tam_Taml | 0.34 | 0% | 92.8% |
| `wp_bpe_npmi@16384` | 16,384 | 52.4 | 1.29 | 1.51 hin_Deva | 0.23 | 90% | 5.24 | 7.93 tam_Taml | 0.36 | 0% | 92.2% |
| `mdl_bits@16384` | 16,384 | 52.0 | 1.29 | 1.55 hin_Deva | 0.22 | 90% | 5.33 | 8.00 tam_Taml | 0.34 | 0% | 92.8% |
| `wp_bpe_ppig@16384` | 16,384 | 51.4 | 1.30 | 1.58 hin_Deva | 0.22 | 80% | 5.34 | 8.09 tam_Taml | 0.36 | 0% | 91.7% |
| `wp_bpe_pig@16384` | 16,384 | 51.4 | 1.30 | 1.55 hin_Deva | 0.21 | 80% | 5.38 | 8.08 tam_Taml | 0.34 | 0% | 92.8% |
| `unigram_lm@16384` | 16,384 | 51.6 | 1.30 | 1.65 hin_Deva | 0.16 | 80% | 5.31 | 7.77 tam_Taml | 0.34 | 0% | 91.7% |
| `length_penalized@16384` | 16,384 | 50.8 | 1.31 | 1.67 hin_Deva | 0.20 | 80% | 5.42 | 8.18 tam_Taml | 0.35 | 0% | 92.3% |
| `bpe_frequency@16384` | 16,384 | 50.6 | 1.32 | 1.63 hin_Deva | 0.19 | 80% | 5.50 | 8.21 tam_Taml | 0.34 | 0% | 93.5% |
| `morpheme_bpe@16384` | 16,384 | 52.3 | 1.34 | 1.74 hin_Deva | 0.15 | 90% | 5.32 | 7.94 tam_Taml | 0.34 | 0% | 93.5% |
| `wordpiece_pmi@65536` | 65,536 | 130.3 | 1.37 | 2.44 hin_Deva | 0.35 | 70% | 2.00 | 3.20 tam_Taml | 0.49 | 29% | 88.8% |
| `wp_bpe_ft@65536` | 65,536 | 130.3 | 1.37 | 2.44 hin_Deva | 0.34 | 70% | 2.00 | 3.20 tam_Taml | 0.49 | 29% | 88.9% |
| `wordpiece_pmi@32768` | 32,768 | 130.3 | 1.38 | 2.47 hin_Deva | 0.33 | 70% | 2.04 | 3.20 tam_Taml | 0.48 | 29% | 89.8% |
| `wp_bpe_ft@32768` | 32,768 | 130.3 | 1.38 | 2.47 hin_Deva | 0.32 | 70% | 2.05 | 3.20 tam_Taml | 0.45 | 29% | 90.5% |
| `wordpiece_pmi@16384` | 16,384 | 130.3 | 1.39 | 2.47 hin_Deva | 0.31 | 70% | 2.06 | 3.20 tam_Taml | 0.45 | 29% | 90.5% |
| `wp_bpe_ft@16384` | 16,384 | 130.3 | 1.39 | 2.47 hin_Deva | 0.31 | 70% | 2.05 | 3.20 tam_Taml | 0.45 | 29% | 90.5% |
| `wp_bpe_base@65536` | 65,536 | 130.3 | 1.41 | 2.55 hin_Deva | 0.30 | 70% | 2.04 | 3.20 tam_Taml | 0.48 | 29% | 90.3% |
| `wp_bpe_base@32768` | 32,768 | 130.4 | 1.41 | 2.55 hin_Deva | 0.30 | 70% | 2.06 | 3.20 tam_Taml | 0.45 | 29% | 91.1% |
| `wp_bpe_base@16384` | 16,384 | 130.4 | 1.41 | 2.55 hin_Deva | 0.30 | 70% | 2.06 | 3.19 tam_Taml | 0.45 | 29% | 91.0% |
| `super_bpe@65536` | 65,536 | 25.1 | 1.68 | 2.03 rus_Cyrl | 0.12 | 20% | 9.62 | 16.45 tam_Taml | 0.41 | 0% | 79.0% |
| `super_bpe@32768` | 32,768 | 28.0 | 1.72 | 2.11 rus_Cyrl | 0.12 | 20% | 9.72 | 14.78 tam_Taml | 0.36 | 0% | 92.0% |
| `super_bpe@16384` | 16,384 | 31.5 | 1.75 | 2.19 rus_Cyrl | 0.13 | 10% | 8.80 | 13.13 tam_Taml | 0.35 | 0% | 93.9% |

GPT-2-swap tier (English-centric), sorted by mean in-mix premium:

| Tokenizer | \|V\| | EN tokens/sentence | In-mix mean | In-mix max | In-mix σ ln | In-mix <= 1.5 | Out-of-mix mean | Out-of-mix max | Out-of-mix σ ln | Out-of-mix <= 1.5 | Out-of-mix byte fallback |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp_bpe_npmi` | 50,256 | 46.1 | 2.08 | 3.01 hin_Deva | 0.19 | 0% | 4.24 | 5.52 ben_Beng | 0.29 | 0% | 60.8% |
| `mdl_bits` | 50,256 | 45.8 | 2.09 | 3.12 hin_Deva | 0.21 | 0% | 4.24 | 5.56 ben_Beng | 0.32 | 0% | 59.2% |
| `wp_bpe_ppig` | 50,256 | 45.5 | 2.11 | 3.13 hin_Deva | 0.22 | 0% | 4.25 | 5.59 ben_Beng | 0.33 | 0% | 56.6% |
| `wp_bpe_pig` | 50,256 | 45.5 | 2.19 | 3.36 hin_Deva | 0.24 | 0% | 4.42 | 5.60 ben_Beng | 0.27 | 0% | 63.0% |
| `length_penalized` | 50,256 | 45.4 | 2.21 | 3.50 hin_Deva | 0.28 | 0% | 4.44 | 5.76 tam_Taml | 0.32 | 0% | 58.5% |
| `morpheme_bpe` | 50,256 | 46.0 | 2.30 | 3.96 hin_Deva | 0.32 | 0% | 5.15 | 7.37 ben_Beng | 0.33 | 0% | 75.5% |
| `bpe_frequency` | 50,256 | 45.3 | 2.33 | 3.87 hin_Deva | 0.31 | 0% | 4.94 | 7.47 ben_Beng | 0.33 | 0% | 69.6% |
| `unigram_lm` | 50,256 | 45.8 | 2.43 | 4.46 hin_Deva | 0.38 | 0% | 5.20 | 6.95 ben_Beng | 0.26 | 0% | 75.6% |
| `mdl_bits_g2` | 50,256 | 27.0 | 3.16 | 4.88 hin_Deva | 0.31 | 0% | 7.16 | 9.68 tam_Taml | 0.36 | 0% | 59.0% |
| `wp_bpe_ppig_g2` | 50,256 | 26.8 | 3.20 | 4.97 hin_Deva | 0.32 | 0% | 7.21 | 9.74 tam_Taml | 0.36 | 0% | 58.1% |
| `wp_bpe_npmi_g2` | 50,256 | 27.1 | 3.21 | 4.85 hin_Deva | 0.30 | 0% | 7.13 | 9.63 tam_Taml | 0.36 | 0% | 59.4% |
| `length_penalized_g2` | 50,256 | 26.8 | 3.26 | 5.40 hin_Deva | 0.37 | 0% | 7.42 | 9.75 tam_Taml | 0.34 | 0% | 61.1% |
| `wp_bpe_npmi_sp300` | 50,256 | 26.0 | 3.30 | 5.62 hin_Deva | 0.32 | 0% | 7.60 | 12.41 tam_Taml | 0.30 | 0% | 60.9% |
| `wp_bpe_pig_g2` | 50,256 | 26.9 | 3.30 | 5.19 hin_Deva | 0.34 | 0% | 7.41 | 9.73 tam_Taml | 0.33 | 0% | 61.8% |
| `mdl_bits_sp` | 50,256 | 26.1 | 3.33 | 5.21 hin_Deva | 0.33 | 0% | 7.51 | 9.75 ben_Beng | 0.30 | 0% | 63.5% |
| `wp_bpe_npmi_sp` | 50,256 | 26.3 | 3.34 | 5.15 hin_Deva | 0.32 | 0% | 7.46 | 9.68 ben_Beng | 0.30 | 0% | 64.1% |
| `wp_bpe_ppig_sp300` | 50,256 | 25.4 | 3.35 | 5.85 hin_Deva | 0.34 | 0% | 7.91 | 12.70 tam_Taml | 0.32 | 0% | 58.5% |
| `mdl_bits_sp300` | 50,256 | 25.9 | 3.37 | 5.84 hin_Deva | 0.34 | 0% | 7.87 | 12.47 tam_Taml | 0.30 | 0% | 61.1% |
| `wp_bpe_ppig_sp` | 50,256 | 25.6 | 3.41 | 5.38 hin_Deva | 0.34 | 0% | 7.78 | 10.21 tam_Taml | 0.31 | 0% | 62.2% |
| `bpe_frequency_g2` | 50,256 | 26.7 | 3.41 | 5.55 hin_Deva | 0.38 | 0% | 8.13 | 12.66 ben_Beng | 0.36 | 0% | 70.9% |
| `gpt2` | 50,256 | 26.7 | 3.46 | 7.41 hin_Deva | 0.43 | 0% | 8.24 | 15.54 tam_Taml | 0.40 | 0% | 70.6% |
| `morpheme_bpe_g2` | 50,256 | 27.6 | 3.46 | 6.27 hin_Deva | 0.44 | 0% | 8.44 | 12.28 ben_Beng | 0.35 | 0% | 76.4% |
| `wp_bpe_pig_sp300` | 50,256 | 25.4 | 3.49 | 6.30 hin_Deva | 0.36 | 0% | 8.30 | 12.69 tam_Taml | 0.29 | 0% | 64.6% |
| `length_penalized_sp300` | 50,256 | 25.3 | 3.52 | 6.82 hin_Deva | 0.40 | 0% | 8.35 | 12.79 tam_Taml | 0.29 | 0% | 64.9% |
| `wp_bpe_pig_sp` | 50,256 | 25.6 | 3.55 | 5.69 hin_Deva | 0.36 | 0% | 7.93 | 10.20 tam_Taml | 0.28 | 0% | 64.9% |
| `morpheme_bpe_sp300` | 50,256 | 26.2 | 3.56 | 7.57 hin_Deva | 0.45 | 0% | 9.62 | 15.87 tam_Taml | 0.33 | 0% | 84.7% |
| `length_penalized_sp` | 50,256 | 25.4 | 3.58 | 6.07 hin_Deva | 0.40 | 0% | 7.99 | 10.28 tam_Taml | 0.30 | 0% | 62.8% |
| `morpheme_bpe_sp` | 50,256 | 26.1 | 3.67 | 6.59 hin_Deva | 0.44 | 0% | 9.19 | 12.98 ben_Beng | 0.31 | 0% | 79.6% |
| `bpe_frequency_sp300` | 50,256 | 25.1 | 3.71 | 7.02 hin_Deva | 0.42 | 0% | 9.18 | 16.51 tam_Taml | 0.32 | 0% | 75.8% |
| `unigram_lm_g2` | 50,256 | 27.2 | 3.76 | 7.05 hin_Deva | 0.49 | 0% | 9.08 | 14.69 tam_Taml | 0.34 | 0% | 84.1% |
| `bpe_frequency_sp` | 50,256 | 25.3 | 3.79 | 6.47 hin_Deva | 0.42 | 0% | 9.13 | 13.39 ben_Beng | 0.31 | 0% | 75.5% |
| `unigram_lm_sp300` | 50,256 | 26.1 | 3.84 | 7.90 hin_Deva | 0.49 | 0% | 9.47 | 14.09 tam_Taml | 0.27 | 0% | 82.9% |
| `unigram_lm_sp` | 50,256 | 26.2 | 4.04 | 7.36 hin_Deva | 0.52 | 0% | 9.50 | 15.12 tam_Taml | 0.31 | 0% | 84.5% |
| `super_bpe` | 50,256 | 24.3 | 4.11 | 7.15 hin_Deva | 0.40 | 0% | 9.35 | 13.89 ben_Beng | 0.31 | 0% | 71.4% |
| `tm_englishcode_32000` | 32,000 | 24.7 | 5.76 | 13.52 hin_Deva | 0.64 | 0% | 11.37 | 16.89 tam_Taml | 0.34 | 0% | 92.9% |
| `tm_englishcode_50256` | 50,256 | 22.9 | 6.16 | 14.58 hin_Deva | 0.66 | 0% | 12.26 | 18.21 tam_Taml | 0.34 | 0% | 92.9% |

Premium by language: the lowest and highest mean in-mix premium of each tier and GPT-2; "UTF-8 bytes" is the byte ratio, the premium of a tokenizer with one token per byte. The English row gives tokens (bytes) per sentence.

| Language | Group | UTF-8 bytes | `wp_bpe_scp@16384` | `super_bpe@16384` | `wp_bpe_npmi` | `tm_englishcode_50256` | `gpt2` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| eng_Latn | reference | (130.5) | 86.3 | 31.5 | 46.1 | 22.9 | 26.7 |
| deu_Latn | in-mix | 1.18 | 1.33 | 1.71 | 1.65 | 2.35 | 2.14 |
| fra_Latn | in-mix | 1.24 | 1.33 | 1.65 | 1.65 | 2.50 | 2.00 |
| spa_Latn | in-mix | 1.21 | 1.33 | 1.63 | 1.70 | 2.33 | 1.98 |
| rus_Cyrl | in-mix | 1.97 | 1.01 | 2.19 | 2.29 | 11.26 | 5.68 |
| cmn_Hans | in-mix | 0.91 | 0.42 | 1.44 | 2.17 | 5.24 | 3.15 |
| jpn_Jpan | in-mix | 1.27 | 0.44 | 1.52 | 2.25 | 7.49 | 2.97 |
| arb_Arab | in-mix | 1.60 | 1.02 | 1.86 | 2.47 | 9.16 | 4.36 |
| hin_Deva | in-mix | 2.55 | 0.95 | 2.11 | 3.01 | 14.58 | 7.41 |
| hun_Latn | in-mix | 1.16 | 1.35 | 1.75 | 1.91 | 3.67 | 2.66 |
| fin_Latn | in-mix | 1.10 | 1.26 | 1.69 | 1.70 | 3.02 | 2.28 |
| kor_Hang | out-of-mix | 1.19 | 1.73 | 4.85 | 2.76 | 6.83 | 5.03 |
| tha_Thai | out-of-mix | 2.76 | 2.80 | 11.37 | 3.55 | 15.72 | 8.98 |
| tam_Taml | out-of-mix | 3.19 | 4.81 | 13.13 | 5.36 | 18.21 | 15.54 |
| amh_Ethi | out-of-mix | 1.71 | 2.57 | 7.03 | 4.80 | 9.77 | 7.70 |
| ell_Grek | out-of-mix | 2.17 | 1.99 | 8.90 | 5.06 | 12.39 | 6.51 |
| heb_Hebr | out-of-mix | 1.39 | 1.41 | 5.66 | 2.63 | 7.93 | 4.35 |
| ben_Beng | out-of-mix | 2.61 | 3.91 | 10.67 | 5.52 | 14.93 | 9.56 |

Findings:

- Mean in-mix premium: intrinsic tier (30% multilingual mix) 1.04-1.75 (median 1.29); GPT-2-swap tier (English-centric) 2.08-6.16 (median 3.39). 39 of 39 intrinsic-tier tokenizers are below the lowest swap-tier tokenizer.
- Mean out-of-mix premium: intrinsic tier (30% multilingual mix) 2.00-9.72 (median 5.08); GPT-2-swap tier (English-centric) 4.24-12.26 (median 7.89). 12 of 39 intrinsic-tier tokenizers are below the lowest swap-tier tokenizer.
- Absolute cost, mean tokens per sentence over the group's languages (English plays no part), median over each tier: in-mix intrinsic tier 66.5 vs swap tier 93.2; out-of-mix intrinsic tier 260.9 vs swap tier 207.5.
- Latin vs non-Latin in-mix languages (premium averaged over the tier's tokenizers): intrinsic tier (30% multilingual mix) 1.31 on deu_Latn, fra_Latn, spa_Latn, hun_Latn, fin_Latn vs 1.29 on rus_Cyrl, cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva; GPT-2-swap tier (English-centric) 2.17 on deu_Latn, fra_Latn, spa_Latn, hun_Latn, fin_Latn vs 4.54 on rus_Cyrl, cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva.
- Intrinsic tier (30% multilingual mix), 39 tokenizers: highest premium 16.45 (`super_bpe@65536` on tam_Taml); worst in-mix language hin_Deva for 32, rus_Cyrl for 3, hun_Latn for 2, spa_Latn for 2; worst out-of-mix language tam_Taml for 39; byte-fallback tokens are 72.9-93.9% of out-of-mix tokens. Its lowest in-mix premium, `wp_bpe_scp@16384` (1.04), comes with 86.3 tokens per English sentence against the tier's median 50.6: the premium is relative to each tokenizer's own English cost.
- GPT-2-swap tier (English-centric), 36 tokenizers: highest premium 18.21 (`tm_englishcode_50256` on tam_Taml); worst in-mix language hin_Deva for 36; worst out-of-mix language tam_Taml for 22, ben_Beng for 14; byte-fallback tokens are 56.6-92.9% of out-of-mix tokens. Its lowest in-mix premium, `wp_bpe_npmi` (2.08), comes with 46.1 tokens per English sentence against the tier's median 26.2: the premium is relative to each tokenizer's own English cost.
- 9 tokenizers encode English at under 1.5 bytes per token, close to raw bytes (`wordpiece_pmi@16384`, `wp_bpe_base@16384`, `wp_bpe_ft@16384`, `wordpiece_pmi@32768`, `wp_bpe_base@32768`, `wp_bpe_ft@32768`, `wordpiece_pmi@65536`, `wp_bpe_base@65536`, `wp_bpe_ft@65536`): their premium is on average 0.06 from the UTF-8 byte ratio (column "UTF-8 bytes"), so it reflects the script's byte length, not fair compression.
<!-- END GENERATED: flores -->

### Known limitations

- **One scale, one split.** Sections 1-5 use one 52.5 MB train split (seed 0). The swap (section 6) trains on different text. Tokenizer training is deterministic; only LMs have seeds.
- **PMI-ranked variants barely compress.** `wordpiece_pmi`, `wp_bpe_ft`, `wp_bpe_base` encode at ~1.01-1.03 bytes/token using 282-1,050 of 16,384 tokens. This is the scorers' intended behaviour.
- **No duplicate variants.** The fingerprint check found no byte-identical tokenizers among the 13 variants at any size.
- **Not benchmarked.** `unigram_sampling` and `byte_entropy` are implemented but not in the suite. BPE-dropout appears only in section 3.
- **Proxy LM.** A NoPE Transformer (no positional encoding); rankings at this scale need not carry to real LMs. With 3 seeds, the top five of the small LM are within noise; the big LM resolves four above `bpe_frequency` but not their mutual order.
- **Morphology.** Morph F1 uses SIGMORPHON 2022 gold for 5 languages. `morpheme_bpe` has supervision the others do not.
- **Throughput.** Single-threaded, load-dependent. Three measurements exist (sections 2/4, section 5 laptop, section 5 Colab); compare within one table only. The pinned-laptop table has the smallest run-to-run spread (median 1.02x).
