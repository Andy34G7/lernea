# Modular Rust + Python Tokenizers Research & Benchmarking Platform

A high-performance **Rust (`PyO3` / `maturin`) + Python** research and benchmarking platform for designing, modifying, and evaluating subword and dynamic tokenization algorithms across:
1. **Tokka-Bench** (multilingual natural languages + programming languages)
2. **Extended Intrinsic Diagnostics** (Rényi efficiency $E_\alpha$, Morphological Boundary F1, Identifier Boundary F1, Vocabulary Utilization, and Systems Throughput in MB/s)
3. **Extrinsic Proxy-LM Evaluation** (a small causal Transformer trained under a fixed UTF-8 byte budget, evaluated via **Validation Bits-Per-Byte [BPB]** across English, Multilingual, and Code splits locally or on Google Colab GPUs via `colab-cli`). The proxy LM is *not* a Llama/GPT block: it uses PyTorch `TransformerEncoderLayer`s (pre-LayerNorm, ReLU FFN, biases) with a causal mask, tied input/output embeddings and **no positional encoding of any kind** (NoPE: token order reaches the model only through the causal mask). Keep that in mind when comparing its BPB numbers with results from RoPE/SwiGLU models.

---

## Architecture & Repository Layout

```text
├── Cargo.toml                      # Rust workspace manifest
├── pyproject.toml                  # Maturin + uv build & CLI configuration
├── crates/
│   ├── tokenisers_core/            # Pure Rust core library (unit, integration & stress tests)
│   │   └── src/
│   │       ├── arena.rs            # Doubly-linked symbol arena & inverted pair index
│   │       ├── vocab.rs            # Byte-level lossless base alphabet & vocabulary mapping
│   │       ├── scoring/            # Pluggable scorers: Frequency (BPE), WordPiece/PMI, MDL (ΔBits), Length-Penalized
│   │       ├── engine/             # Unified bottom-up merge engine & Top-Down Unigram EM + Viterbi pruning
│   │       ├── curricula/          # Two-Stage SuperBPE & Morpheme-Guided boundary priors
│   │       ├── inference/          # Greedy, BPE-Dropout, Unigram FFBS/k-best sampling, Byte-Entropy patching
│   │       └── export/             # Canonical HuggingFace tokenizer.json serializer
│   └── tokenisers_py/              # PyO3 C-extension bindings (`tokenisers._core`) with GIL release
├── python/tokenisers/
│   ├── models.py                   # High-level Python API (BpeTokenizer, UnigramTokenizer, SuperBpeTokenizer, MorphemeBpeTokenizer, ByteEntropyTokenizer)
│   ├── wrapper.py                  # DynamicResearchTokenizer (HuggingFace PreTrainedTokenizer subclass)
│   ├── data/                       # Streaming & cached dataset pipeline (FineWeb-Edu, FineWeb-2, StarCoder, SIGMORPHON)
│   ├── dashboard/                  # Artificial Analysis-style interactive Web UI & REST API server (`tokenisers-dashboard`)
│   └── benchmarks/
│       ├── tokka/                  # Tokka-Bench multilingual + code evaluation suite
│       ├── intrinsic/              # Rényi efficiency, Morphological F1, Identifier F1, Throughput MB/s
│       └── colab/                  # Extrinsic Proxy-LM fixed-byte-budget trainer & Colab GPU orchestrator
├── vendor/tokka-bench/             # Vendored bgub/tokka-bench reference suite
├── benchmarks/results/             # Generated JSON, CSV, and Markdown leaderboards
└── tests/e2e/                      # Automated E2E & adversarial tests (Tiers 1–5 + WP-BPE + Dashboard)
```

---

## Implemented Tokenizer Algorithms & Modifications

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

A pair seen fewer than `min_frequency` times is never merged: 2 for `wordpiece_pmi`, `wp_bpe_ft`,
`wp_bpe_scp`, `wp_bpe_npmi`, `wp_bpe_ppig` and `mdl_bits`, 1 for the other merge variants. Other
defaults: $\alpha = 1$ (`wordpiece_pmi`), $d = 0.75$ (`wp_bpe_scp`), $\alpha = 0.5$
(`wp_bpe_npmi`), $\eta = 0.35$ (`wp_bpe_ppig`), 24 bits per vocabulary entry (`mdl_bits`),
$\beta = 0.5$ (`length_penalized`), and for `unigram_lm` 10 EM iterations per pruning round with
shrinking factor 0.75. The benchmark suite trains with these values and $\lambda = 2$ for
`morpheme_bpe`; each result file records every training parameter under `meta.variants` (files
written before this was added, including the committed results from `603a983`, list only the scorer
and non-default values; that commit used the defaults above).

---

## Quickstart & CLI Usage

### 1. Build & Run Tests
```bash
# Install the dev tools (maturin, pytest); a plain `uv sync` removes them
uv sync --extra dev

# Run the native Rust test suite
cargo test --workspace

# Build the PyO3 extension in release mode & run the Python E2E suite
uv run maturin develop --release
uv run pytest tests/e2e/ -v
```

### 2. Launch Interactive Benchmark Dashboard (Artificial Analysis-Style Web UI)
```bash
# Starts local web server at http://127.0.0.1:8765 with interactive charts & live tokenization playground
uv run python -m tokenisers.dashboard --port 8765
```

### 3. Prepare Standardized Corpus
```bash
# Real corpus: FineWeb-Edu + FineWeb-2 (10 languages) + code, exact duplicates removed,
# document-disjoint train/val/test, provenance in manifest.json (default 50 MiB train;
# StarCoder needs a Hugging Face token, see Data access)
uv run python -m tokenisers.data --preset small --sigmorphon eng fra hun rus spa
# Synthetic offline test corpus (labelled synthetic everywhere)
uv run python -m tokenisers.data --preset smoke --offline
```

**Data access.** FineWeb-Edu, FineWeb-2 and SIGMORPHON 2022 are public. The code domain reads the
gated `bigcode/starcoderdata`, which needs a Hugging Face token whose account has accepted the
dataset's terms. Setting `HF_TOKEN` in the environment (optional) enables it; `huggingface_hub` also
uses a token saved by `hf auth login`. Without a token that can read StarCoder, the code domain is
read from the ungated `codeparrot/github-code-clean` instead, with the same six languages and
per-language byte budgets, and the substitution is recorded under `fallbacks` in the dataset
`manifest.json` and in every result JSON (`meta.dataset.fallbacks`). Whether a token is present is
part of the dataset cache key, so adding one later builds a new dataset instead of reusing the
fallback one. The committed results used StarCoder: `HF_TOKEN` was not set, but a saved login token
with access was (manifest: `hf_token_present: true`, no fallbacks).

### 4. Run Benchmarks
All three tiers read the prepared dataset (step 3): tokenizers train on its **train** split only;
tokka and the intrinsic suite evaluate on the held-out **test** split. The proxy LM trains on
documents of the same **train** split the tokenizers were trained on (shuffled, each used at most
once; *not* held out from tokenizer training) and is scored on the held-out **val** split.
Disjointness of train and evaluation documents is asserted at runtime. It holds at document level
only: de-duplication drops exact (SHA-256) duplicate documents, not near-duplicates or shared
boilerplate. In the default dataset 7 of 666 test and 5 of 687 val documents share a line of at
least 200 characters with a train document. The default vocabulary is
16,384 (`--vocab-size 32768` / `65536` for the larger sizes). Every result JSON carries a `meta`
block with the dataset manifest, requested and actual vocabulary sizes, per-variant training
parameters and warnings, seeds, hardware/backend and git commit. Variants that produce
byte-identical tokenizers are detected and reported as one result.

```bash
# Tier 1: Tokka-Bench metrics (vendored tokka-bench definitions) per language of the test split.
# The macro average only includes languages with >= 3 held-out documents and >= 10 KB
# (--min-docs-per-lang / --min-bytes-per-lang); the others are reported and listed with their sizes.
uv run python -m tokenisers.benchmarks.tokka

# Tier 2: Rényi efficiency, utilisation, SIGMORPHON 2022 morph F1 (test split gold; morpheme_bpe
# lexicon from the train split), identifier F1 (mined from held-out code), throughput (median of 5)
uv run python -m tokenisers.benchmarks.intrinsic

# Tier 3: proxy-LM bits per byte, fixed 10 MB train budget, 3 seeds (mean ± sd), on a Colab GPU.
# A Colab failure stops the run unless --allow-local-fallback is given (the fallback is recorded);
# --local runs on this machine. The backend actually used is recorded and shown with the table.
uv run python -m tokenisers.benchmarks.colab --gpu T4

# Copy the result JSON into the tables below
uv run python scripts/gen_readme_tables.py
```

`--offline` runs everything on the synthetic test corpus (labelled synthetic in the output).

**Colab.** Tier 3 on Colab needs the `colab` CLI (`google-colab-cli`) **>= 0.7.5**: 0.6.0's `exec`
is broken. The run checks `colab version` before it starts a session; upgrade with
`uv tool upgrade google-colab-cli`. The job is not one long `colab exec`: the benchmark uploads the
job, starts it detached on the VM (output in `/content/job.log`) and polls it every
`--poll-interval` seconds (default 60) with short `colab exec` calls. Polls that hang or lose the
connection are retried, and so are the uploads and the final download of `result.json` (5
consecutive failures of a step abort the run). `colab exec` exits 0 even when the remote
code raises, so success is read from the `status.json` the job writes last. A remote exception
fails the run with the remote traceback; so does a job that dies without a status (log tail
shown) or runs past `--timeout`. The session is always stopped.

**Memory.** Peak training memory grows with the train split and, for the merge-based variants,
with the vocabulary size. Measured whole-process peaks at V = 65,536 on the 52.5 MB default train
split: `super_bpe` 2.9 GiB, the largest plain BPE variant (`wp_bpe_pig`) 1.25 GiB, `unigram_lm`
0.5 GiB (on 25 MB: 1.5 / 0.67 / 0.40 GiB). The CLIs estimate 600 MiB plus 75 bytes per text byte
for `super_bpe` and 60 for every other variant, at least 20% above every measurement
(`MEASURED_TRAIN_PEAKS` in `tests/e2e/test_bench_reporting.py`): ~4.2 GiB for the 50 MiB default
with `super_bpe`, ~3.5 GiB without it. Before preparing the dataset they
exit if that exceeds the memory the process may use: `MemAvailable` minus a reserve of max(2 GiB,
15% of RAM) for the rest of the machine, or the free part of a cgroup / `systemd-run -p MemoryMax=`
limit if that is smaller. The reserve applies inside a cgroup too, so a `MemoryMax=` scope never
raises the allowance. Lower `--train-bytes` or drop the variant from `--variants`; or, if a
`MemoryMax=` scope is the safety net, pass `--skip-memory-check` inside it.

---

## Benchmark Results

The tables below are generated from `benchmarks/results/*/` by `scripts/gen_readme_tables.py`; a
test fails if they differ from the committed JSON. The numbers previously shown here came from
synthetic data, evaluated on training text, with a proxy LM that silently ran on a local CPU; they
are kept, marked invalid, in [`benchmarks/results/legacy_smoke/`](benchmarks/results/legacy_smoke/).

**Scale and setup.** All committed results come from one run of the suite at commit `603a983` on
the default dataset (`python -m tokenisers.data --preset small`): a 52.5 MB train split (11,102
documents: FineWeb-Edu English 50%, FineWeb-2 in 10 languages 30%, StarCoder code in 6 languages
20%; no fallback source was needed) with held-out val (3.1 MB) and test (3.1 MB) splits. Tokka and
intrinsic results exist for V = 16,384 (sections 1 and 2) and V = 32,768 / 65,536 (section 4);
the proxy LM (section 3) ran at V = 16,384 only, on a Colab T4 GPU with the default 10 MB byte
budget and 3 seeds, and again at commit `6d260c2` with the big proxy LM (100 MB budget, 3 seeds) on
a Colab L4. Tokenizer training and every other step ran locally on an Intel Core Ultra 7
155H (22 threads, no GPU, 15 GB RAM shared with a desktop session). Runs whose arguments include
`--skip-memory-check` started after the desktop's memory use had grown: the preflight estimate
(4.2 GiB) exceeded the 3.0 GiB it then allowed, so they ran inside a hard 4 GiB cgroup limit
(`systemd-run -p MemoryMax=4G`) instead, above the measured 2.9 GiB peak. Section 5 re-times the
same tokenizers on a Colab CPU runtime and on one pinned laptop core. Each section's header
lists the exact arguments, data sizes, backend and commit. Known limitations are listed after
section 5.

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
- fixed budget 10,000,000 UTF-8 bytes of LM training text per tokenizer, drawn from the same train split the tokenizers were trained on (each document used once); seeds [0, 1, 2] (mean ± sample sd)

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
- requested V = 16384; real corpus: tokenizers trained on the train split (52.5 MB); evaluated on the held-out val split (3.1 MB); commit `6d260c23b6`; generated 2026-10-31T20:24:38Z
- arguments: `--preset big --job-dir /home/andey/.cache/tokenisers/big-lm-job --resume --concurrency 1`
- backend: Colab GPU `NVIDIA L4`
- proxy LM: pre-LayerNorm causal Transformer (torch TransformerEncoderLayer: ReLU FFN, biases), no positional encoding of any kind (NoPE: order comes only from the causal mask), tied input/output embeddings; d_model=512, n_layers=8, n_heads=8, d_ffn=2048, max_seq_len=512
- fixed budget 100,000,000 UTF-8 bytes of LM training text per tokenizer, drawn from the train split the tokenizers were trained on plus an LM text pool with no val or test document (each document used once; 62.1 MB of the 110.0 MB of LM text came from the pool, `meta.lm_text`); seeds [0, 1, 2] (mean ± sample sd)

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

Setup of this run: `--preset big --resume --concurrency 1` from commit `6d260c2` on one Colab L4 (22 GB);
the 33 runs (11 tokenizers x 3 seeds) trained one after another. Model: 8 layers, d_model 512, 8 heads,
FFN 2048, sequence length 512; 100 MB byte budget per run; seeds 0, 1, 2. Measured wall time: 5 h 03 min
from launch (20:51 IST) to the result on the laptop (01:54:38 IST). Of that, uploading the job took
7 min 20 s and training plus evaluation on the VM 17,699.8 s (4 h 55 min, `meta.lm_seconds`), a mean of
536 s (8.9 min) per run; the run files do not record per-run time. Two earlier launches of the same job
ran out of GPU memory, with `--concurrency 3` (processes at 9.1, 7.1 and 5.1 GiB) and with
`--concurrency 2` (one process at 11.1 GiB, the other at 9.1 GiB and asking for 2 GiB more). The
failing allocation is the fp32 logits of one 64 x 512 x 16,384 batch (2 GiB). The preset now defaults
to `--concurrency 1`.

##### Small vs big proxy LM
<!-- BEGIN GENERATED: colab_compare -->
- the 11 tokenizers both proxy-LM tiers trained; small LM: d_model=128, n_layers=2, 10,000,000-byte budget, seeds [0, 1, 2], `Tesla T4`; big LM: d_model=512, n_layers=8, 100,000,000-byte budget, seeds [0, 1, 2], `NVIDIA L4`
- rank: by mean macro val BPB (1 = best); Δ: macro val BPB minus `bpe_frequency`'s in the same tier (negative = better) ± its standard error √(sd²/n + sd_ref²/n) over the n seeds (sample sd)
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

With the small LM, the four tokenizers ahead of `bpe_frequency` are within about two standard errors
of it, so the order of the top five is not resolved by 3 seeds. With the big LM, `wp_bpe_ppig`,
`wp_bpe_pig`, `wp_bpe_npmi` and `super_bpe` beat `bpe_frequency` by several standard errors; among the
three wp_bpe variants, neighbouring means differ by less than two standard errors of their difference,
so their order is not resolved. Below `bpe_frequency` the order changes: `morpheme_bpe` and
`unigram_lm` move from clearly worse to within one standard error of it, and `length_penalized` drops
from 6th to 9th.

#### Big proxy LM: settings and how to run

`--preset big` selects a larger proxy LM (same architecture as above): 8 layers, d_model 512, 8 heads,
FFN 2048, sequence length 512 (~25M non-embedding parameters, ~8.4M embedding parameters at V = 16,384).
One hyperparameter set for every tokenizer: AdamW lr 6e-4, weight decay 0.1, warmup 2% of steps then
cosine decay, gradient clip 1.0, 64 x 512 = 32,768 tokens per step, bf16 autocast when the GPU supports
it, 100 MB byte budget per run, seeds 0/1/2. Default GPU L4; 11 variants (wordpiece_pmi, wp_bpe_base and
wp_bpe_ft are left out); `--concurrency N` (variant, seed) runs share the GPU as separate processes, each
writing its own result file (default 1: on the L4, 2 and 3 ran out of memory). Results go to
`benchmarks/results/colab_big/`. Explicit flags override the preset.

LM text. A 100 MB budget needs 110 MB of LM text (budget + 10%, each document used once), but the
tokenizers stay the ones trained on the 52.5 MB default train split. The preset therefore adds an LM text
pool (`--lm-text-bytes 120000000`, `DataPipeline.lm_pool`): the train split of the dataset with the same
sources, revisions, seed and split rule but a 120 MB train budget (cached under its own key next to the
default dataset), minus every default train/val/test document by sha256. The LM trains on 110.0 MB drawn
from the train split plus the pool (62.1 MB of it from the pool). At runtime the CLI checks that this text
shares no document with val or test, and that every variant's encoded LM text covers the budget. BPB is
measured on the same val split as the small tier. The pool's provenance (its dataset manifest, document
and byte counts) is in `meta.lm_text` of the results.

```bash
# 1. Locally: pool, tokenizers, encodings into the job dir; no LM training.
uv run python -m tokenisers.benchmarks.colab --preset big --prepare-only --skip-memory-check
# 2. On Colab (L4): uploads the job, trains, downloads each finished run as it lands.
uv run python -m tokenisers.benchmarks.colab --preset big --resume
```

Without `--prepare-only` one command does both steps. The job (`config.json` and one compressed `.npz` of
token ids per tokenizer) is kept in `benchmarks/results/colab_big/job/` (gitignored; `--job-dir` moves
it). If a run fails or is interrupted, re-run it with `--resume`: no tokenizer training or encoding, and
the finished (variant, seed) run files already downloaded are uploaded to the new session and not
retrained. `--resume` refuses a job prepared with different model, training or data arguments;
`--concurrency` may change (e.g. lower it after an out-of-memory error). Ctrl-C or `kill <pid>` stops
the Colab session before the CLI exits (a job started with `&` from a script ignores SIGINT; use `kill`).
`--gpu none` runs on a CPU runtime (for a smoke test with toy sizes).

Measured on the laptop (full scale, `--prepare-only`): 372 s and 3.2 GiB peak RSS with the pool
downloaded from Hugging Face (~3 min of that; the pool's dataset is 135 MB on disk), 254 s and 3.5 GiB
once cached (tokenizer training 104 s, encoding 144 s). Every tokenizer reproduces the committed V = 16,384
small-tier run exactly (validation bytes/token per domain identical). The job is 463 MB (11 files of
41-47 MB). Smoke test on a Colab CPU runtime through the same code path (`--preset big --gpu none`,
d_model 32, 1 layer, 200 KB budget, bpe_frequency + bpe_dropout_p0.1, 2 seeds, `--concurrency 2`): the
first session was dropped after the two bpe_frequency runs had been downloaded; `--resume` uploaded them
into a new session, which trained the two missing runs (11 min on the VM: one wave of two concurrent
runs, as in the first session) and returned the full result.

Upload: the job uploads these pre-encoded ids, about 5 min at ~1.5 MB/s (again after each resume into a
new session). The alternative, uploading the LM text (~38 MB compressed) and saved tokenizers and encoding
on the VM after building the extension there (as `benchmarks/throughput.py` does), would take ~4 min
(~1 min build, ~2 min encoding, under a minute of upload) and needs a clean git tree and more code, so it
is not used.

GPU time: the estimate made before the run (~1.3B training tokens at ~230 MFLOP per token, about 2-4 h
of L4 time at 20-40 TFLOPS sustained in bf16) was low. The measured run trained 1.33 billion tokens in
4 h 55 min including evaluation (about 75,000 tokens/s) at concurrency 1. Compute units used were not
recorded. The default `--timeout` is 6 h (this run took 5 h 03 min); a job past it stops, and
`--resume` continues it.

### 4. Larger Vocabularies (V = 32,768 and 65,536)

Same train and test data as sections 1 and 2; only `--vocab-size` differs. The proxy LM was not run at
these sizes.

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

The MB/s columns of sections 2 and 4 come from the full suite runs on the shared laptop. Here the
same tokenizers (all 13 variants at the three sizes, retrained with the same code, data and
parameters; each reproduced the bytes/token and vocabulary size of sections 2 and 4 exactly) are
timed again with the same `IntrinsicEvaluator.benchmark_throughput` on the same held-out
documents, in two full passes over all tokenizers so the run-to-run spread is measured. Each
tokenizer is saved, reloaded on the measuring machine and must produce token ids identical
(sha256) to the original on every document before it is timed. The three measurements are
compared under Known limitations.

```bash
# Train, save and verify all tokenizers once into a job directory, then time them.
uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp --prepare-only
uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp --colab   # Colab CPU runtime
taskset -c 3 uv run python -m tokenisers.benchmarks.throughput --job-dir /tmp/tp  # one laptop core
```

`--colab` creates a CPU runtime (no accelerator), uploads the saved tokenizers, the documents and
`git archive HEAD` (so it refuses uncommitted code changes), builds and installs the native
extension on the VM, and runs the job there detached, polled like the proxy-LM job.

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

**Laptop, pinned to one performance core (re-run):**

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

How much does the tokenizer matter when everything else is fixed? `python -m
tokenisers.benchmarks.nanogpt_swap` trains GPT-2 small (12 layers, 12 heads x 64, d = 768, context
1024, learned positions, pre-LayerNorm, GELU, tied embeddings, no dropout) with llm.c's PyTorch
recipe once per tokenizer and scores validation bits per byte on the same text. The trainer is
[`vendor/llmc/train_gpt2_swap.py`](vendor/llmc/): llm.c's `train_gpt2.py` at `f1e2ace`, changed
only inside delimited `SWAP` blocks; the hyperparameters are those of llm.c's
`scripts/pyrun_gpt2_124M.sh` (AdamW (0.9, 0.95), lr 6e-4, weight decay 0.1 on 2-D weights, grad
clip 1.0, 524,288 tokens per step, bf16 autocast, SDPA flash attention, `torch.compile`). Identical
for every tokenizer, the only changes are: the vocabulary (tokenizer + one document separator,
padded to a multiple of 128; padded ids are kept out of the softmax as in llm.c's CUDA trainer),
600 steps with the warmup scaled by the same fraction (700 of 18,865 -> 22) and the same cosine
decay to 0, the micro-batch (gradient accumulation keeps 524,288 tokens per step), val BPB on the
whole val set every 50 steps (`--eval-every`; 100 in the committed run) and after the last, data
paths and a result file. A test checks that the run arguments differ from llm.c's script only in
those fields and that the trainer differs from llm.c's file only inside `SWAP` blocks.

**Tokenizers (12).** GPT-2 (tiktoken, 50,257 with `<|endoftext|>`, the baseline); TokenMonster
`englishcode-50256` and `englishcode-32000` (`consistent` mode, as in @alexjc's TokenMonster
speedrun record); and our 9 variants `bpe_frequency`, `wp_bpe_pig`, `wp_bpe_ppig`, `wp_bpe_npmi`,
`super_bpe`, `unigram_lm`, `length_penalized`, `morpheme_bpe` (SIGMORPHON train-split lexicon) and
`mdl_bits`, trained with the suite's default parameters at 50,256 tokens on the first 50 MB of
FineWeb train documents, so that with the separator they have 50,257 like GPT-2. Each tokenizer
gets one separator id, placed between documents exactly where GPT-2's shards have `<|endoftext|>`.
Every prebuilt TokenMonster `englishcode` vocabulary NFD-normalizes its input and so is lossy
(decode(encode(x)) != x on 1,053 of the 14,982 val documents; capcode also turns U+0130 into "I");
the benchmark uses the closest lossless variant: the `nocapcode` vocabulary of the same size and
mode with the normalization flag cleared (same tokens and ids). Every tokenizer must reproduce
every val document exactly before it is trained. On the val text GPT-2 gives 4.44 bytes per token,
TokenMonster 5.32 (50,256) and 4.95 (32,000), `super_bpe` 5.03, and the other 8 of ours 2.70-2.75:
their pre-tokenizer isolates every whitespace run as its own token, so at 524,288 tokens per step
they see about 40% fewer bytes than GPT-2.

**Data.** The text is the speedrun's own pre-tokenized GPT-2 FineWeb shards
(`kjj0/fineweb10B-gpt2`), decoded back to documents and re-encoded by each tokenizer into llm.c
`.bin` files. Val = the 14,982 complete documents (46.5 MB) in the speedrun's val window (the first
20 x 524,288 GPT-2 tokens of `fineweb_val_000000.bin`); BPB = NLL in bits over every val token
(separators included) / UTF-8 bytes of those documents, with the last window padded and masked, so
every tokenizer is scored on the same bytes. Train = the train-shard documents in order (one pass,
no repeats), minus any document whose sha256 equals a val document's; each tokenizer's stream
holds 600 x 524,288 tokens plus 1%. A tokenizer with more bytes per token therefore sees more text
in the same 600 steps: compute and tokens per step are fixed, text is not. The step at which a
curve first reaches the speedrun's target (3.28 nats per GPT-2 token, converted to BPB with GPT-2's
bytes per token on this val text) is reported; at 600 of llm.c's ~19,000 steps it is not expected
to be reached.

```bash
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --prepare-only  # local CPU: tokenizers
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --pilot --gpu L4  # ~20 timed steps
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --gpu L4  # 12 x 600 steps, one session
uv run python -m tokenisers.benchmarks.nanogpt_swap --job-dir /tmp/swap --gpu L4 --resume  # after a drop
```

One Colab GPU session (L4, A100 or H100; a T4 has no bf16) runs the tokenizers one after another;
each finished tokenizer's result file is downloaded on the next poll, and `--resume` uploads the
finished ones to a new session so they are not retrained. The VM builds the native extension from
`git archive HEAD`, downloads and decodes the shards and encodes the text; only the saved
tokenizers and the code are uploaded.

Pilot timings (`--pilot`: 20 steps of GPT-2 and `bpe_frequency`, val encoding of all 12, projected
full-run time): [L4](benchmarks/results/nanogpt_swap/pilot_l4.md),
[A100](benchmarks/results/nanogpt_swap/pilot_a100.md).

Results: one run of all 12 tokenizers in one Colab A100 session (`--gpu A100 --eval-every 100`),
rendered from [`nanogpt_swap.json`](benchmarks/results/nanogpt_swap/nanogpt_swap.json); every run's
per-step train loss and lr are in
[`nanogpt_swap_steps.csv`](benchmarks/results/nanogpt_swap/nanogpt_swap_steps.csv). A 3x peak lr
(`--lr 1.8e-3`) was stopped after GPT-2, which it made worse at every val pass:
[note](benchmarks/results/nanogpt_swap_lr1p8e-3/README.md).

<!-- BEGIN GENERATED: nanogpt_swap -->
- NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, torch 2.11.0+cu130, Colab session `tokenisers-swap_job-4c6b4d48`; commit `ceb11bac38`; generated 2026-11-01T02:01:33Z
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

### Known limitations

- **One scale, one split.** Everything uses one 52.5 MB train split built with split seed 0; the
  100 / 200 MB presets were not run. Tokenizer training is deterministic, so only the proxy LM has
  seeds (3 per tokenizer, mean ± sample sd).
- **PMI-ranked variants barely compress.** `wordpiece_pmi`, `wp_bpe_ft` and `wp_bpe_base` encode
  held-out text at ~1.01–1.03 bytes per token and use 282–1,050 of their 16,384 tokens on it (tokka
  "unique tokens used"): their scores rank rare pairs highest, so the vocabulary fills with long
  tokens that hardly recur. `wp_bpe_scp` sits in between (~2.0 macro bytes/token). These are the
  scorers' behaviour as specified, not failed runs; their proxy-LM BPB is correspondingly poor.
- **No duplicate variants at this scale.** The fingerprint check found no byte-identical tokenizers
  among the 13 variants at any size, so all 13 are reported. `wordpiece_pmi` and `wp_bpe_ft` are
  distinct but nearly identical in every metric.
- **Not benchmarked.** `unigram_sampling` and `byte_entropy` are implemented but not in the suite.
  BPE-dropout appears only in section 3, as `bpe_dropout_p0.1`: the `bpe_frequency` tokenizer with
  dropout p = 0.1 on the LM's training ids (validation ids deterministic).
- **Proxy LM.** A 2-layer, d_model = 128, NoPE Transformer trained on 10 MB per tokenizer; rankings
  at this scale need not carry over to real LMs. Its training documents come from the split the
  tokenizers were trained on; only the val split it is scored on is held out. With 3 seeds, the
  macro BPB of the top five (2.854–2.892) differ by about the seed-to-seed spread (sd up to 0.029),
  so their order is not significant; the gaps to `unigram_lm` and below are larger. The big proxy LM
  (8 layers, d_model = 512, 100 MB per tokenizer, 11 of the variants) separates four tokenizers from
  `bpe_frequency` and reorders the ones below it (section 3, small vs big proxy LM); it is still a
  proxy: ~33M parameters trained on 27-62M tokens.
- **Morphology.** Morph F1 uses SIGMORPHON 2022 gold for 5 languages (eng, fra, hun, rus, spa).
  `morpheme_bpe` is trained with a lexicon from the SIGMORPHON train split (gold test words
  removed), i.e. with supervision the other variants do not get.
- **Throughput.** Single-threaded encode/decode, one call per document, median of 5 timed passes
  over the 3.1 MB test split. MB/s depend on the machine and its load; three measurements of the
  same tokenizers exist:
  - *Sections 2 and 4*: the full-suite runs on the shared laptop, unpinned, with the desktop
    session active. Two runs of the V = 65,536 suite differed by up to 1.7x (1.55x encode, 1.71x
    decode; the first overlapped a busier desktop and the start of another benchmark job; the
    committed numbers are from the second, and all other metrics were identical).
  - *Section 5, laptop*: the same tokenizers re-timed pinned to one performance core
    (`taskset -c 3`, 4.8 GHz max, `powersave` governor, no system settings changed; the desktop
    session and the Colab job's 30 s status polls ran on the other cores). Its two passes differ by a median of 1.02x and at most 1.15x
    (encode) / 1.13x (decode). Against sections 2 and 4 it is a median 1.07x faster at encoding
    and 1.18x at decoding, and up to 1.5x / 2.0x faster in the V = 65,536 table.
  - *Section 5, Colab*: a CPU runtime whose 2 vCPUs are one AMD EPYC 7B12 core with 2 hyperthreads
    (KVM). It was not quieter than the pinned laptop: the median pass-to-pass spread is the same
    (1.03x encode, 1.02x decode) but single cells differ by up to 1.50x / 1.52x between its two
    passes (`wp_bpe_ft` at V = 16,384; 10 of 39 tokenizers differ by more than 1.2x in encode or
    decode). The cause was not isolated: the core's two hyperthreads also run the Jupyter kernel
    that answers the job's status polls, and the host is shared. Its MB/s are a median 0.85x
    (encode) / 0.84x (decode) of the pinned laptop's.

  Compare MB/s between tokenizers within one table and one pass, not across tables, and treat a
  difference smaller than the table's spread as noise. The pinned laptop table has the smallest
  spread.
