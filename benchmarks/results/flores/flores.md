- FLORES+ `openlanguagedata/flores_plus` at `1147e2dc59f3` (CC BY-SA 4.0), devtest split only: 1,012 parallel sentences per language, each encoded on its own with no separator; never used for training
- in-mix: the 10 FineWeb-2 languages of the data pipeline's training mix (FineWeb-2 config -> FLORES+ file): deu_Latn, fra_Latn, spa_Latn, rus_Cyrl, cmn_Hani -> cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva, hun_Latn, fin_Latn; out-of-mix (in no tokenizer's training mix): kor_Hang, tha_Thai, tam_Taml, amh_Ethi, ell_Grek, heb_Hebr, ben_Beng
- intrinsic tier: 13 variants x V = 16,384 / 32,768 / 65,536, retrained as the committed intrinsic results (real dataset `hf-fb5b8df7af00b10f`, 52.5 MB train split); 39 of 39 reproduce the committed test-split bytes/token exactly
- swap tier: the GPT-2-swap tokenizers (English FineWeb text, about 50k tokens) from the prepared jobs `nanogpt-swap-job`, `nanogpt-swap-job-sp`, `nanogpt-swap-pool-g2`, `nanogpt-swap-pool-sp300`
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

GPT-2-swap tier (English FineWeb), sorted by mean in-mix premium:

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

- Mean in-mix premium: intrinsic tier (30% multilingual mix) 1.04-1.75 (median 1.29); GPT-2-swap tier (English FineWeb) 2.08-6.16 (median 3.39). 39 of 39 intrinsic-tier tokenizers are below the lowest swap-tier tokenizer.
- Mean out-of-mix premium: intrinsic tier (30% multilingual mix) 2.00-9.72 (median 5.08); GPT-2-swap tier (English FineWeb) 4.24-12.26 (median 7.89). 12 of 39 intrinsic-tier tokenizers are below the lowest swap-tier tokenizer.
- Absolute cost, mean tokens per sentence over the group's languages (English plays no part), median over each tier: in-mix intrinsic tier 66.5 vs swap tier 93.2; out-of-mix intrinsic tier 260.9 vs swap tier 207.5.
- Latin vs non-Latin in-mix languages (premium averaged over the tier's tokenizers): intrinsic tier (30% multilingual mix) 1.31 on deu_Latn, fra_Latn, spa_Latn, hun_Latn, fin_Latn vs 1.29 on rus_Cyrl, cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva; GPT-2-swap tier (English FineWeb) 2.17 on deu_Latn, fra_Latn, spa_Latn, hun_Latn, fin_Latn vs 4.54 on rus_Cyrl, cmn_Hans, jpn_Jpan, arb_Arab, hin_Deva.
- Intrinsic tier (30% multilingual mix), 39 tokenizers: highest premium 16.45 (`super_bpe@65536` on tam_Taml); worst in-mix language hin_Deva for 32, rus_Cyrl for 3, hun_Latn for 2, spa_Latn for 2; worst out-of-mix language tam_Taml for 39; byte-fallback tokens are 72.9-93.9% of out-of-mix tokens. Its lowest in-mix premium, `wp_bpe_scp@16384` (1.04), comes with 86.3 tokens per English sentence against the tier's median 50.6: the premium is relative to each tokenizer's own English cost.
- GPT-2-swap tier (English FineWeb), 36 tokenizers: highest premium 18.21 (`tm_englishcode_50256` on tam_Taml); worst in-mix language hin_Deva for 36; worst out-of-mix language tam_Taml for 22, ben_Beng for 14; byte-fallback tokens are 56.6-92.9% of out-of-mix tokens. Its lowest in-mix premium, `wp_bpe_npmi` (2.08), comes with 46.1 tokens per English sentence against the tier's median 26.2: the premium is relative to each tokenizer's own English cost.
- 9 tokenizers encode English at under 1.5 bytes per token, close to raw bytes (`wordpiece_pmi@16384`, `wp_bpe_base@16384`, `wp_bpe_ft@16384`, `wordpiece_pmi@32768`, `wp_bpe_base@32768`, `wp_bpe_ft@32768`, `wordpiece_pmi@65536`, `wp_bpe_base@65536`, `wp_bpe_ft@65536`): their premium is on average 0.06 from the UTF-8 byte ratio (column "UTF-8 bytes"), so it reflects the script's byte length, not fair compression.
