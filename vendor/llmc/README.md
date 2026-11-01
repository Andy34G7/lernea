# Vendored from karpathy/llm.c

- Upstream: https://github.com/karpathy/llm.c at commit `f1e2ace651495b74ae22d45d1723443fd00ecd3a`
  (2025-05-10). `train_gpt2.py` last changed upstream in `7b929300217ff1a974b63791a228928b39b26409`.
- License: MIT, Copyright (c) 2024 Andrej Karpathy ([`LICENSE`](LICENSE), copied unchanged).
- `train_gpt2.py`, `pyrun_gpt2_124M.sh` (upstream `scripts/`) and `LICENSE`: unchanged copies.
- `train_gpt2_swap.py`: `train_gpt2.py` plus blocks between `# >>> SWAP` and `# <<< SWAP`, used by
  `python -m tokenisers.benchmarks.nanogpt_swap`. `diff train_gpt2.py train_gpt2_swap.py` shows every
  change; `tests/e2e/test_bench_nanogpt_swap.py` checks that nothing else differs.

Why llm.c: modded-nanogpt's speedrun target (3.28 val loss) is llm.c's GPT-2 (124M) reproduction,
and modded-nanogpt itself started as a variant of this PyTorch trainer. Its first commit is already
modified (RMSNorm, no biases, other schedule) and its llm.c record (`records/track_1_short/2024-10-13_llmc`)
holds only a log, so the unmodified PyTorch trainer and run script come from llm.c.

The recipe is `pyrun_gpt2_124M.sh` (the PyTorch twin of llm.c's `run_gpt2_124M.sh`). The benchmark
changes only these arguments, identically for every tokenizer: `--num_iterations` 600 (from 18,865),
`--warmup_iters` 22 (700 x 600 / 18,865), `--val_loss_every` 50 (from 250), `--batch_size` (the
micro-batch; `--total_batch_size` stays 524,288, so only the gradient accumulation changes) and the
data/log paths. The SWAP blocks, active only when `$SWAP_CONFIG` is set:

- the model's vocabulary is the tokenizer's plus one document separator, padded to a multiple of
  128; the padded ids are set to -inf before the loss, so they take no part in the softmax (as in
  llm.c's CUDA trainer, which pads to 50,304 but normalises over the 50,257 real ids);
- llm.c's val loop (the mean of `val_max_steps` batch means) is replaced by the summed NLL over every
  target of the complete val stream (last window padded with ignored targets), giving val loss in
  nats per token and bits per UTF-8 byte of the val documents, at llm.c's val steps;
- the GPT-2 tiktoken encoding is not loaded (llm.c uses it only to sample and to write tensors, both
  off in the recipe);
- per-step and per-eval records, parameter count and peak GPU memory go to a JSON result file.
