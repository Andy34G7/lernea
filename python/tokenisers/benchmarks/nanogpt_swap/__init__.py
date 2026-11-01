"""
Tokenizer-swap benchmark: llm.c's GPT-2 (124M) training recipe with everything fixed except the
tokenizer, scored by validation bits per byte on the same FineWeb documents for every tokenizer.

``swap_job.py`` runs on the Colab VM (data, tokenizers, trainer); ``__main__.py`` is the CLI
(``python -m tokenisers.benchmarks.nanogpt_swap``): local preparation, the Colab run, the results.
The trainer is ``vendor/llmc/train_gpt2_swap.py``, llm.c's ``train_gpt2.py`` with delimited
``SWAP`` blocks.
"""
