use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use rayon::prelude::*;

use crate::error::{Result, TokeniserError};
use crate::pretokenizer::Pretokenizer;
use crate::vocab::{TokenId, Vocabulary, BASE_VOCAB_SIZE};

/// Deterministic parallel accumulation over corpus words. Chunk boundaries depend only on the
/// corpus length (never the thread count) and chunk results are combined sequentially in order,
/// so float sums are bit-identical across runs, processes and machines.
fn accumulate_over_corpus<F>(corpus: &[(Vec<u8>, u64)], width: usize, per_word: F) -> (f64, Vec<f64>)
where
    F: Fn(&[u8], f64, &mut f64, &mut [f64]) + Sync,
{
    let chunk_len = corpus.len().div_ceil(64).max(1);
    let partials: Vec<(f64, Vec<f64>)> = corpus
        .par_chunks(chunk_len)
        .map(|chunk| {
            let (mut ll, mut acc) = (0.0, vec![0.0; width]);
            for (bytes, count) in chunk {
                per_word(bytes, *count as f64, &mut ll, &mut acc);
            }
            (ll, acc)
        })
        .collect();
    let (mut ll, mut acc) = (0.0, vec![0.0; width]);
    for (part_ll, part_acc) in partials {
        ll += part_ll;
        for (a, p) in acc.iter_mut().zip(part_acc) {
            *a += p;
        }
    }
    (ll, acc)
}

/// Numerically stable logsumexp for two log-probabilities: ln(exp(a) + exp(b)).
#[inline(always)]
pub fn logsumexp2(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }
    let max = a.max(b);
    let diff = (a - b).abs();
    if diff > 50.0 {
        max
    } else {
        max + (-diff).exp().ln_1p()
    }
}

/// A node within the arena-allocated CompactPrefixTrie.
#[derive(Clone, Debug)]
pub struct TrieNode {
    pub token_id: Option<TokenId>,
    pub log_prob: f64,
    pub children: Vec<(u8, u32)>,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            token_id: None,
            log_prob: f64::NEG_INFINITY,
            children: Vec::new(),
        }
    }
}

/// Compact Prefix Trie optimized for rapid subword prefix matching over UTF-8 bytes.
#[derive(Clone, Debug)]
pub struct CompactTrie {
    pub nodes: Vec<TrieNode>,
    pub root_table: [Option<u32>; 256],
    pub max_token_len: usize,
}

impl Default for CompactTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactTrie {
    pub fn new() -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(65536),
            root_table: [None; 256],
            max_token_len: 0,
        };
        trie.nodes.push(TrieNode::default()); // Root at index 0
        trie
    }

    pub fn build(vocab: &[(Vec<u8>, TokenId, f64)]) -> Self {
        let mut trie = Self::new();
        for (bytes, id, log_prob) in vocab {
            trie.insert(bytes, *id, *log_prob);
        }
        trie
    }

    pub fn insert(&mut self, bytes: &[u8], token_id: TokenId, log_prob: f64) {
        if bytes.is_empty() {
            return;
        }

        self.max_token_len = self.max_token_len.max(bytes.len());
        let first_byte = bytes[0];

        let mut curr_idx = match self.root_table[first_byte as usize] {
            Some(idx) => idx,
            None => {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(TrieNode::default());
                self.root_table[first_byte as usize] = Some(new_idx);
                new_idx
            }
        };

        for &b in &bytes[1..] {
            curr_idx = self.get_or_create_child(curr_idx, b);
        }

        let node = &mut self.nodes[curr_idx as usize];
        node.token_id = Some(token_id);
        node.log_prob = log_prob;
    }

    #[inline(always)]
    fn get_or_create_child(&mut self, parent_idx: u32, byte: u8) -> u32 {
        let node = &self.nodes[parent_idx as usize];
        for &(b, child_idx) in &node.children {
            if b == byte {
                return child_idx;
            }
        }

        let new_child_idx = self.nodes.len() as u32;
        self.nodes.push(TrieNode::default());

        let node = &mut self.nodes[parent_idx as usize];
        node.children.push((byte, new_child_idx));
        node.children.sort_by_key(|&(b, _)| b);
        new_child_idx
    }

    /// Finds all matching subwords starting at the beginning of `text`.
    /// Returns `(match_len, token_id, log_prob)`.
    #[inline]
    pub fn common_prefix_search(&self, text: &[u8]) -> Vec<(usize, TokenId, f64)> {
        let mut matches = Vec::with_capacity(8);
        if text.is_empty() {
            return matches;
        }

        let first_byte = text[0];
        let mut curr_idx = match self.root_table[first_byte as usize] {
            Some(idx) => idx,
            None => return matches,
        };

        let node = &self.nodes[curr_idx as usize];
        if let Some(id) = node.token_id {
            matches.push((1, id, node.log_prob));
        }

        for (offset, &b) in text[1..].iter().enumerate() {
            let node = &self.nodes[curr_idx as usize];
            let mut next_idx = None;

            for &(edge_byte, child_idx) in &node.children {
                if edge_byte == b {
                    next_idx = Some(child_idx);
                    break;
                }
                if edge_byte > b {
                    break;
                }
            }

            match next_idx {
                Some(idx) => {
                    curr_idx = idx;
                    let child_node = &self.nodes[curr_idx as usize];
                    if let Some(id) = child_node.token_id {
                        matches.push((offset + 2, id, child_node.log_prob));
                    }
                }
                None => break,
            }
        }

        matches
    }
}

/// Dynamic lattice edge discovered during prefix search.
#[derive(Clone, Copy, Debug)]
pub struct LatticeEdge {
    pub start: usize,
    pub end: usize,
    pub token_id: TokenId,
    pub log_prob: f64,
}

/// Viterbi step for 1-best dynamic programming.
#[derive(Clone, Copy, Debug)]
pub struct ViterbiStep {
    pub log_prob: f64,
    pub prev_pos: usize,
    pub token_id: TokenId,
}

pub struct ViterbiDecoder;

impl ViterbiDecoder {
    /// Optimal 1-best Viterbi segmentation for a byte sequence.
    /// If `skip_token` is Some(id), that token is excluded from candidate transitions.
    pub fn decode(
        text: &[u8],
        trie: &CompactTrie,
        skip_token: Option<TokenId>,
    ) -> Option<(f64, Vec<TokenId>)> {
        let n = text.len();
        if n == 0 {
            return Some((0.0, Vec::new()));
        }

        let mut dp = vec![
            ViterbiStep {
                log_prob: f64::NEG_INFINITY,
                prev_pos: 0,
                token_id: 0,
            };
            n + 1
        ];
        dp[0].log_prob = 0.0;

        for start in 0..n {
            if dp[start].log_prob == f64::NEG_INFINITY {
                continue;
            }

            let matches = trie.common_prefix_search(&text[start..]);
            for (len, id, log_prob) in matches {
                if Some(id) == skip_token {
                    continue;
                }

                let end = start + len;
                let candidate_score = dp[start].log_prob + log_prob;

                if candidate_score > dp[end].log_prob {
                    dp[end] = ViterbiStep {
                        log_prob: candidate_score,
                        prev_pos: start,
                        token_id: id,
                    };
                }
            }
        }

        if dp[n].log_prob == f64::NEG_INFINITY {
            return None;
        }

        // Backtrack
        let mut tokens = Vec::new();
        let mut curr = n;
        while curr > 0 {
            tokens.push(dp[curr].token_id);
            curr = dp[curr].prev_pos;
        }
        tokens.reverse();

        Some((dp[n].log_prob, tokens))
    }
}

/// Forward-Backward EM step over pre-tokenized words.
pub struct UnigramEmEngine;

impl UnigramEmEngine {
    /// Computes log-marginal likelihood and posterior expected counts for a single word.
    pub fn forward_backward_word(
        word: &[u8],
        trie: &CompactTrie,
        edges_buf: &mut Vec<LatticeEdge>,
        alpha: &mut Vec<f64>,
        beta: &mut Vec<f64>,
    ) -> (f64, Vec<(TokenId, f64)>) {
        let n = word.len();
        edges_buf.clear();

        for start in 0..n {
            let matches = trie.common_prefix_search(&word[start..]);
            for (len, id, log_prob) in matches {
                edges_buf.push(LatticeEdge {
                    start,
                    end: start + len,
                    token_id: id,
                    log_prob,
                });
            }
        }

        // Forward pass: compute alpha
        alpha.clear();
        alpha.resize(n + 1, f64::NEG_INFINITY);
        alpha[0] = 0.0;

        for edge in edges_buf.iter() {
            if alpha[edge.start] != f64::NEG_INFINITY {
                let term = alpha[edge.start] + edge.log_prob;
                alpha[edge.end] = logsumexp2(alpha[edge.end], term);
            }
        }

        let total_log_prob = alpha[n];
        if total_log_prob == f64::NEG_INFINITY {
            return (f64::NEG_INFINITY, Vec::new());
        }

        // Backward pass: compute beta
        beta.clear();
        beta.resize(n + 1, f64::NEG_INFINITY);
        beta[n] = 0.0;

        for edge in edges_buf.iter().rev() {
            if beta[edge.end] != f64::NEG_INFINITY {
                let term = edge.log_prob + beta[edge.end];
                beta[edge.start] = logsumexp2(beta[edge.start], term);
            }
        }

        // Expected count accumulation
        let mut expected = Vec::with_capacity(edges_buf.len());
        for edge in edges_buf.iter() {
            if alpha[edge.start] != f64::NEG_INFINITY && beta[edge.end] != f64::NEG_INFINITY {
                let posterior_log = alpha[edge.start] + edge.log_prob + beta[edge.end] - total_log_prob;
                let post = posterior_log.exp();
                if post > 1e-12 {
                    expected.push((edge.token_id, post));
                }
            }
        }

        (total_log_prob, expected)
    }

    /// Parallel EM step across corpus words.
    pub fn em_step(
        corpus: &[(Vec<u8>, u64)],
        trie: &CompactTrie,
        vocab_size: usize,
    ) -> (f64, Vec<f64>) {
        let (total_corpus_ll, expected_counts) =
            accumulate_over_corpus(corpus, vocab_size, |bytes, w_cnt, ll, counts| {
                let mut edges = Vec::with_capacity(bytes.len() * 4);
                let mut alpha = Vec::with_capacity(bytes.len() + 1);
                let mut beta = Vec::with_capacity(bytes.len() + 1);
                let (word_ll, word_expected) =
                    Self::forward_backward_word(bytes, trie, &mut edges, &mut alpha, &mut beta);
                if word_ll.is_finite() {
                    *ll += word_ll * w_cnt;
                    for (id, p) in word_expected {
                        if let Some(c) = counts.get_mut(id as usize) {
                            *c += p * w_cnt;
                        }
                    }
                }
            });

        // M-step: Laplace smoothed normalization
        let epsilon = 1e-7;
        let mut total_exp = 0.0;
        for &c in &expected_counts {
            total_exp += c + epsilon;
        }

        let ln_total = total_exp.ln();
        let new_log_probs: Vec<f64> = expected_counts
            .iter()
            .map(|&c| (c + epsilon).ln() - ln_total)
            .collect();

        (total_corpus_ll, new_log_probs)
    }
}

/// Trained Unigram Language Model tokenizer.
#[derive(Clone, Debug)]
pub struct UnigramModel {
    pub vocab: Vocabulary,
    pub log_probs: Vec<f64>,
    pub trie: CompactTrie,
    /// The segmentation encoding (and sampling) runs on: the one the training word counts were
    /// made with (`Whitespace` from `new`; `UnigramTrainer::train` sees only word counts, so its
    /// caller sets it).
    pub pretokenizer: Pretokenizer,
}

/// `x` rounded to 12 decimal places: a value every JSON reader parses back to exactly `x`.
///
/// `serde_json` writes the shortest round-trip decimal, but Hugging Face `tokenizers` is built
/// without serde_json's `float_roundtrip` feature and reads `d1..dn` with `k` decimals as
/// `(d1..dn as f64) / 10^k`, which is 1 ULP off for ~14% of 17-digit log-probs. Viterbi ties
/// between permutations of the same tokens ("~~|~" vs "~|~~") are decided by such ULPs, so HF
/// and native ids would differ. With at most 12 decimals and |x| < 9000 (every finite
/// log-probability of an f64 is > -745) the digits fit in 2^53 and 10^k is exact, so that one
/// division is correctly rounded too.
pub fn json_exact(x: f64) -> f64 {
    if x.is_finite() {
        format!("{x:.12}").parse().unwrap_or(x)
    } else {
        x
    }
}

impl UnigramModel {
    /// Scores are stored as [`json_exact`] values, so the exported tokenizer.json makes Hugging
    /// Face run Viterbi on bit-identical scores (see there).
    pub fn new(vocab: Vocabulary, log_probs: Vec<f64>) -> Self {
        let log_probs: Vec<f64> = log_probs.into_iter().map(json_exact).collect();
        let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = vocab
            .id_to_bytes_map()
            .iter()
            .enumerate()
            .map(|(i, bytes)| (bytes.clone(), i as TokenId, log_probs[i]))
            .collect();
        let trie = CompactTrie::build(&indexed_vocab);
        Self {
            vocab,
            log_probs,
            trie,
            pretokenizer: Pretokenizer::Whitespace,
        }
    }

    /// Viterbi 1-best segmentation of a byte slice into token IDs, run separately on every
    /// `pretokenizer` segment (the units the model was trained on).
    pub fn encode(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.pretokenizer
            .segment_bytes(bytes)
            .into_iter()
            .flat_map(|seg| match ViterbiDecoder::decode(seg, &self.trie, None) {
                Some((_, tokens)) => tokens,
                // Absolute fallback: raw base bytes
                None => seg.iter().map(|&b| b as TokenId).collect(),
            })
            .collect()
    }

    /// Parallel batch encoding across multiple texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<TokenId>> {
        texts
            .par_iter()
            .map(|text| self.encode(text.as_bytes()))
            .collect()
    }

    /// Decodes token IDs back to raw bytes.
    pub fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        self.vocab.decode(ids)
    }

    /// Decodes token IDs losslessly to a UTF-8 String.
    pub fn decode_lossless(&self, ids: &[TokenId]) -> Result<String> {
        self.vocab.decode_lossless(ids)
    }
}

/// Longest seed substring considered by [`UnigramTrainer::train`].
const MAX_SEED_LEN: usize = 16;

/// The `k` most frequent substrings of 2..=`MAX_SEED_LEN` bytes, counting every (overlapping)
/// occurrence in every word weighted by the word's count, ordered by (count desc, bytes asc).
///
/// Memory is one 8-byte suffix reference per corpus byte plus a `k`-entry heap, not one hash-map
/// entry per distinct substring (~140 bytes per corpus byte on real text). Suffixes are sorted by
/// their first `MAX_SEED_LEN` bytes, so all occurrences of a substring are one contiguous run of
/// suffixes; a single pass sums each run and offers it to the heap when the run ends.
fn top_substrings(corpus: &[(Vec<u8>, u64)], k: usize) -> Vec<Vec<u8>> {
    let key = |&(w, s): &(u32, u32)| {
        let word = &corpus[w as usize].0;
        &word[s as usize..word.len().min(s as usize + MAX_SEED_LEN)]
    };
    let to_u32 = |x: usize| u32::try_from(x).expect("Unigram training corpus exceeds u32 indexing");
    // Every suffix with at least 2 bytes left in its word.
    let mut suffixes: Vec<(u32, u32)> =
        Vec::with_capacity(corpus.iter().map(|(word, _)| word.len().saturating_sub(1)).sum());
    suffixes.extend(
        corpus
            .iter()
            .enumerate()
            .flat_map(|(w, (word, _))| (0..word.len().saturating_sub(1)).map(move |s| (to_u32(w), to_u32(s)))),
    );
    suffixes.sort_unstable_by(|a, b| key(a).cmp(key(b)));

    // Max-heap whose top is the worst kept substring: lowest count, then largest bytes.
    let mut heap: BinaryHeap<(Reverse<u64>, Vec<u8>)> = BinaryHeap::new();
    let mut offer = |count: u64, sub: &[u8]| {
        if heap.len() < k {
            heap.push((Reverse(count), sub.to_vec()));
        } else if let Some(mut worst) = heap.peek_mut() {
            if (Reverse(count), sub) < (worst.0, worst.1.as_slice()) {
                *worst = (Reverse(count), sub.to_vec());
            }
        }
    };
    // run[l]: summed count of the current run of suffixes that share their first l bytes.
    let mut run = [0u64; MAX_SEED_LEN + 1];
    let mut prev: &[u8] = &[];
    for suf in &suffixes {
        let cur = key(suf);
        let lcp = prev.iter().zip(cur).take_while(|(a, b)| a == b).count();
        // Runs longer than the common prefix end at `prev` (which belongs to each of them).
        for l in (lcp + 1).max(2)..=prev.len() {
            offer(run[l], &prev[..l]);
            run[l] = 0;
        }
        for count in &mut run[2..=cur.len()] {
            *count += corpus[suf.0 as usize].1;
        }
        prev = cur;
    }
    for l in 2..=prev.len() {
        offer(run[l], &prev[..l]);
    }
    heap.into_sorted_vec().into_iter().map(|(_, sub)| sub).collect()
}

/// Unigram LM Trainer implementing log-space EM and loss-based pruning with base-256 byte protection.
pub struct UnigramTrainer {
    pub target_vocab_size: usize,
    pub pruning_rate: f64,
    pub em_sub_iterations: usize,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            target_vocab_size: 16384,
            pruning_rate: 0.15,
            em_sub_iterations: 3,
        }
    }
}

impl UnigramTrainer {
    pub fn new(target_vocab_size: usize, pruning_rate: f64) -> Self {
        Self {
            target_vocab_size,
            pruning_rate: pruning_rate.clamp(0.01, 0.50),
            em_sub_iterations: 3,
        }
    }

    /// Viterbi (best-path) approximation of each token's pruning loss, NOT the exact drop in
    /// marginal likelihood: for every word whose 1-best path uses token `t`,
    /// `loss[t] += count * (log P(best path) - log P(best path that avoids t))`.
    /// Tokens absent from every 1-best path get loss 0. Single bytes are never scored.
    pub fn compute_pruning_losses(
        corpus: &[(Vec<u8>, u64)],
        trie: &CompactTrie,
        vocab: &[(Vec<u8>, TokenId, f64)],
    ) -> Vec<f64> {
        accumulate_over_corpus(corpus, vocab.len(), |bytes, w_cnt, _, losses| {
            let Some((best_score, best_path)) = ViterbiDecoder::decode(bytes, trie, None) else {
                return;
            };
            let unique_tokens: HashSet<TokenId> = best_path.into_iter().collect();
            for token_id in unique_tokens {
                let idx = token_id as usize;
                // Strict base 256 byte protection: length 1 tokens are never evaluated
                if idx >= vocab.len() || vocab[idx].0.len() == 1 {
                    continue;
                }
                losses[idx] += match ViterbiDecoder::decode(bytes, trie, Some(token_id)) {
                    Some((alt_score, _)) => (best_score - alt_score).max(0.0) * w_cnt,
                    None => 1e6 * w_cnt,
                };
            }
        })
        .1
    }

    /// Full training loop from seed vocabulary to target vocabulary size.
    ///
    /// Deterministic: the result (vocabulary, ids and scores) depends only on the multiset of
    /// `(word, count)` entries, not on their order, hash seeds or the number of threads.
    pub fn train(&self, corpus: &[(Vec<u8>, u64)]) -> Result<UnigramModel> {
        if self.target_vocab_size < BASE_VOCAB_SIZE {
            return Err(TokeniserError::VocabSizeTooSmall(self.target_vocab_size));
        }
        if corpus.is_empty() {
            return Err(TokeniserError::EmptyCorpus);
        }
        // Callers often build the corpus from a hash map; fix the order so float sums are stable.
        let mut sorted_corpus = corpus.to_vec();
        sorted_corpus.sort_unstable();
        let corpus = &sorted_corpus[..];

        // 1. Seed vocabulary: all 256 base bytes (ids 0..256), then the most frequent substrings
        // of length 2..=16, ordered by (count desc, bytes asc) so ids are reproducible.
        let seed_target = (self.target_vocab_size * 3).max(BASE_VOCAB_SIZE + 500);
        let mut current_vocab: Vec<(Vec<u8>, f64)> = (0..=255u8).map(|b| (vec![b], -12.0)).collect();
        current_vocab.extend(
            top_substrings(corpus, seed_target - BASE_VOCAB_SIZE)
                .into_iter()
                .map(|sub| (sub, -12.0)),
        );

        // 2. Iterative EM & Pruning loop
        loop {
            let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                .iter()
                .enumerate()
                .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                .collect();

            let mut trie = CompactTrie::build(&indexed_vocab);

            // Run EM sub-iterations
            for _ in 0..self.em_sub_iterations {
                let (_, new_log_probs) = UnigramEmEngine::em_step(corpus, &trie, current_vocab.len());
                for (i, &nlp) in new_log_probs.iter().enumerate() {
                    current_vocab[i].1 = nlp;
                }
                let reindexed: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                    .iter()
                    .enumerate()
                    .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                    .collect();
                trie = CompactTrie::build(&reindexed);
            }

            if current_vocab.len() <= self.target_vocab_size {
                break;
            }

            // Pruning step
            let indexed_vocab: Vec<(Vec<u8>, TokenId, f64)> = current_vocab
                .iter()
                .enumerate()
                .map(|(i, (bytes, lp))| (bytes.clone(), i as TokenId, *lp))
                .collect();
            let losses = Self::compute_pruning_losses(corpus, &trie, &indexed_vocab);

            let mut candidates: Vec<(usize, f64)> = losses
                .into_iter()
                .enumerate()
                .map(|(idx, loss)| {
                    if current_vocab[idx].0.len() == 1 {
                        (idx, f64::INFINITY) // Base byte protected
                    } else {
                        (idx, loss)
                    }
                })
                .collect();

            candidates.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| current_vocab[a.0].0.cmp(&current_vocab[b.0].0))
            });

            let current_size = current_vocab.len();
            let excess = current_size.saturating_sub(self.target_vocab_size);
            let prune_target = ((current_size as f64) * self.pruning_rate).ceil() as usize;
            let num_to_prune = prune_target.min(excess);

            let mut keep_indices = HashSet::with_capacity(current_size - num_to_prune);
            for &(idx, _) in &candidates[num_to_prune..] {
                keep_indices.insert(idx);
            }
            // Always keep protected single bytes
            for &(idx, loss) in &candidates {
                if loss.is_infinite() {
                    keep_indices.insert(idx);
                }
            }

            let mut sorted_keep: Vec<usize> = keep_indices.into_iter().collect();
            sorted_keep.sort_unstable();

            let mut next_vocab = Vec::with_capacity(sorted_keep.len());
            // Put base 256 bytes first
            for b in 0..=255u8 {
                let byte_vec = vec![b];
                if let Some(pos) = current_vocab.iter().position(|(bytes, _)| bytes == &byte_vec) {
                    next_vocab.push(current_vocab[pos].clone());
                } else {
                    next_vocab.push((byte_vec, -15.0));
                }
            }
            // Add remaining kept tokens in deterministic sorted index order
            for idx in sorted_keep {
                if current_vocab[idx].0.len() > 1 {
                    next_vocab.push(current_vocab[idx].clone());
                }
            }

            current_vocab = next_vocab;
        }

        // 3. Final model assembly
        let mut final_vocab = Vocabulary::new_byte_level();
        let mut final_log_probs = Vec::with_capacity(current_vocab.len());

        for (bytes, lp) in &current_vocab {
            if bytes.len() > 1 {
                final_vocab.add_token(bytes.clone());
            }
            final_log_probs.push(*lp);
        }

        Ok(UnigramModel::new(final_vocab, final_log_probs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The original seeding: count every substring in a hash map, sort, take `k`.
    fn top_substrings_oracle(corpus: &[(Vec<u8>, u64)], k: usize) -> Vec<Vec<u8>> {
        let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for (word, freq) in corpus {
            for start in 0..word.len() {
                for end in (start + 2)..=(start + MAX_SEED_LEN).min(word.len()) {
                    *counts.entry(word[start..end].to_vec()).or_insert(0) += freq;
                }
            }
        }
        let mut sorted: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.into_iter().take(k).map(|(sub, _)| sub).collect()
    }

    #[test]
    fn top_substrings_matches_hash_map_counting() {
        let mut rng = fastrand::Rng::with_seed(7);
        for trial in 0..300 {
            // Small alphabets (incl. 0x00 and 0xff) force long shared prefixes and many ties.
            let alphabet: &[u8] = match trial % 3 {
                0 => b"ab",
                1 => &[0x00, b'a', 0xff],
                _ => b"abcdefgh",
            };
            let corpus: Vec<(Vec<u8>, u64)> = (0..rng.usize(1..30))
                .map(|_| {
                    let word = (0..rng.usize(0..40)).map(|_| alphabet[rng.usize(..alphabet.len())]).collect();
                    (word, rng.u64(0..5))
                })
                .collect();
            for k in [0, 1, 7, 50, 10_000] {
                assert_eq!(
                    top_substrings(&corpus, k),
                    top_substrings_oracle(&corpus, k),
                    "trial {trial} k {k} corpus {corpus:?}"
                );
            }
        }
    }
}
