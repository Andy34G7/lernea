use fastrand::Rng;

use crate::engine::unigram::{CompactTrie, logsumexp2};
use crate::pretokenizer::Pretokenizer;
use crate::vocab::TokenId;

/// Upper bound on the k-best path entries (24 bytes each, so about 24 MB) that
/// [`UnigramSampler::sample_kbest`] holds at once for one segment.
const MAX_KBEST_ENTRIES: usize = 1 << 20;

/// An edge in the subword lattice.
#[derive(Debug, Clone, Copy)]
pub struct LatticeEdge {
    pub start_pos: usize,
    pub end_pos: usize,
    pub token_id: TokenId,
    pub log_prob: f64,
}

/// Stochastic Unigram Lattice Sampler supporting exact FFBS and k-best path sampling.
///
/// Temperature semantics (both samplers): a segmentation `s` is weighted by
/// `P(s)^(1/temperature) = prod_{t in s} p(t)^(1/temperature)`. `temperature = 1` samples the
/// model's exact posterior; SentencePiece's smoothing parameter is `alpha = 1/temperature`
/// (alpha = 0.1 <=> temperature = 10); `temperature -> 0` approaches Viterbi. No hidden scaling.
/// Like `UnigramModel::encode`, the lattice never spans a `pretokenizer` segment boundary.
pub struct UnigramSampler<'a> {
    pub trie: &'a CompactTrie,
    /// The model's segmentation (`UnigramModel::pretokenizer`); `new` uses `Whitespace`.
    pub pretokenizer: Pretokenizer,
}

impl<'a> UnigramSampler<'a> {
    pub fn new(trie: &'a CompactTrie) -> Self {
        Self { trie, pretokenizer: Pretokenizer::Whitespace }
    }

    /// Positions processed together: the lattice and the k-best rows are built one block at a
    /// time, so their memory grows with `sqrt(len)` instead of `len`. A block is at least
    /// `max_token_len` long, so no token skips a whole block.
    fn block_len(&self, len: usize) -> usize {
        let l = self.trie.max_token_len.max(1);
        (((len + 1) as f64 * l as f64).sqrt() as usize).max(l)
    }

    /// Incoming lattice edges of end positions `from..to` of one segment (`from >= 1`; entry
    /// `i - from` is position `i`), in ascending start order. A position no token ends at gets a
    /// byte fallback edge from the previous position, so every position is reachable.
    fn incoming_window(&self, seg: &[u8], from: usize, to: usize) -> Vec<Vec<LatticeEdge>> {
        let mut incoming = vec![Vec::new(); to - from];
        for start in from.saturating_sub(self.trie.max_token_len)..to - 1 {
            for (len, token_id, log_prob) in self.trie.common_prefix_search(&seg[start..]) {
                let end = start + len;
                if end >= to {
                    break;
                }
                if end >= from {
                    let edge = LatticeEdge { start_pos: start, end_pos: end, token_id, log_prob };
                    incoming[end - from].push(edge);
                }
            }
        }
        for (i, edges) in (from..to).zip(&mut incoming) {
            if edges.is_empty() {
                let token_id = seg[i - 1] as TokenId;
                edges.push(LatticeEdge { start_pos: i - 1, end_pos: i, token_id, log_prob: -15.0 });
            }
        }
        incoming
    }

    /// Forward-Filtering Backward-Sampling: an exact sample from `P(s)^(1/temperature)`
    /// (normalised over all segmentations). `seed = None` draws a fresh random seed.
    ///
    /// The lattice never spans a `pretokenizer` segment boundary, so `P(s)` factorises over segments
    /// and sampling each segment on its own draws from exactly the same distribution. Memory
    /// grows with the longest segment, not the whole document: an 8-byte forward table per byte
    /// of it, plus one block of the lattice (see `block_len`), which is never built whole.
    pub fn sample_ffbs(&self, bytes: &[u8], temperature: f64, seed: Option<u64>) -> Vec<TokenId> {
        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }
        let mut tokens = Vec::with_capacity(bytes.len());
        for seg in self.pretokenizer.segment_bytes(bytes) {
            tokens.extend(self.sample_ffbs_segment(seg, temperature, &mut rng));
        }
        tokens
    }

    /// FFBS over one segment (`seg` must not span a `pretokenizer` segment boundary).
    fn sample_ffbs_segment(&self, seg: &[u8], temperature: f64, rng: &mut Rng) -> Vec<TokenId> {
        let n = seg.len();
        let block = self.block_len(n);
        let inv_temp = 1.0 / temperature.max(1e-4);

        // Forward Pass: compute log_alpha(i)
        let mut log_alpha = vec![f64::NEG_INFINITY; n + 1];
        log_alpha[0] = 0.0;

        for from in (1..=n).step_by(block) {
            let to = (from + block).min(n + 1);
            for (i, edges) in (from..to).zip(self.incoming_window(seg, from, to)) {
                for edge in &edges {
                    let prev_alpha = log_alpha[edge.start_pos];
                    if prev_alpha > f64::NEG_INFINITY {
                        let weight = prev_alpha + inv_temp * edge.log_prob;
                        log_alpha[i] = logsumexp2(log_alpha[i], weight);
                    }
                }
            }
        }

        // Backward Sampling Pass, rebuilding the lattice one block at a time.
        let mut curr_pos = n;
        let mut tokens = Vec::new();
        let (mut from, mut incoming) = (n + 1, Vec::new());

        while curr_pos > 0 {
            if curr_pos < from {
                from = (curr_pos + 1).saturating_sub(block).max(1);
                incoming = self.incoming_window(seg, from, curr_pos + 1);
            }
            let edges = &incoming[curr_pos - from];
            let mut probs = Vec::with_capacity(edges.len());

            for edge in edges {
                let prev_alpha = log_alpha[edge.start_pos];
                let logit = prev_alpha + inv_temp * edge.log_prob - log_alpha[curr_pos];
                probs.push(logit.exp().max(0.0));
            }

            let sum_prob: f64 = probs.iter().sum();
            let mut choice_val = if sum_prob > 0.0 {
                rng.f64() * sum_prob
            } else {
                0.0
            };

            // Default to the last reachable edge if rounding leaves choice_val slightly > 0.
            let mut selected_idx = probs.iter().rposition(|&p| p > 0.0).unwrap_or(0);
            for (idx, &p) in probs.iter().enumerate() {
                choice_val -= p;
                if choice_val <= 0.0 {
                    selected_idx = idx;
                    break;
                }
            }

            let chosen_edge = &edges[selected_idx];
            tokens.push(chosen_edge.token_id);
            curr_pos = chosen_edge.start_pos;
        }

        tokens.reverse();
        tokens
    }

    /// Samples one of the `k` highest-scoring segmentations (multi-backpointer Viterbi) with
    /// probability proportional to `P(s)^(1/temperature)` renormalised over those `k`
    /// (SentencePiece `nbest_size = k`). `k = 1` is deterministic Viterbi.
    ///
    /// The k-best choice is made per `pretokenizer` segment (word or whitespace run), like a
    /// k-best encoder applied to each pre-token: the lattice never spans segments, so they are
    /// independent, and each one draws from its own top `k`.
    ///
    /// Memory: the forward pass keeps the k-best rows of only the last `max_token_len`
    /// positions, saves them at the start of every block of `B = sqrt(len * max_token_len)`
    /// positions, and the trace-back recomputes one block at a time from those checkpoints
    /// (twice the forward work). A segment of `len` bytes therefore holds about
    /// `2 * k * sqrt(len * max_token_len)` path entries instead of `k * len`. `k` is capped so
    /// that stays within `MAX_KBEST_ENTRIES` (about 24 MB): `k = 64` is kept in full up to a
    /// 4 MB segment at `max_token_len = 16` (16 MB at 4); past that `k` shrinks as
    /// `1 / sqrt(len)` (still about 30 at 20 MB) instead of exhausting memory.
    pub fn sample_kbest(&self, bytes: &[u8], k: usize, temperature: f64, seed: Option<u64>) -> Vec<TokenId> {
        let mut rng = Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }
        let mut tokens = Vec::with_capacity(bytes.len());
        for seg in self.pretokenizer.segment_bytes(bytes) {
            tokens.extend(self.sample_kbest_segment(seg, k, temperature, &mut rng));
        }
        tokens
    }

    /// Appends the top-`k` rows of positions `from..to` to `rows`, whose last row is position
    /// `from - 1` and which holds at least the `max_token_len` rows before `from`.
    fn push_kbest_rows(&self, seg: &[u8], k: usize, from: usize, to: usize, rows: &mut Vec<Vec<PathEntry>>) {
        let base = from - rows.len();
        let mut cands = Vec::new();
        for edges in self.incoming_window(seg, from, to) {
            cands.clear();
            for edge in &edges {
                for (prev_rank, prev_entry) in rows[edge.start_pos - base].iter().enumerate() {
                    cands.push(PathEntry {
                        score: prev_entry.score + edge.log_prob,
                        prev_pos: edge.start_pos,
                        token_id: edge.token_id,
                        prev_rank: prev_rank as u32,
                    });
                }
            }
            cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            // Copy out exactly the top k: `cands` holds up to (incoming edges) * k entries.
            rows.push(cands[..cands.len().min(k)].to_vec());
        }
    }

    /// k-best sampling over one segment (`seg` must not span a `pretokenizer` segment boundary).
    fn sample_kbest_segment(&self, seg: &[u8], k: usize, temperature: f64, rng: &mut Rng) -> Vec<TokenId> {
        let n = seg.len();
        let l = self.trie.max_token_len.max(1);
        let block = self.block_len(n);
        let block_starts: Vec<usize> = (1..=n).step_by(block).collect();
        let block_end = |from: usize| (from + block).min(n + 1);
        // Rows held at once: `l` per checkpoint, plus the rows of one block and its context.
        // ponytail: k still shrinks past ~4 MB segments; Eppstein's k-shortest-paths algorithm
        // would keep the full k if that ever matters.
        let k = k.clamp(1, (MAX_KBEST_ENTRIES / ((block_starts.len() + 1) * l + block)).max(1));

        // Forward pass. `rows` ends at the last position computed; `checkpoints[j]` holds the
        // `l` rows before block j, enough to recompute it.
        let mut rows = vec![vec![PathEntry { score: 0.0, prev_pos: 0, token_id: 0, prev_rank: 0 }]];
        let mut checkpoints = Vec::with_capacity(block_starts.len());
        for &from in &block_starts {
            rows.drain(..rows.len().saturating_sub(l));
            checkpoints.push(rows.clone());
            self.push_kbest_rows(seg, k, from, block_end(from), &mut rows);
        }

        let terminal_paths = rows.last().expect("row 0 always exists");
        let chosen_rank = if k == 1 {
            0 // Strictly deterministic Viterbi 1-best
        } else {
            let inv_temp = 1.0 / temperature.max(1e-4);
            let max_score = terminal_paths[0].score;
            let unnorm_probs: Vec<f64> = terminal_paths
                .iter()
                .map(|p| (inv_temp * (p.score - max_score)).exp())
                .collect();
            let sum_p: f64 = unnorm_probs.iter().sum();

            let mut r = rng.f64() * sum_p;
            let mut chosen_rank = unnorm_probs.len() - 1;
            for (idx, &p) in unnorm_probs.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    chosen_rank = idx;
                    break;
                }
            }
            chosen_rank
        };

        // Trace back block by block; the last block's rows are still in `rows`.
        let mut tokens = Vec::new();
        let (mut curr_pos, mut curr_rank) = (n, chosen_rank);
        for (j, &from) in block_starts.iter().enumerate().rev() {
            if j + 1 < block_starts.len() {
                rows = std::mem::take(&mut checkpoints[j]);
                self.push_kbest_rows(seg, k, from, block_end(from), &mut rows);
            }
            let base = block_end(from) - rows.len();
            while curr_pos >= from {
                let entry = &rows[curr_pos - base][curr_rank];
                tokens.push(entry.token_id);
                curr_pos = entry.prev_pos;
                curr_rank = entry.prev_rank as usize;
            }
        }

        tokens.reverse();
        tokens
    }
}

/// A k-best table entry: a path to this position (`prev_rank` < k <= `MAX_KBEST_ENTRIES`).
#[derive(Clone, Debug)]
struct PathEntry {
    score: f64,
    prev_pos: usize,
    token_id: TokenId,
    prev_rank: u32,
}
