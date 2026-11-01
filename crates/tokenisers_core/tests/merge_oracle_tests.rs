//! The training engine must merge exactly what a brute-force recount + rescore would merge, for
//! every scorer, and encoding must reproduce the training segmentation.
mod common;

use std::collections::{BTreeMap, HashMap};

use common::{apply_leftmost, merges_in, random_corpus, reference_encode, reference_train, Merges};
use tokenisers_core::curricula::morpheme::{MorphemeGuidedScorer, MorphemeLexicon};
use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::index::Pair;
use tokenisers_core::pretokenizer::Pretokenizer;
use tokenisers_core::scoring::{
    BpeScorer, LengthPenalizedScorer, MdlScorer, PairContext, PairScorer, ScoreDeps, ScoredPair, WordPieceScorer,
    WpBpeBaseScorer, WpBpeFtScorer, WpBpeNpmiScorer, WpBpePigScorer, WpBpePpigScorer, WpBpeScpScorer,
};
use tokenisers_core::vocab::{TokenId, Vocabulary};

/// Reads every context field in a non-monotone way and keeps the default (`Arbitrary`) deps.
#[derive(Debug)]
struct OddScorer;

impl PairScorer for OddScorer {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        let wobble = ((ctx.total_tokens + ctx.active_vocab_size as u64 + pair.0 as u64) % 5) as f64;
        ctx.freq_uv as f64 * (1.0 + wobble) - (ctx.freq_u as f64).sqrt()
    }
    fn name(&self) -> &'static str {
        "odd"
    }
}

/// Monotone in N up to a rounding-sized wobble (< 1e-12), like float noise in a real
/// `MonotoneTotals` scorer. The engine's tolerance band must absorb it.
#[derive(Debug)]
struct NoisyMonotoneScorer;

impl PairScorer for NoisyMonotoneScorer {
    fn score(&self, pair: Pair, ctx: &PairContext) -> f64 {
        let wobble = ((ctx.total_tokens * 7919 + pair.0 as u64 * 31 + pair.1 as u64) % 5) as f64;
        ctx.freq_uv as f64 + wobble * 1e-13
    }
    fn name(&self) -> &'static str {
        "noisy_monotone"
    }
    fn deps(&self) -> ScoreDeps {
        ScoreDeps::MonotoneTotals
    }
}

fn scorers() -> Vec<(&'static str, Box<dyn PairScorer>, u64)> {
    let mut lexicon = MorphemeLexicon::new();
    for (word, morphemes) in [("aab", ["a", "ab"]), ("abab", ["ab", "ab"]), ("aaa", ["a", "aa"]), ("abc", ["ab", "c"])] {
        let parts: Vec<&[u8]> = morphemes.iter().map(|m| m.as_bytes()).collect();
        assert!(lexicon.insert_segmentation(word.as_bytes(), &parts));
    }
    let morpheme = MorphemeGuidedScorer::new(WpBpePigScorer::new(1), lexicon, 2.0, 1.0);
    vec![
        ("bpe", Box::new(BpeScorer::new()), 1),
        ("bpe_min2", Box::new(BpeScorer::new()), 2),
        ("length_penalized", Box::new(LengthPenalizedScorer::new(0.5, 1)), 1),
        ("length_rewarded", Box::new(LengthPenalizedScorer::new(-1.0, 1)), 1),
        ("wordpiece", Box::new(WordPieceScorer::new(1.0, 1)), 1),
        ("wordpiece_unsmoothed", Box::new(WordPieceScorer::new(0.0, 1)), 1),
        ("wp_bpe_base", Box::new(WpBpeBaseScorer::new()), 1),
        ("wp_bpe_ft", Box::new(WpBpeFtScorer::new(2)), 2),
        ("wp_bpe_pig", Box::new(WpBpePigScorer::new(1)), 1),
        ("wp_bpe_scp", Box::new(WpBpeScpScorer::new(2, 0.75)), 2),
        ("wp_bpe_npmi", Box::new(WpBpeNpmiScorer::new(2, 0.5)), 2),
        ("wp_bpe_npmi_min1", Box::new(WpBpeNpmiScorer::new(1, 1.0)), 1),
        ("wp_bpe_ppig", Box::new(WpBpePpigScorer::new(2, 0.35)), 2),
        ("mdl", Box::new(MdlScorer::new(24.0, 1)), 1),
        ("mdl_free", Box::new(MdlScorer::new(0.0, 2)), 2),
        ("morpheme_pig", Box::new(morpheme), 1),
        ("noisy_monotone", Box::new(NoisyMonotoneScorer), 1),
        ("arbitrary_deps", Box::new(OddScorer), 1),
    ]
}

#[test]
fn engine_matches_bruteforce_argmax_for_every_scorer() {
    const TRIALS: u64 = 300;
    for (name, scorer, min_freq) in scorers() {
        let mut merged = 0;
        for seed in 0..TRIALS {
            let mut rng = fastrand::Rng::with_seed(seed);
            let alphabet: &[u8] = [&b"ab"[..], b"abc", b"abcd"][(seed % 3) as usize];
            let corpus = if name.starts_with("mdl") {
                // MDL stops once merges stop paying for their entry, which on i.i.d. letters is
                // almost at once: build words from a few repeated chunks so there is structure.
                let chunks: Vec<Vec<u8>> = (0..4).map(|_| random_corpus(&mut rng, alphabet, 1, 4)[0].0.clone()).collect();
                let mut counts = std::collections::BTreeMap::new();
                for _ in 0..30 {
                    let word: Vec<u8> = (0..rng.usize(1..4)).flat_map(|_| chunks[rng.usize(0..4)].clone()).collect();
                    *counts.entry(word).or_insert(0u64) += rng.u64(1..6) * 20;
                }
                counts.into_iter().collect()
            } else {
                random_corpus(&mut rng, alphabet, 30, 10)
            };
            let target = 256 + 30;

            let model = BottomUpTrainer::new(scorer.as_ref(), target)
                .with_min_frequency(min_freq)
                .train(&corpus)
                .unwrap();
            let (ref_merges, ref_words) = reference_train(scorer.as_ref(), &corpus, target, min_freq);

            assert_eq!(model.merges, ref_merges, "{name}: merge sequence differs from brute force (seed {seed})");
            assert!(model.pretokenize, "{name}: single-segment training words must give a segmented model");

            // Encoding a training word reproduces its training segmentation.
            let words = corpus.iter().filter(|(w, f)| !w.is_empty() && *f > 0);
            for ((word, _), ref_seg) in words.zip(&ref_words) {
                assert_eq!(&model.encode(word), ref_seg, "{name}: encode != training segmentation (seed {seed})");
            }
            merged += model.merges.len();
        }
        assert!(merged as u64 >= 3 * TRIALS, "{name}: only {merged} merges in {TRIALS} trials");
    }
}

#[test]
fn space_prefix_training_matches_bruteforce_and_encodes_text_like_training() {
    // Word counts in space_prefix segments (" ab", "\t", ...): merges must equal brute force, and
    // encoding whole lines must give each segment's training segmentation, which needs the
    // encoder to pre-split exactly like training (a whitespace-run split would cut " ab").
    let mut spaced_merges = 0;
    for (name, scorer, min_freq) in scorers() {
        for seed in 0..60u64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let alphabet: &[u8] = [&b"ab  "[..], b"abc  \t", b"ab \n"][(seed % 3) as usize];
            let lines: Vec<Vec<u8>> = (0..15)
                .map(|_| (0..rng.usize(1..20)).map(|_| alphabet[rng.usize(0..alphabet.len())]).collect())
                .collect();
            let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
            for line in &lines {
                for seg in Pretokenizer::SpacePrefix.segment_bytes(line) {
                    *counts.entry(seg.to_vec()).or_default() += 1;
                }
            }
            let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
            let target = 256 + 25;
            let model = BottomUpTrainer::new(scorer.as_ref(), target)
                .with_min_frequency(min_freq)
                .with_pretokenizer(Pretokenizer::SpacePrefix)
                .train(&corpus)
                .unwrap();
            let (ref_merges, ref_words) = reference_train(scorer.as_ref(), &corpus, target, min_freq);
            assert_eq!(model.merges, ref_merges, "{name}: merge sequence differs from brute force (seed {seed})");
            assert!(model.pretokenize && model.pretokenizer == Pretokenizer::SpacePrefix, "{name} (seed {seed})");

            let trained: HashMap<&[u8], &Vec<TokenId>> =
                corpus.iter().map(|(w, _)| w.as_slice()).zip(&ref_words).collect();
            for line in &lines {
                let expected: Vec<TokenId> = Pretokenizer::SpacePrefix
                    .segment_bytes(line)
                    .into_iter()
                    .flat_map(|seg| trained[seg].iter().copied())
                    .collect();
                assert_eq!(model.encode(line), expected, "{name}: encode != training (seed {seed}, {line:?})");
            }
            spaced_merges += model.merges.iter().filter(|&&(_, _, w)| {
                let b = model.vocab.get_bytes(w).unwrap();
                b[0] == b' ' && b.iter().any(|c| !c.is_ascii_whitespace())
            }).count();
        }
    }
    assert!(spaced_merges > 500, "space + word merges must be exercised: {spaced_merges}");
}

#[test]
fn every_builtin_scorer_declares_narrow_deps() {
    for (name, scorer, _) in scorers() {
        let expected_arbitrary = name == "arbitrary_deps";
        assert_eq!(scorer.deps() == ScoreDeps::Arbitrary, expected_arbitrary, "{name}");
    }
}

#[test]
fn encode_matches_rank_replay_on_unseen_text() {
    for seed in 0..200u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let alphabet: &[u8] = if seed % 2 == 0 { b"ab" } else { b"abc" };
        let corpus = random_corpus(&mut rng, alphabet, 25, 12);
        let scorer = WpBpePigScorer::new(1);
        let model = BottomUpTrainer::new(&scorer, 256 + 40).train(&corpus).unwrap();
        let whole = BottomUpModel::new(model.vocab.clone(), model.merges.clone());
        assert!(!whole.pretokenize);

        for _ in 0..10 {
            let len = rng.usize(0..60);
            let text: Vec<u8> = (0..len)
                .map(|_| if rng.u8(0..6) == 0 { b' ' } else { alphabet[rng.usize(0..alphabet.len())] })
                .collect();
            let expected = reference_encode(&model.merges, &text);
            assert_eq!(model.encode(&text), expected, "segmented encode (seed {seed})");
            assert_eq!(whole.encode(&text), expected, "whole-text encode (seed {seed})");
            assert_eq!(model.decode(&expected).unwrap(), text);
        }
    }
}

#[test]
fn encode_matches_rank_replay_for_arbitrary_merge_lists() {
    // Random merge lists contain duplicate tokens ("abc" from (ab,c) and (a,bc)) and pairs merged
    // at several ranks; the heap encoder must still equal replaying every rank in order.
    let mut repeated_pairs = 0;
    for seed in 0..300u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut vocab = tokenisers_core::vocab::Vocabulary::new_byte_level();
        let mut ids: Vec<u32> = b"abc".iter().map(|&b| b as u32).collect();
        let mut merges = Vec::new();
        for _ in 0..40 {
            let (u, v) = (ids[rng.usize(0..ids.len())], ids[rng.usize(0..ids.len())]);
            let mut bytes = vocab.get_bytes(u).unwrap().to_vec();
            bytes.extend_from_slice(vocab.get_bytes(v).unwrap());
            if bytes.len() > 6 {
                continue;
            }
            let w = vocab.add_token(bytes);
            if !ids.contains(&w) {
                ids.push(w);
            }
            merges.push((u, v, w));
        }
        let distinct: std::collections::HashSet<_> = merges.iter().map(|&(u, v, _)| (u, v)).collect();
        repeated_pairs += merges.len() - distinct.len();
        let model = BottomUpModel::new(vocab, merges);
        for _ in 0..20 {
            let text: Vec<u8> = (0..rng.usize(0..40)).map(|_| b"abc"[rng.usize(0..3)]).collect();
            assert_eq!(model.encode(&text), reference_encode(&model.merges, &text), "seed {seed}");
        }
    }
    assert!(repeated_pairs > 0, "generator must exercise repeated pairs");
}

#[test]
fn overlapping_self_pairs_are_counted_and_merged_leftmost() {
    // "aaa" holds two overlapping (a, a) positions but only one merge applies.
    let corpus = vec![(b"aaa".to_vec(), 3u64), (b"aaaa".to_vec(), 1), (b"abab".to_vec(), 2)];
    for (name, scorer, min_freq) in scorers() {
        let model = BottomUpTrainer::new(scorer.as_ref(), 262)
            .with_min_frequency(min_freq)
            .train(&corpus)
            .unwrap();
        let (ref_merges, _) = reference_train(scorer.as_ref(), &corpus, 262, min_freq);
        assert_eq!(model.merges, ref_merges, "{name}");
    }
    let model = BottomUpTrainer::new(&BpeScorer, 257).train(&corpus).unwrap();
    let aa = model.merges[0].2;
    assert_eq!(model.merges[0].0, b'a' as u32);
    assert_eq!(model.encode(b"aaa"), vec![aa, b'a' as u32]);
}

#[test]
fn wordpiece_regression_from_review() {
    // Review repro: after merging (a, b), f(a) drops to 2, so (a, c) = 2/9 must beat (d, e).
    let corpus = vec![
        (b"ab".to_vec(), 10u64),
        (b"ac".to_vec(), 2),
        (b"de".to_vec(), 3),
        (b"d".to_vec(), 3),
        (b"e".to_vec(), 3),
    ];
    let scorer = WordPieceScorer::new(1.0, 1);
    let model = BottomUpTrainer::new(&scorer, 258).train(&corpus).unwrap();
    let pairs: Vec<(u8, u8)> = model.merges.iter().map(|&(u, v, _)| (u as u8, v as u8)).collect();
    assert_eq!(pairs, vec![(b'a', b'b'), (b'a', b'c')]);
}

#[test]
fn training_reaches_target_size_when_pairs_remain() {
    let text = common::pseudo_text(7, 20_000);
    let corpus: Vec<(Vec<u8>, u64)> = {
        let mut counts = std::collections::BTreeMap::new();
        for seg in tokenisers_core::SuperBpeTrainer::segment_text(&text) {
            *counts.entry(seg).or_insert(0u64) += 1;
        }
        counts.into_iter().collect()
    };
    for (name, scorer, min_freq) in scorers() {
        let model = BottomUpTrainer::new(scorer.as_ref(), 400)
            .with_min_frequency(min_freq)
            .train(&corpus)
            .unwrap();
        if name.starts_with("mdl") {
            // MDL stops once no merge shortens the description: exactly where brute force stops.
            let (ref_merges, _) = reference_train(scorer.as_ref(), &corpus, 400, min_freq);
            assert_eq!(model.merges, ref_merges, "{name}");
            continue;
        }
        assert_eq!(model.vocab.len(), 400, "{name} stopped early");
    }
}

fn pseudo_corpus(seed: u64, n_bytes: usize) -> Vec<(Vec<u8>, u64)> {
    let mut counts = BTreeMap::new();
    for seg in SuperBpeTrainer::segment_text(&common::pseudo_text(seed, n_bytes)) {
        *counts.entry(seg).or_insert(0u64) += 1;
    }
    counts.into_iter().collect()
}

#[test]
fn mdl_entry_cost_controls_where_training_stops() {
    // Review finding: entry_cost 0 / 24 / 1e6 used to give identical vocabularies, because a
    // constant offset never changes the argmax. With MDL stopping a costlier entry stops earlier,
    // and since the order is unchanged each run is a prefix of the cheaper one.
    let corpus = pseudo_corpus(7, 20_000);
    let train = |cost: f64| BottomUpTrainer::new(&MdlScorer::new(cost, 1), 3000).train(&corpus).unwrap().merges;
    let runs: Vec<Merges> = [0.0, 24.0, 500.0, 1e6].into_iter().map(train).collect();
    let sizes: Vec<usize> = runs.iter().map(Vec::len).collect();
    assert!(sizes[0] < 3000 - 256, "MDL must stop on its own before the cap: {sizes:?}");
    assert!(sizes[0] > sizes[1] && sizes[1] > sizes[2] && sizes[2] > 0, "{sizes:?}");
    assert_eq!(sizes[3], 0, "no merge pays for a 1e6-bit entry");
    for w in runs.windows(2) {
        assert_eq!(w[1][..], w[0][..w[1].len()], "entry cost must not reorder merges");
    }
}

/// Brute-force two-stage SuperBPE: every step recounts pairs, token counts and N over the current
/// merge units and merges the best `scorer.score`. Units are segments in stage 1 and whole lines
/// in stage 2. Returns the merges, whether stage 2 started, and each line's final tokens.
fn reference_super_bpe(
    scorer: &dyn PairScorer,
    lines: &[&str],
    config: &SuperBpeConfig,
    min_frequency: u64,
) -> (Merges, bool, Vec<Vec<TokenId>>) {
    let to_ids = |b: &[u8]| b.iter().map(|&x| x as TokenId).collect::<Vec<_>>();
    let mut units: Vec<Vec<Vec<TokenId>>> = lines
        .iter()
        .map(|l| SuperBpeTrainer::segment_text(l).iter().map(|s| to_ids(s)).collect())
        .collect();
    let mut vocab = Vocabulary::new_byte_level();
    let mut merges = Vec::new();
    let mut stage2 = false;
    while vocab.len() < config.v_final {
        if !stage2 && vocab.len() >= config.v_switch {
            stage2 = true;
            units = units.into_iter().map(|segs| vec![segs.concat()]).collect();
        }
        let mut pairs: BTreeMap<Pair, u64> = BTreeMap::new();
        let mut tokens: HashMap<TokenId, u64> = HashMap::new();
        for seq in units.iter().flatten() {
            for &t in seq {
                *tokens.entry(t).or_default() += 1;
            }
            for w in seq.windows(2) {
                *pairs.entry(Pair::new(w[0], w[1])).or_default() += 1;
            }
        }
        let total = tokens.values().sum();
        let best = pairs
            .iter()
            .filter(|&(_, &c)| c >= min_frequency.max(1))
            .filter_map(|(&pair, &count)| {
                let merges = if pair.0 == pair.1 { units.iter().flatten().map(|seq| merges_in(seq, pair)).sum() } else { count };
                let ctx = PairContext {
                    freq_uv: count,
                    merges_uv: merges,
                    freq_u: tokens[&pair.0],
                    freq_v: tokens[&pair.1],
                    total_tokens: total,
                    active_vocab_size: vocab.len(),
                    len_u: vocab.get_bytes(pair.0).unwrap().len(),
                    len_v: vocab.get_bytes(pair.1).unwrap().len(),
                };
                let score = scorer.score(pair, &ctx);
                score.is_finite().then_some(ScoredPair { score, pair, version: 0 })
            })
            .max();
        let Some(best) = best else {
            if stage2 {
                break;
            }
            stage2 = true;
            units = units.into_iter().map(|segs| vec![segs.concat()]).collect();
            continue;
        };
        let (u, v) = (best.pair.0, best.pair.1);
        let mut bytes = vocab.get_bytes(u).unwrap().to_vec();
        bytes.extend_from_slice(vocab.get_bytes(v).unwrap());
        let w = vocab.add_token(bytes);
        merges.push((u, v, w));
        for seq in units.iter_mut().flatten() {
            *seq = apply_leftmost(seq, u, v, w);
        }
    }
    (merges, stage2, units.into_iter().map(|segs| segs.concat()).collect())
}

#[test]
fn super_bpe_matches_bruteforce_argmax_for_every_scorer() {
    // Review finding: SuperBpeTrainer re-scored only pairs whose count changed, so marginal- or
    // N-dependent scorers merged on stale scores. It now runs the exact BottomUpTrainer engine.
    let mut cross_merges = 0;
    for (name, scorer, min_freq) in scorers() {
        for seed in 0..100u64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let alphabet: &[u8] = [&b"ab "[..], b"abc ", b"abcd  "][(seed % 3) as usize];
            let texts: Vec<String> = (0..12)
                .map(|_| (0..rng.usize(1..24)).map(|_| alphabet[rng.usize(0..alphabet.len())] as char).collect())
                .collect();
            let lines: Vec<&str> = texts.iter().map(String::as_str).collect();
            let config = SuperBpeConfig::new(256 + 8, 256 + 25).unwrap();
            let mut trainer = SuperBpeTrainer::new(config.clone(), scorer.as_ref());
            trainer.min_frequency = min_freq;
            let model = trainer.train(&lines).unwrap();
            let (ref_merges, ref_stage2, ref_lines) = reference_super_bpe(scorer.as_ref(), &lines, &config, min_freq);

            assert_eq!(model.merges, ref_merges, "{name}: merge sequence differs from brute force (seed {seed})");
            assert_eq!(model.pretokenize, !ref_stage2, "{name} (seed {seed})");
            for (line, ref_tokens) in lines.iter().zip(&ref_lines) {
                assert_eq!(&model.encode(line.as_bytes()), ref_tokens, "{name}: encode != training (seed {seed})");
            }
            cross_merges += model.merges.iter().filter(|&&(_, _, w)| {
                let b = model.vocab.get_bytes(w).unwrap();
                b.contains(&b' ') && b.iter().any(|&c| c != b' ')
            }).count();
        }
    }
    assert!(cross_merges > 100, "stage 2 must learn cross-whitespace tokens: {cross_merges}");
}

#[test]
fn super_bpe_wordpiece_regression_from_review() {
    // Same repro as `wordpiece_regression_from_review`, through SuperBpeTrainer.
    let mut lines = vec!["ab"; 10];
    lines.extend(["ac"; 2]);
    lines.extend(["de"; 3]);
    lines.extend(["d"; 3]);
    lines.extend(["e"; 3]);
    let scorer = WordPieceScorer::new(1.0, 1);
    let model = SuperBpeTrainer::new(SuperBpeConfig::new(258, 258).unwrap(), &scorer).train(&lines).unwrap();
    let pairs: Vec<(u8, u8)> = model.merges.iter().map(|&(u, v, _)| (u as u8, v as u8)).collect();
    assert_eq!(pairs, vec![(b'a', b'b'), (b'a', b'c')]);
}
