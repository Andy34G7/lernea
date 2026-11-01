//! `Pretokenizer` segmentation: the default `Whitespace` mode is the old maximal-run split, and
//! `SpacePrefix` (GPT-2-style) lets a word take the one U+0020 space before it.
mod common;

use std::collections::BTreeMap;

use tokenisers_core::curricula::super_bpe::SuperBpeTrainer;
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::engine::unigram::{CompactTrie, UnigramTrainer};
use tokenisers_core::inference::unigram_sample::UnigramSampler;
use tokenisers_core::pretokenizer::Pretokenizer::{self, SpacePrefix, Whitespace};
use tokenisers_core::scoring::BpeScorer;

fn segs(p: Pretokenizer, bytes: &[u8]) -> Vec<Vec<u8>> {
    p.segment_bytes(bytes).into_iter().map(<[u8]>::to_vec).collect()
}

fn strs(p: Pretokenizer, text: &str) -> Vec<String> {
    p.segment_text(text).into_iter().map(|s| String::from_utf8(s).unwrap()).collect()
}

#[test]
fn space_prefix_edge_cases() {
    let cases: &[(&str, &[&str])] = &[
        ("", &[]),
        ("hello", &["hello"]),
        ("the cat", &["the", " cat"]),
        (" lead", &[" lead"]),
        ("  lead", &[" ", " lead"]),
        ("trail ", &["trail", " "]),
        ("trail  ", &["trail", "  "]),
        (" ", &[" "]),
        ("   ", &["   "]),
        ("a  b", &["a", " ", " b"]),
        ("a   b", &["a", "  ", " b"]),
        ("a \nb", &["a", " \n", "b"]),
        ("a \n b", &["a", " \n", " b"]),
        ("a\n\n b", &["a", "\n\n", " b"]),
        ("a\tb", &["a", "\t", "b"]),
        ("a \tb", &["a", " \t", "b"]),
        ("a\t b", &["a", "\t", " b"]),
        ("a\t\t  b", &["a", "\t\t ", " b"]),
        ("中文\u{3000}文本", &["中文", "\u{3000}", "文本"]),
        ("中文 \u{3000}文本", &["中文", " \u{3000}", "文本"]),
        ("中文\u{3000} 文本", &["中文", "\u{3000}", " 文本"]),
        ("日本語のテキスト、中文文本。", &["日本語のテキスト、中文文本。"]),
        ("a\u{a0}b", &["a", "\u{a0}", "b"]),
        ("a\u{a0} b", &["a", "\u{a0}", " b"]),
        ("x\u{2028} y", &["x", "\u{2028}", " y"]),
        ("def f():\n    return 1", &["def", " f():", "\n   ", " return", " 1"]),
        ("a\r\n b", &["a", "\r\n", " b"]),
        // Not White_Space: zero-width space, BOM, Mongolian vowel separator stay inside words.
        ("a \u{200b}b", &["a", " \u{200b}b"]),
        ("a \u{feff}\u{180e}b", &["a", " \u{feff}\u{180e}b"]),
    ];
    for &(text, expected) in cases {
        assert_eq!(strs(SpacePrefix, text), expected, "space_prefix {text:?}");
    }
}

#[test]
fn whitespace_mode_is_unchanged_on_the_same_edge_cases() {
    assert_eq!(strs(Whitespace, "a  b"), ["a", "  ", "b"]);
    assert_eq!(strs(Whitespace, " lead"), [" ", "lead"]);
    assert_eq!(strs(Whitespace, "a \nb"), ["a", " \n", "b"]);
    assert_eq!(strs(Whitespace, "中文 \u{3000}文本"), ["中文", " \u{3000}", "文本"]);
    assert!(strs(Whitespace, "").is_empty());
    assert_eq!(Pretokenizer::default(), Whitespace);
}

#[test]
fn invalid_utf8_counts_as_non_whitespace_in_both_modes() {
    let raw = b"a\xff \xc3\xe3\x80\x80 \x80 b";
    assert_eq!(segs(Whitespace, raw), [&b"a\xff"[..], b" ", b"\xc3", b"\xe3\x80\x80 ", b"\x80", b" ", b"b"]);
    assert_eq!(segs(SpacePrefix, raw), [&b"a\xff"[..], b" \xc3", b"\xe3\x80\x80", b" \x80", b" b"]);
    assert_eq!(segs(SpacePrefix, b" \xff\xfe"), [b" \xff\xfe"]);
    assert_eq!(segs(SpacePrefix, b"\xff  "), [&b"\xff"[..], b"  "]);
    // A space never takes the continuation bytes of a whitespace char apart: U+3000 = E3 80 80.
    assert_eq!(segs(SpacePrefix, "\u{3000} x".as_bytes()), ["\u{3000}".as_bytes(), b" x"]);
}

/// Independent reference: char-level units, maximal runs, then (space_prefix) each non-whitespace
/// run moves the trailing U+0020 of the whitespace segment before it to its own front.
fn reference(p: Pretokenizer, bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut units: Vec<(Vec<u8>, bool)> = Vec::new();
    for chunk in bytes.utf8_chunks() {
        for c in chunk.valid().chars() {
            units.push((c.to_string().into_bytes(), c.is_whitespace()));
        }
        if !chunk.invalid().is_empty() {
            units.push((chunk.invalid().to_vec(), false));
        }
    }
    let mut runs: Vec<(Vec<u8>, bool)> = Vec::new();
    for (b, ws) in units {
        match runs.last_mut() {
            Some((run, rws)) if *rws == ws => run.extend(b),
            _ => runs.push((b, ws)),
        }
    }
    let mut out: Vec<(Vec<u8>, bool)> = Vec::new();
    for (run, ws) in runs {
        let mut word = run;
        if p == SpacePrefix && !ws {
            if let Some((prev, true)) = out.last_mut() {
                if prev.last() == Some(&b' ') {
                    prev.pop();
                    word.insert(0, b' ');
                    if prev.is_empty() {
                        out.pop();
                    }
                }
            }
        }
        out.push((word, ws));
    }
    out.into_iter().map(|(s, _)| s).collect()
}

/// Random byte strings over spaces, other whitespace, letters, CJK and invalid UTF-8.
fn random_bytes(rng: &mut fastrand::Rng) -> Vec<u8> {
    const PIECES: &[&[u8]] = &[
        b" ", b" ", b" ", b"  ", b"\t", b"\n", b"\r\n", "\u{3000}".as_bytes(), "\u{a0}".as_bytes(),
        "\u{2028}".as_bytes(), b"a", b"bc", b"x.", "中".as_bytes(), "é".as_bytes(), b"\xff", b"\xe3\x80",
        b"\x80", "\u{200b}".as_bytes(),
    ];
    (0..rng.usize(0..24)).flat_map(|_| PIECES[rng.usize(0..PIECES.len())].to_vec()).collect()
}

#[test]
fn segments_cover_the_input_and_match_the_reference() {
    let mut rng = fastrand::Rng::with_seed(11);
    for _ in 0..20_000 {
        let bytes = random_bytes(&mut rng);
        for p in [Whitespace, SpacePrefix] {
            let got = segs(p, &bytes);
            assert_eq!(got.concat(), bytes, "{p:?} must cover {bytes:?}");
            assert!(got.iter().all(|s| !s.is_empty()), "{p:?} empty segment for {bytes:?}");
            assert_eq!(got, reference(p, &bytes), "{p:?} for {bytes:?}");
            // Contiguous slices of the input, in order.
            let mut pos = 0;
            for s in p.segment_bytes(&bytes) {
                assert_eq!(s.as_ptr(), bytes[pos..].as_ptr());
                pos += s.len();
            }
        }
        assert_eq!(segs(Whitespace, &bytes), SuperBpeTrainer::segment_bytes(&bytes).iter().map(|s| s.to_vec()).collect::<Vec<_>>());
    }
}

#[test]
fn space_prefix_segments_are_a_space_plus_word_or_pure_whitespace() {
    let mut rng = fastrand::Rng::with_seed(12);
    for _ in 0..5_000 {
        let bytes = random_bytes(&mut rng);
        let ws_runs = segs(Whitespace, &bytes);
        let sp = segs(SpacePrefix, &bytes);
        // Same words (minus the moved space), at most as many segments.
        assert!(sp.len() <= ws_runs.len());
        for s in &sp {
            let body = s.strip_prefix(b" ").filter(|b| !b.is_empty()).unwrap_or(s);
            let parts = segs(Whitespace, body);
            assert_eq!(parts.len(), 1, "segment {s:?} of {bytes:?} mixes whitespace and words");
        }
    }
}

#[test]
fn space_prefix_round_trips_losslessly_through_a_trained_model() {
    let text = common::pseudo_text(3, 30_000);
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for seg in SpacePrefix.segment_text(&text) {
        *counts.entry(seg).or_insert(0) += 1;
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let model = BottomUpTrainer::new(&BpeScorer, 600).with_pretokenizer(SpacePrefix).train(&corpus).unwrap();
    assert!(model.pretokenize && model.pretokenizer == SpacePrefix);
    let spaced = (256..model.vocab.len() as u32)
        .filter(|&id| model.vocab.get_bytes(id).unwrap().starts_with(b" "))
        .count();
    assert!(spaced > 50, "space-prefixed words must be learned: {spaced}");
    let mut rng = fastrand::Rng::with_seed(5);
    for probe in [text.as_bytes().to_vec(), b"\xff  \xe3\x80 \x80\n x".to_vec()]
        .into_iter()
        .chain((0..500).map(|_| random_bytes(&mut rng)))
    {
        let ids = model.encode(&probe);
        assert_eq!(model.decode(&ids).unwrap(), probe);
        let dropped = model.encode_with_dropout(&probe, 0.3, Some(1));
        assert_eq!(model.decode(&dropped).unwrap(), probe);
    }
    assert!(model.encode(text.as_bytes()).len() * 2 < text.len(), "merges must apply");
}

#[test]
fn bpe_dropout_scope_is_the_space_prefix_segment() {
    // " ab" is one segment: dropout must re-queue skipped merges within it, so p -> 0 gives the
    // deterministic ids, and p = 1 gives raw bytes, exactly as for whitespace segments.
    let corpus = vec![(b" ab".to_vec(), 10u64), (b"ab".to_vec(), 3)];
    let model = BottomUpTrainer::new(&BpeScorer, 259).with_pretokenizer(SpacePrefix).train(&corpus).unwrap();
    let text = b"ab ab  ab";
    assert_eq!(model.encode_with_dropout(text, 1.0, Some(3)), text.iter().map(|&b| b as u32).collect::<Vec<_>>());
    assert_eq!(model.encode_with_dropout(text, 1e-9, Some(3)), model.encode(text));
    let whole = model.encode(b" ab");
    assert_eq!(whole.len(), 1, "\" ab\" must be one token");
    assert_eq!(model.encode(text), [model.encode(b"ab"), whole.clone(), vec![b' ' as u32], whole].concat());
}

#[test]
fn whitespace_model_still_splits_the_space_from_the_word() {
    // The same word counts in the default mode: a model whose words never hold " a" never merges
    // across, and encodes "x ab" as "x", " ", "ab" whatever the merges.
    let corpus = vec![(b"ab".to_vec(), 10u64), (b" ".to_vec(), 10)];
    let model = BottomUpTrainer::new(&BpeScorer, 258).train(&corpus).unwrap();
    assert!(model.pretokenize && model.pretokenizer == Whitespace);
    let ab = model.encode(b"ab");
    assert_eq!(model.encode(b" ab"), [vec![b' ' as u32], ab].concat());
}

#[test]
fn unigram_encodes_and_samples_per_space_prefix_segment() {
    // "a b" with a token " b": space_prefix segments are "a", " b", so " b" is usable, but the
    // token "a b" spanning a segment boundary never is, by Viterbi, FFBS or k-best.
    let vocab = vec![
        (b"a".to_vec(), 1, -5.0),
        (b" ".to_vec(), 2, -5.0),
        (b"b".to_vec(), 3, -5.0),
        (b" b".to_vec(), 4, -1.0),
        (b"a b".to_vec(), 5, -0.1),
    ];
    let trie = CompactTrie::build(&vocab);
    let sp = UnigramSampler { trie: &trie, pretokenizer: SpacePrefix };
    let ws = UnigramSampler::new(&trie);
    for seed in 0..50 {
        assert_eq!(sp.sample_ffbs(b"a b", 1e-3, Some(seed)), vec![1, 4]);
        assert_eq!(sp.sample_kbest(b"a b", 1, 1.0, Some(seed)), vec![1, 4]);
        for ids in [sp.sample_ffbs(b"a b", 1.0, Some(seed)), sp.sample_kbest(b"a b", 4, 1.0, Some(seed))] {
            assert!(ids == [1, 4] || ids == [1, 2, 3], "{ids:?}");
        }
        assert_eq!(ws.sample_ffbs(b"a b", 1.0, Some(seed)), vec![1, 2, 3]);
    }

    let text = common::pseudo_text(4, 20_000);
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for seg in SpacePrefix.segment_text(&text) {
        *counts.entry(seg).or_insert(0) += 1;
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let mut model = UnigramTrainer::new(400, 0.2).train(&corpus).unwrap();
    model.pretokenizer = SpacePrefix;
    let ids = model.encode(text.as_bytes());
    let per_segment: Vec<u32> = SpacePrefix.segment_bytes(text.as_bytes()).iter().flat_map(|s| model.encode(s)).collect();
    assert_eq!(ids, per_segment);
    assert_eq!(model.decode(&ids).unwrap(), text.as_bytes());
    assert!(ids.iter().any(|&id| model.vocab.get_bytes(id).unwrap().starts_with(b" ") && id >= 256));
}

#[test]
fn rebuilt_models_default_to_whitespace() {
    let model = BottomUpModel::new(tokenisers_core::vocab::Vocabulary::new_byte_level(), vec![]);
    assert!(!model.pretokenize && model.pretokenizer == Whitespace);
    assert_eq!(Pretokenizer::from_name("space_prefix").unwrap(), SpacePrefix);
    assert_eq!(Pretokenizer::from_name("whitespace").unwrap().name(), "whitespace");
    assert!(Pretokenizer::from_name("gpt2").is_err());
}
