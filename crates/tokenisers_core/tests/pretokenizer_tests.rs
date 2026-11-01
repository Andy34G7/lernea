//! `Pretokenizer` segmentation: the default `Whitespace` mode is the old maximal-run split,
//! `SpacePrefix` (GPT-2-style) lets a word take the one U+0020 space before it, and `Gpt2` is
//! GPT-2's exact regex pre-split.
mod common;

use std::collections::BTreeMap;

use tokenisers_core::curricula::super_bpe::SuperBpeTrainer;
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::engine::unigram::{CompactTrie, UnigramTrainer};
use tokenisers_core::inference::unigram_sample::UnigramSampler;
use tokenisers_core::pretokenizer::Pretokenizer::{self, Gpt2, SpacePrefix, Whitespace};
use tokenisers_core::pretokenizer::GPT2_PATTERN;
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
    assert_eq!(Pretokenizer::from_name("gpt2").unwrap(), Gpt2);
    assert!(Pretokenizer::from_name("GPT2").is_err() && Pretokenizer::from_name("gpt-2").is_err());
    for p in Pretokenizer::ALL {
        assert_eq!(Pretokenizer::from_name(p.name()).unwrap(), p);
    }
}

// ---------------------------------------------------------------------------------------------
// Gpt2: GPT-2's regex pre-split
// ---------------------------------------------------------------------------------------------

#[test]
fn gpt2_pattern_is_tiktokens_gpt2_pat_str() {
    // tiktoken 0.14.0, tiktoken_ext/openai_public.py: r50k_pat_str, the "gpt2" encoding's pat_str.
    assert_eq!(
        GPT2_PATTERN,
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
    );
}

#[test]
fn gpt2_edge_cases() {
    let cases: &[(&str, &[&str])] = &[
        ("", &[]),
        ("hello", &["hello"]),
        ("the cat", &["the", " cat"]),
        (" lead", &[" lead"]),
        ("  lead", &[" ", " lead"]),
        ("trail ", &["trail", " "]),
        ("trail  ", &["trail", "  "]),
        ("   ", &["   "]),
        ("a  b", &["a", " ", " b"]),
        ("a   b", &["a", "  ", " b"]),
        // \s+(?!\S): a whitespace run before a word gives up only its last char, and only a
        // U+0020 joins the word; any other last char is a segment of its own.
        ("a \nb", &["a", " ", "\n", "b"]),
        ("a \n b", &["a", " \n", " b"]),
        ("a\n\nb", &["a", "\n", "\n", "b"]),
        ("a\n\n b", &["a", "\n\n", " b"]),
        ("a\tb", &["a", "\t", "b"]),
        ("a\t\t  b", &["a", "\t\t ", " b"]),
        ("a\r\n\r\nb", &["a", "\r\n\r", "\n", "b"]),
        ("x\n", &["x", "\n"]),
        ("x \n ", &["x", " \n "]),
        ("中文\u{3000}文本", &["中文", "\u{3000}", "文本"]),
        ("中文 \u{3000}文本", &["中文", " ", "\u{3000}", "文本"]),
        ("a\u{a0} b", &["a", "\u{a0}", " b"]),
        // Contractions: only these seven, lower case, and before anything; a space never joins one.
        ("it's", &["it", "'s"]),
        ("don't they're we've I'm you'll he'd", &["don", "'t", " they", "'re", " we", "'ve", " I", "'m", " you", "'ll", " he", "'d"]),
        ("IT'S", &["IT", "'", "S"]),
        ("'s", &["'s"]),
        (" 's", &[" '", "s"]),
        ("x 'll", &["x", " '", "ll"]),
        ("'sa", &["'s", "a"]),
        ("rock'n'roll", &["rock", "'", "n", "'", "roll"]),
        ("''s", &["''", "s"]),
        ("o'clock's", &["o", "'", "clock", "'s"]),
        ("it’s", &["it", "’", "s"]), // U+2019 is not an apostrophe to GPT-2
        // Digits: \p{N} runs, never joined with letters or punctuation.
        ("abc123def", &["abc", "123", "def"]),
        ("2024-11-01", &["2024", "-", "11", "-", "01"]),
        (" 3.14159", &[" 3", ".", "14159"]),
        ("$1,299.00", &["$", "1", ",", "299", ".", "00"]),
        ("x²+½ ⅷ ٣٤ १२", &["x", "²", "+", "½", " ⅷ", " ٣٤", " १२"]),
        // Punctuation and symbol runs, with the space before them.
        ("hello, world!!!", &["hello", ",", " world", "!!!"]),
        ("a -- b", &["a", " --", " b"]),
        ("f(x): pass", &["f", "(", "x", "):", " pass"]),
        ("emoji 🚀🔥 party", &["emoji", " 🚀🔥", " party"]),
        ("👩‍👩‍👧", &["👩‍👩‍👧"]),
        ("e\u{301}t\u{e9}", &["e", "\u{301}", "t\u{e9}"]), // a combining mark is \p{M}, not \p{L}
        // Unicode letters: every script's \p{L}, including CJK, kana and Hangul.
        ("Héllo wörld naïve", &["Héllo", " wörld", " naïve"]),
        ("日本語のテキスト、中文文本。", &["日本語のテキスト", "、", "中文文本", "。"]),
        ("한국어 텍스트", &["한국어", " 텍스트"]),
        ("Привет, мир", &["Привет", ",", " мир"]),
        // Not White_Space: zero-width space, BOM, Mongolian vowel separator are symbols (Cf).
        ("a \u{200b}b", &["a", " \u{200b}", "b"]),
    ];
    for &(text, expected) in cases {
        assert_eq!(strs(Gpt2, text), expected, "gpt2 {text:?}");
    }
}

/// `Gpt2` segments of the original GPT-2 release pattern, run by fancy-regex directly: an
/// independent check of tiktoken's "equivalent, but executes faster" rewrite and of the cutting.
fn original_gpt2(text: &str) -> Vec<String> {
    let re = fancy_regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
    re.find_iter(text).map(|m| m.unwrap().as_str().to_string()).collect()
}

fn random_text(rng: &mut fastrand::Rng) -> String {
    const PIECES: &[&str] = &[
        " ", " ", " ", "  ", "\t", "\n", "\r\n", "\u{3000}", "\u{a0}", "\u{2028}", "\u{85}", "a", "bc", "Z", "x.",
        "中", "é", "e\u{301}", "1", "42", "½", "²", "٣", "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "’",
        "!", "...", "-", "$", "🚀", "\u{200b}", "\u{feff}", "\u{fffd}", "_", "#",
    ];
    (0..rng.usize(0..30)).map(|_| PIECES[rng.usize(0..PIECES.len())]).collect()
}

#[test]
fn gpt2_segments_tile_the_text_and_match_the_original_pattern() {
    let mut rng = fastrand::Rng::with_seed(21);
    for _ in 0..30_000 {
        let text = random_text(&mut rng);
        let got = strs(Gpt2, &text);
        assert_eq!(got.concat(), text);
        assert!(got.iter().all(|s| !s.is_empty()), "empty segment for {text:?}");
        assert_eq!(got, original_gpt2(&text), "for {text:?}");
        // Every segment segments to itself, so trained words are single segments (`pretokenize`).
        for s in &got {
            assert_eq!(strs(Gpt2, s), [s.clone()], "segment {s:?} of {text:?} is not stable");
        }
    }
}

#[test]
fn gpt2_reads_invalid_utf8_as_one_replacement_char_per_invalid_sequence() {
    let cases: &[(&[u8], &[&[u8]])] = &[
        (b"\xff", &[b"\xff"]),
        (b"a\xffb", &[b"a", b"\xff", b"b"]),
        (b"a \xff\xfe!b", &[b"a", b" \xff\xfe!", b"b"]),
        (b"\xe3\x80 x", &[b"\xe3\x80", b" x"]),
        (b"it\xc3's", &[b"it", b"\xc3'", b"s"]), // a symbol before "'s" takes the apostrophe
        (b"1\x802", &[b"1", b"\x80", b"2"]),
        (b"\xff  \x80", &[b"\xff", b" ", b" \x80"]),
        (b"\xff\n", &[b"\xff", b"\n"]),
        ("\u{fffd}\u{ff}".as_bytes(), &["\u{fffd}".as_bytes(), "\u{ff}".as_bytes()]),
        (b"\xef\xbf\xbd\xff", &[b"\xef\xbf\xbd\xff"]),
    ];
    for &(bytes, expected) in cases {
        assert_eq!(segs(Gpt2, bytes), expected, "gpt2 {bytes:?}");
    }
    // In general: the segments of the text with every invalid sequence replaced by U+FFFD, mapped
    // back to the original bytes.
    let mut rng = fastrand::Rng::with_seed(22);
    for _ in 0..10_000 {
        let bytes = random_bytes(&mut rng);
        let got = segs(Gpt2, &bytes);
        assert_eq!(got.concat(), bytes);
        let replaced: Vec<String> = got.iter().map(|s| String::from_utf8_lossy(s).into_owned()).collect();
        // (Also: no invalid sequence is split, or its two parts would read as two U+FFFD.)
        assert_eq!(replaced, strs(Gpt2, &String::from_utf8_lossy(&bytes)), "for {bytes:?}");
    }
}

#[test]
fn gpt2_completes_a_whitespace_run_too_long_for_the_regex() {
    // fancy-regex fails with a stack overflow on ~1M whitespace chars before a word (tiktoken
    // panics); the split must go on exactly as `\s+(?!\S)` defines it.
    let n = 1_500_000;
    for (ws, word, expected_tail) in [(" ", "x", vec![" x".to_string()]), ("\n", "ab", vec!["\n".into(), "ab".into()])] {
        let text = format!("a{}{word}", ws.repeat(n));
        let got = strs(Gpt2, &text);
        let mut expected = vec!["a".to_string(), ws.repeat(n - 1)];
        expected.extend(expected_tail);
        assert_eq!(got, expected, "{ws:?} run");
    }
    let text = format!("x{}", " \t".repeat(n));
    assert_eq!(strs(Gpt2, &text), ["x".to_string(), " \t".repeat(n)]);
    let text = format!("{}\u{3000}y z", " ".repeat(n));
    assert_eq!(strs(Gpt2, &text), [" ".repeat(n), "\u{3000}".into(), "y".into(), " z".into()]);
}

#[test]
fn gpt2_round_trips_losslessly_through_a_trained_model() {
    let text = common::pseudo_text(3, 30_000) + " it's 1999, don't 3.14 — naïve!! \n\n  end";
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for seg in Gpt2.segment_text(&text) {
        *counts.entry(seg).or_insert(0) += 1;
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let model = BottomUpTrainer::new(&BpeScorer, 600).with_pretokenizer(Gpt2).train(&corpus).unwrap();
    assert!(model.pretokenize && model.pretokenizer == Gpt2);
    // No learned token mixes letters with digits or punctuation (GPT-2's sub-split).
    for id in 256..model.vocab.len() as u32 {
        let tok = String::from_utf8_lossy(model.vocab.get_bytes(id).unwrap()).into_owned();
        assert_eq!(strs(Gpt2, &tok).len(), 1, "token {tok:?} spans GPT-2 segments");
    }
    let mut rng = fastrand::Rng::with_seed(5);
    let mut probes = vec![text.as_bytes().to_vec(), b"\xff  \xe3\x80 \x80\n x it\xc3's".to_vec()];
    probes.extend((0..500).map(|_| random_bytes(&mut rng)));
    probes.extend((0..500).map(|_| random_text(&mut rng).into_bytes()));
    for probe in probes {
        let ids = model.encode(&probe);
        assert_eq!(model.decode(&ids).unwrap(), probe);
        let per_segment: Vec<u32> = Gpt2.segment_bytes(&probe).iter().flat_map(|s| model.encode(s)).collect();
        assert_eq!(ids, per_segment, "encode must pre-split like training for {probe:?}");
        let dropped = model.encode_with_dropout(&probe, 0.3, Some(1));
        assert_eq!(model.decode(&dropped).unwrap(), probe);
    }
    assert!(model.encode(text.as_bytes()).len() * 2 < text.len(), "merges must apply");
}

#[test]
fn bpe_dropout_scope_is_the_gpt2_segment() {
    let corpus = vec![(b" ab".to_vec(), 10u64), (b"ab".to_vec(), 3), (b"12".to_vec(), 5)];
    let model = BottomUpTrainer::new(&BpeScorer, 260).with_pretokenizer(Gpt2).train(&corpus).unwrap();
    let text = b"ab ab12  ab";
    assert_eq!(model.encode_with_dropout(text, 1.0, Some(3)), text.iter().map(|&b| b as u32).collect::<Vec<_>>());
    assert_eq!(model.encode_with_dropout(text, 1e-9, Some(3)), model.encode(text));
    let whole = model.encode(b" ab");
    assert_eq!(whole.len(), 1, "\" ab\" must be one token");
    let twelve = model.encode(b"12");
    assert_eq!(twelve.len(), 1);
    assert_eq!(model.encode(text), [model.encode(b"ab"), whole.clone(), twelve, vec![b' ' as u32], whole].concat());
}

#[test]
fn unigram_encodes_and_samples_per_gpt2_segment() {
    // "it's": segments "it", "'s"; the token "t'" spanning them is never usable.
    let vocab = vec![
        (b"i".to_vec(), 1, -0.5),
        (b"t".to_vec(), 2, -5.0),
        (b"'".to_vec(), 3, -5.0),
        (b"s".to_vec(), 4, -0.5),
        (b"it".to_vec(), 5, -1.0),
        (b"'s".to_vec(), 6, -1.0),
        (b"t'".to_vec(), 7, -0.01),
    ];
    let trie = CompactTrie::build(&vocab);
    let g2 = UnigramSampler { trie: &trie, pretokenizer: Gpt2 };
    for seed in 0..50 {
        for ids in [g2.sample_ffbs(b"it's", 1.0, Some(seed)), g2.sample_kbest(b"it's", 4, 1.0, Some(seed))] {
            assert!(!ids.contains(&7), "{ids:?}");
        }
        assert_eq!(g2.sample_ffbs(b"it's", 1e-3, Some(seed)), vec![5, 6]);
    }
    // Without the split, "i" + "t'" + "s" is the best path.
    assert_eq!(UnigramSampler::new(&trie).sample_ffbs(b"it's", 1e-3, Some(0)), vec![1, 7, 4]);

    let text = common::pseudo_text(4, 20_000) + " it's 2024, isn't it?";
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for seg in Gpt2.segment_text(&text) {
        *counts.entry(seg).or_insert(0) += 1;
    }
    let corpus: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    let mut model = UnigramTrainer::new(400, 0.2).train(&corpus).unwrap();
    model.pretokenizer = Gpt2;
    let ids = model.encode(text.as_bytes());
    let per_segment: Vec<u32> = Gpt2.segment_bytes(text.as_bytes()).iter().flat_map(|s| model.encode(s)).collect();
    assert_eq!(ids, per_segment);
    assert_eq!(model.decode(&ids).unwrap(), text.as_bytes());
    let raw = b"\xff it\xc3's \x80\x80 x";
    assert_eq!(model.decode(&model.encode(raw)).unwrap(), raw);
}

// ---------------------------------------------------------------------------------------------
// Hf: lossless Hugging Face pre-tokenizers, run by `tokenizers`
// ---------------------------------------------------------------------------------------------

use serde_json::{json, Value};
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, PreTokenizerWrapper};

fn byte_level(use_regex: bool) -> Value {
    json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": use_regex})
}

fn split(regex: &str, behavior: &str, invert: bool) -> Value {
    json!({"type": "Split", "pattern": {"Regex": regex}, "behavior": behavior, "invert": invert})
}

fn seq(parts: &[Value]) -> Value {
    json!({"type": "Sequence", "pretokenizers": parts})
}

const PUNCT: fn() -> Value = || json!({"type": "Punctuation", "behavior": "Isolated"});
const DIGITS: fn() -> Value = || json!({"type": "Digits", "individual_digits": true});

/// Every allowed component, alone and in Sequences. Names ending in "inside chars" have a splitter
/// after ByteLevel that cuts inside multi-byte chars.
fn hf_configs() -> Vec<(&'static str, Value)> {
    vec![
        ("split isolated", split(r"\p{N}+", "Isolated", false)),
        ("split merged with previous", split(" ", "MergedWithPrevious", false)),
        ("split merged with next", split(r"\s+", "MergedWithNext", false)),
        ("split contiguous", split("[,.!?]", "Contiguous", false)),
        ("split inverted", split(r"\p{L}+", "Isolated", true)),
        ("split string", json!({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false})),
        ("digits", json!({"type": "Digits", "individual_digits": false})),
        ("digits individual", DIGITS()),
        ("punctuation", PUNCT()),
        ("punctuation contiguous", json!({"type": "Punctuation", "behavior": "Contiguous"})),
        ("punctuation default", json!({"type": "Punctuation"})),
        ("fixed length", json!({"type": "FixedLength", "length": 3})),
        ("byte level regex", byte_level(true)),
        ("byte level", byte_level(false)),
        ("llama digits", seq(&[DIGITS(), byte_level(true)])),
        ("punctuation isolated", seq(&[split(r"\s+", "MergedWithNext", false), PUNCT(), byte_level(false)])),
        ("nested", seq(&[seq(&[DIGITS(), PUNCT()]), split(r"\p{Han}+", "Isolated", false)])),
        ("punctuation inside chars", seq(&[byte_level(true), PUNCT()])),
        // Falcon's: `Digits` after ByteLevel isolates "¹" (byte B9) inside "ス" (E3 82 B9).
        ("falcon inside chars", seq(&[
            json!({"type": "Punctuation", "behavior": "Contiguous"}),
            json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
            json!({"type": "Digits", "individual_digits": false}),
            split("[0-9][0-9][0-9]", "Isolated", false),
        ])),
    ]
}

const HF_EDGE: &[&str] = &[
    "",
    " ",
    "   ",
    "hello",
    " lead",
    "trail  ",
    "abc123def 2024-11-01 3.14159 $1,299.00 a1b2",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "Привет, мир! Héllo wörld e\u{301}t\u{e9} naïve x²+½ ٣٤ १२",
    "emoji 🚀🔥 party 👩\u{200d}👩\u{200d}👧!",
    "a  b\t\t c\n\n\r\n  \u{3000}x\u{a0} y \u{2028}z   ",
    "it's don't 'll mixed中文text123ДА",
    "\u{fffd}\u{200b}\u{feff}\0\u{1c}",
];

/// HF's own boundaries (`get_splits` offsets in the original text) of `config` on `text`.
fn hf_own(config: &Value, text: &str) -> Vec<(usize, usize)> {
    let pre: PreTokenizerWrapper = serde_json::from_value(config.clone()).unwrap();
    let mut pts = PreTokenizedString::from(text);
    pre.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetReferential::Original, OffsetType::Byte).into_iter().map(|(_, o, _)| o).collect()
}

fn spans(segments: &[Vec<u8>]) -> Vec<(usize, usize)> {
    let mut pos = 0;
    segments.iter().map(|s| { pos += s.len(); (pos - s.len(), pos) }).collect()
}

#[test]
fn hf_segments_tile_the_text_and_equal_hfs_own_boundaries() {
    let mut rng = fastrand::Rng::with_seed(31);
    let texts: Vec<String> = HF_EDGE.iter().map(|s| s.to_string()).chain((0..3000).map(|_| random_text(&mut rng))).collect();
    for (name, config) in hf_configs() {
        let p = Pretokenizer::hf(config.clone()).unwrap();
        let (mut compared, mut cut_chars) = (0, 0);
        for text in &texts {
            let got = segs(p, text.as_bytes());
            assert_eq!(got.concat(), text.as_bytes(), "{name}: {text:?}");
            assert!(got.iter().all(|s| !s.is_empty()), "{name}: empty segment for {text:?}");
            // A segment on its own splits into itself, unless it is part of a char.
            for s in got.iter().filter(|s| std::str::from_utf8(s).is_ok()) {
                assert_eq!(segs(p, s), [s.clone()], "{name}: segment {s:?} of {text:?} splits again");
            }
            // HF's own offsets in the original text are per char: they tile exactly unless a piece
            // starts or ends inside a char (tests/e2e checks those pieces against Python's HF).
            let own = hf_own(&config, text);
            if own == spans(&got) {
                compared += 1;
            } else {
                assert!(name.ends_with("inside chars"), "{name}: {text:?}: native {:?} != HF {own:?}", spans(&got));
                assert!(got.iter().any(|s| std::str::from_utf8(s).is_err()), "{name}: {text:?}");
                cut_chars += 1;
            }
        }
        assert!(compared > texts.len() / 3, "{name}: {compared}");
        assert_eq!(cut_chars > 0, name.ends_with("inside chars"), "{name}: {cut_chars}");
    }
}

#[test]
fn hf_edge_cases() {
    let llama = Pretokenizer::hf(seq(&[DIGITS(), byte_level(true)])).unwrap();
    assert_eq!(strs(llama, "it's 2024 abc123def!"), ["it", "'s", " ", "2", "0", "2", "4", " abc", "1", "2", "3", "def", "!"]);
    assert_eq!(strs(llama, "  x \u{3000}中文"), [" ", " x", " ", "\u{3000}", "中文"]);
    let punct = Pretokenizer::hf(PUNCT()).unwrap();
    assert_eq!(strs(punct, "a,b!! c... 中文。"), ["a", ",", "b", "!", "!", " c", ".", ".", ".", " 中文", "。"]);
    let fixed = Pretokenizer::hf(json!({"type": "FixedLength", "length": 2})).unwrap();
    assert_eq!(strs(fixed, "abcde中文🚀"), ["ab", "cd", "e中", "文🚀"]);
    let triples = Pretokenizer::hf(seq(&[byte_level(true), split("[0-9][0-9][0-9]", "Isolated", false)])).unwrap();
    assert_eq!(strs(triples, "x 1234567 é"), ["x", " ", "123", "456", "7", " é"]);
    // After ByteLevel, "á" (C3 A1) is "Ã¡" and "¡" is punctuation: the cut is inside the char.
    let inside = Pretokenizer::hf(seq(&[byte_level(true), PUNCT()])).unwrap();
    assert_eq!(segs(inside, "á, x".as_bytes()), [&b"\xc3"[..], b"\xa1", b",", b" x"]);
    let digits = Pretokenizer::hf(seq(&[byte_level(true), json!({"type": "Digits", "individual_digits": false})])).unwrap();
    assert_eq!(segs(digits, "ス".as_bytes()), [&b"\xe3\x82"[..], b"\xb9"]);
}

#[test]
fn hf_reads_each_invalid_utf8_byte_as_one_nul() {
    let mut rng = fastrand::Rng::with_seed(32);
    let probes: Vec<Vec<u8>> = (0..2000).map(|_| random_bytes(&mut rng)).chain([b"it\xc3's \xff\xfe!! 12\x803".to_vec()]).collect();
    for (name, config) in hf_configs() {
        let p = Pretokenizer::hf(config).unwrap();
        for bytes in &probes {
            let got = segs(p, bytes);
            assert_eq!(got.concat(), *bytes, "{name}");
            let nul: Vec<u8> = bytes.utf8_chunks().flat_map(|c| c.valid().bytes().chain(c.invalid().iter().map(|_| 0))).collect();
            let lens = |s: &[Vec<u8>]| s.iter().map(Vec::len).collect::<Vec<_>>();
            assert_eq!(lens(&got), lens(&segs(p, &nul)), "{name}: {bytes:?}");
        }
    }
    let llama = Pretokenizer::hf(seq(&[DIGITS(), byte_level(true)])).unwrap();
    assert_eq!(segs(llama, b"a\xff,b \xe3\x80 12"), [&b"a"[..], b"\xff,", b"b", b" \xe3\x80", b" ", b"1", b"2"]);
}

#[test]
fn hf_rejects_every_component_that_is_not_a_lossless_splitter() {
    let bl = byte_level(true);
    let cases: Vec<(Value, &str)> = vec![
        (json!({"type": "Whitespace"}), "Whitespace drops whitespace"),
        (json!({"type": "WhitespaceSplit"}), "WhitespaceSplit drops whitespace"),
        (json!({"type": "BertPreTokenizer"}), "BertPreTokenizer drops whitespace"),
        (json!({"type": "CharDelimiterSplit", "delimiter": "x"}), "CharDelimiterSplit drops the delimiter"),
        (json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}), "Metaspace replaces spaces"),
        (json!({"type": "UnicodeScripts"}), "UnicodeScripts drops the U+0020 spaces"),
        (split("x", "Removed", false), "Split with behavior Removed"),
        (json!({"type": "Punctuation", "behavior": "Removed"}), "Punctuation with behavior Removed"),
        (json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}), "ByteLevel with add_prefix_space"),
        (seq(&[DIGITS(), seq(&[PUNCT(), json!({"type": "Whitespace"})])]), "Whitespace drops whitespace"),
        (seq(&[DIGITS(), json!({"type": "Metaspace"})]), "Metaspace replaces spaces"),
        (seq(&[bl.clone(), PUNCT(), bl]), "2 ByteLevels"),
        // tokenizers panics on it ("chunk size must be non-zero"), so it must not reach the probe.
        (json!({"type": "FixedLength", "length": 0}), "FixedLength with length 0"),
        (seq(&[byte_level(false), json!({"type": "FixedLength", "length": 0})]), "FixedLength with length 0"),
        (json!({"type": "Bogus"}), "Bogus is not a known lossless splitter"),
        (json!({"type": "Digits"}), "missing field `individual_digits`"),
        (json!("Digits"), "HF pre-tokenizer"),
    ];
    for (config, expected) in cases {
        let err = Pretokenizer::hf(config.clone()).unwrap_err().to_string();
        assert!(err.contains(expected) && err.contains("HF pre-tokenizer"), "{config}: {err}");
    }
}

#[test]
fn hf_configs_are_interned_and_named_hf() {
    let a = Pretokenizer::hf(seq(&[DIGITS(), byte_level(true)])).unwrap();
    let b = Pretokenizer::hf(seq(&[DIGITS(), byte_level(true)])).unwrap();
    let (Pretokenizer::Hf(x), Pretokenizer::Hf(y)) = (a, b) else { panic!("not Hf") };
    assert!(a == b && std::ptr::eq(x, y));
    assert_ne!(a, Pretokenizer::hf(seq(&[DIGITS(), byte_level(false)])).unwrap());
    assert_ne!(Pretokenizer::hf(byte_level(true)).unwrap(), Gpt2);
    assert_eq!(a.name(), "hf");
    assert!(x.byte_level() && !matches!(Pretokenizer::hf(DIGITS()).unwrap(), Pretokenizer::Hf(h) if h.byte_level()));
    assert!(Pretokenizer::from_name("hf").is_err());
}

#[test]
fn hf_byte_level_regex_is_the_gpt2_mode() {
    // HF's ByteLevel(use_regex=true) through `Hf` (char -> byte mapping, NUL for invalid bytes)
    // against the native fancy-regex `Gpt2`: the same segments, training and ids.
    let hf = Pretokenizer::hf(byte_level(true)).unwrap();
    let mut rng = fastrand::Rng::with_seed(33);
    for _ in 0..20_000 {
        let text = random_text(&mut rng);
        assert_eq!(strs(hf, &text), strs(Gpt2, &text), "for {text:?}");
        let bytes = random_bytes(&mut rng);
        assert_eq!(segs(hf, &bytes), segs(Gpt2, &bytes), "for {bytes:?}");
    }
    for text in HF_EDGE {
        assert_eq!(strs(hf, text), strs(Gpt2, text), "for {text:?}");
    }
    let text = common::pseudo_text(6, 40_000) + " it's 1999, don't 3.14 — naïve!! \n\n  end 中文 🚀";
    let counts = |p: Pretokenizer| {
        let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for seg in p.segment_text(&text) {
            *counts.entry(seg).or_insert(0) += 1;
        }
        counts.into_iter().collect::<Vec<_>>()
    };
    let corpus = counts(Gpt2);
    assert_eq!(counts(hf), corpus);
    let native = BottomUpTrainer::new(&BpeScorer, 700).with_pretokenizer(Gpt2).train(&corpus).unwrap();
    let via_hf = BottomUpTrainer::new(&BpeScorer, 700).with_pretokenizer(hf).train(&corpus).unwrap();
    assert!(via_hf.pretokenize && via_hf.merges == native.merges);
    assert_eq!(via_hf.vocab.id_to_bytes_map(), native.vocab.id_to_bytes_map());
    let mut unigram_native = UnigramTrainer::new(500, 0.2).train(&corpus).unwrap();
    unigram_native.pretokenizer = Gpt2;
    let mut unigram_hf = unigram_native.clone();
    unigram_hf.pretokenizer = hf;
    let mut probes = vec![text.into_bytes(), b"\xff  \xe3\x80 \x80\n x it\xc3's 12".to_vec()];
    probes.extend((0..300).map(|_| random_bytes(&mut rng)));
    probes.extend((0..300).map(|_| random_text(&mut rng).into_bytes()));
    for probe in &probes {
        assert_eq!(via_hf.encode(probe), native.encode(probe));
        assert_eq!(via_hf.encode_with_dropout(probe, 0.2, Some(4)), native.encode_with_dropout(probe, 0.2, Some(4)));
        assert_eq!(unigram_hf.encode(probe), unigram_native.encode(probe));
    }
}
