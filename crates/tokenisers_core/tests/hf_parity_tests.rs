//! HF parity: an exported tokenizer.json loaded by the real Hugging Face `tokenizers` crate must
//! produce exactly the ids native encoding produces, for every exportable model family.

use std::collections::BTreeMap;

use tokenisers_core::curricula::super_bpe::{SuperBpeConfig, SuperBpeTrainer};
use tokenisers_core::engine::bottom_up::{BottomUpModel, BottomUpTrainer};
use tokenisers_core::engine::unigram::UnigramTrainer;
use tokenisers_core::export::huggingface::{HfTokenizerJson, WHITESPACE_RANGES};
use tokenisers_core::pretokenizer::Pretokenizer;
use tokenisers_core::scoring::*;
use tokenisers_core::vocab::TokenId;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

const DOCS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. Don't stop!",
    "hello, world! hello, world! hello,world -- (parens) [brackets] {braces}",
    "def foo(x):\n    return x + 1  # comment\n\n\tif x: pass",
    "price: $123.45, abc123 xyz789 1234567 3.14159e-10",
    "Héllo wörld, naïve café résumé façade",
    "日本語のテキスト、中文文本。한국어 텍스트",
    "中文　文本、中文。日本語　の　テキスト",
    "emoji 🚀🔥💯 party 🎉🎉 👩‍👩‍👧",
    "   leading spaces   and   runs\t\ttabs\n\nnewlines\r\n",
    "non\u{a0}breaking\u{a0}space and\u{2003}em\u{2009}thin\u{3000}ideographic",
    "it's they're we've I'm y'all",
];

const TESTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "hello, world!",
    "hello,world",
    "def foo(x):\n    return x + 1  # comment",
    "fn main() { println!(\"{}\", a[0] + b.c); }",
    "abc123 xyz789",
    "Héllo wörld, naïve café",
    "日本語のテキスト 中文文本",
    "中文　文本、中文。",
    "中　の、",
    "emoji 🚀🔥💯 party 🎉 👩‍👩‍👧",
    "a  b   c    d",
    "   leading",
    "trailing   ",
    "it's they're",
    "tabs\t\tand\n\nnewlines\r\n",
    "non\u{a0}breaking\u{3000}ideo\u{2028}line\u{85}nel\u{1680}ogham\u{205f}math",
    "zero\u{200b}width\u{feff}bom\u{180e}mvs\u{1c}fs",
    "",
    " ",
    "x",
];

fn corpus_docs() -> Vec<String> {
    DOCS.iter()
        .enumerate()
        .flat_map(|(i, d)| (0..(5 + i)).map(move |r| format!("{d} {r}")))
        .collect()
}

/// Word counts exactly as the Python binding builds them (segment_text segments).
fn word_counts(docs: &[String]) -> Vec<(Vec<u8>, u64)> {
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for d in docs {
        for seg in SuperBpeTrainer::segment_text(d) {
            *counts.entry(seg).or_insert(0) += 1;
        }
    }
    counts.into_iter().collect()
}

fn hf(json: &HfTokenizerJson) -> Tokenizer {
    json.to_json_string()
        .unwrap()
        .parse()
        .expect("HF tokenizers must load the exported tokenizer.json")
}

fn hf_ids(tok: &Tokenizer, s: &str) -> Vec<TokenId> {
    tok.encode(s, false).unwrap().get_ids().to_vec()
}

fn assert_parity(name: &str, tok: &Tokenizer, native: impl Fn(&str) -> Vec<TokenId>) {
    for s in TESTS {
        assert_eq!(native(s), hf_ids(tok, s), "[{name}] HF ids differ from native for {s:?}");
    }
}

fn bpe_models() -> Vec<(&'static str, BottomUpModel)> {
    let wc = word_counts(&corpus_docs());
    let train = |scorer: &dyn PairScorer| BottomUpTrainer::new(scorer, 420).train(&wc).unwrap();
    vec![
        ("bpe_frequency", train(&BpeScorer)),
        ("wordpiece", train(&WordPieceScorer::default())),
        ("wp_bpe_base", train(&WpBpeBaseScorer::default())),
        ("wp_bpe_ft", train(&WpBpeFtScorer::default())),
        ("wp_bpe_pig", train(&WpBpePigScorer::default())),
        ("wp_bpe_scp", train(&WpBpeScpScorer::default())),
        ("wp_bpe_npmi", train(&WpBpeNpmiScorer::default())),
        ("wp_bpe_ppig", train(&WpBpePpigScorer::default())),
        ("mdl", train(&MdlScorer::default())),
        ("length_penalized", train(&LengthPenalizedScorer::default())),
    ]
}

#[test]
fn bpe_family_export_matches_native_ids() {
    for (name, model) in bpe_models() {
        assert!(model.merges.len() > 20, "[{name}] trained too few merges to be a real test");
        let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
        assert_parity(name, &tok, |s| model.encode(s.as_bytes()));
    }
}

#[test]
fn super_bpe_export_matches_native_ids() {
    let docs = corpus_docs();
    let lines: Vec<&str> = docs.iter().map(String::as_str).collect();
    let config = SuperBpeConfig::new(330, 460).unwrap();
    let model = SuperBpeTrainer::new(config, &BpeScorer).train(&lines).unwrap();
    assert!(!model.pretokenize, "stage-2 SuperBPE model must encode whole text");
    let crossing = model.vocab.id_to_bytes_map().iter().any(|t| {
        let s = String::from_utf8_lossy(t);
        s.chars().any(char::is_whitespace) && s.chars().any(|c| !c.is_whitespace())
    });
    assert!(crossing, "test corpus must produce whitespace-crossing SuperBPE tokens");
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    assert_parity("super_bpe", &tok, |s| model.encode(s.as_bytes()));
}

#[test]
fn unigram_export_matches_native_ids() {
    let model = UnigramTrainer::new(500, 0.2).train(&word_counts(&corpus_docs())).unwrap();
    let tok = hf(&HfTokenizerJson::from_unigram_model(&model, None));
    assert_parity("unigram", &tok, |s| model.encode(s.as_bytes()));
}

/// Reference for "split on special tokens (leftmost-longest, like HF's added vocabulary), then
/// encode each remaining piece".
fn split_specials_then(model: &BottomUpModel, s: &str) -> Vec<TokenId> {
    let specials = model.vocab.special_tokens();
    let (mut out, mut piece_start, mut i) = (Vec::new(), 0, 0);
    while i < s.len() {
        let hit = specials
            .iter()
            .filter(|st| s[i..].starts_with(st.content.as_str()))
            .max_by_key(|st| st.content.len());
        match hit {
            Some(st) => {
                out.extend(model.encode(&s.as_bytes()[piece_start..i]));
                out.push(st.id);
                i += st.content.len();
                piece_start = i;
            }
            None => i += s[i..].chars().next().unwrap().len_utf8(),
        }
    }
    out.extend(model.encode(&s.as_bytes()[piece_start..]));
    out
}

#[test]
fn special_tokens_are_split_out_before_bpe_with_native_ids() {
    let wc = word_counts(&corpus_docs());
    let mut model = BottomUpTrainer::new(&BpeScorer, 400).train(&wc).unwrap();
    // Plain ASCII, one containing a space and one non-ASCII: the last two have a byte-level
    // printable form different from their raw content.
    for st in ["<|endoftext|>", "<sep> x", "▁mask▁"] {
        model.vocab.add_special_token(st).unwrap();
    }
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    for st in model.vocab.special_tokens() {
        assert_eq!(tok.token_to_id(&st.content), Some(st.id), "special {:?}", st.content);
    }
    for s in [
        "hello<|endoftext|>world",
        "<|endoftext|><|endoftext|>",
        "a <sep> x b<sep> xx",
        "日本語▁mask▁テキスト",
        "no specials here",
    ] {
        assert_eq!(split_specials_then(&model, s), hf_ids(&tok, s), "for {s:?}");
    }
}

#[test]
fn whitespace_ranges_are_exactly_char_is_whitespace() {
    for cp in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(cp) else { continue };
        let listed = WHITESPACE_RANGES.iter().any(|&(lo, hi)| lo <= c && c <= hi);
        assert_eq!(listed, c.is_whitespace(), "U+{cp:04X}");
    }
}

#[test]
fn hf_pre_tokenizer_splits_exactly_like_segment_text() {
    let model = &bpe_models()[0].1;
    let tok = hf(&HfTokenizerJson::from_bpe_model(model, None));
    let pre = tok.get_pre_tokenizer().expect("pre_tokenizer");
    let mut probe = String::from("a");
    for &(lo, hi) in &WHITESPACE_RANGES {
        for c in lo..=hi {
            probe.push(c);
            probe.push('b');
        }
    }
    probe.push_str("x\u{200b}\u{feff}\u{180e}\u{1c}\u{1f}\u{ad}y");
    for s in TESTS.iter().copied().chain([probe.as_str()]) {
        let mut pts = PreTokenizedString::from(s);
        pre.pre_tokenize(&mut pts).unwrap();
        let hf_offsets: Vec<(usize, usize)> = pts
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(_, offsets, _)| offsets)
            .collect();
        let mut native_offsets = Vec::new();
        let mut pos = 0;
        for seg in SuperBpeTrainer::segment_bytes(s.as_bytes()) {
            native_offsets.push((pos, pos + seg.len()));
            pos += seg.len();
        }
        assert_eq!(hf_offsets, native_offsets, "for {s:?}");
    }
}

/// With Unicode-whitespace segments a byte-level merge learned inside words can straddle a
/// multi-byte space: here (0xAD, 0xE3) from "中、"-like words would glue "中" to U+3000 (E3 80 80)
/// if text were encoded whole. Training never sees that, so HF must split segments first.
#[test]
fn export_keeps_merges_inside_unicode_whitespace_segments() {
    let words: Vec<String> = ['中', '乭', '亭', '侭', '傭']
        .iter()
        .flat_map(|c| ['、', '。', 'の', 'を', 'に', 'が'].map(move |h| format!("{c}{h}")))
        .collect();
    assert!(words.iter().all(|w| w.as_bytes().windows(2).any(|p| p == [0xAD, 0xE3])));
    let wc: Vec<(Vec<u8>, u64)> = words.iter().map(|w| (w.as_bytes().to_vec(), 10)).collect();
    let model = BottomUpTrainer::new(&BpeScorer, 257).train(&wc).unwrap();
    assert_eq!(model.vocab.get_bytes(256), Some(&[0xAD, 0xE3][..]), "top merge must be (AD, E3)");

    let s = "中\u{3000}中";
    let per_segment: Vec<TokenId> = SuperBpeTrainer::segment_bytes(s.as_bytes())
        .into_iter()
        .flat_map(|seg| model.encode(seg))
        .collect();
    // No merge applies inside either segment, so training-consistent ids are the raw bytes;
    // BPE over the unsplit text would emit token 256 = (AD, E3) straddling U+3000. Native
    // `BottomUpModel::encode` must pre-split the same way when `model.pretokenize` is set.
    assert_eq!(per_segment, s.bytes().map(TokenId::from).collect::<Vec<_>>());
    assert!(model.pretokenize);
    assert_eq!(model.encode(s.as_bytes()), per_segment);
    let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
    assert_eq!(hf_ids(&tok, s), per_segment);
}

// ---------------------------------------------------------------------------------------------
// Space-prefix mode: the exported Split regex must reproduce `Pretokenizer::SpacePrefix`.
// ---------------------------------------------------------------------------------------------

/// Word counts in `pretokenizer` segments, as the Python binding builds them.
fn word_counts_in(docs: &[String], pretokenizer: Pretokenizer) -> Vec<(Vec<u8>, u64)> {
    let mut counts: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    for d in docs {
        for seg in pretokenizer.segment_text(d) {
            *counts.entry(seg).or_insert(0) += 1;
        }
    }
    counts.into_iter().collect()
}

/// Extra probes for the space rules: a space before tabs, newlines and Unicode spaces, runs of
/// spaces before and after words, and spaces at both ends.
const SPACE_TESTS: &[&str] = &[
    "a \tb",
    "a\t b",
    "a \nb \n c",
    "x \u{3000}y\u{3000} z",
    "x\u{a0} y \u{a0}z",
    "  two leading",
    "two trailing  ",
    "   ",
    " \n ",
    "\n\n\tif x:\n    return  y",
    "a \u{200b}b \u{feff}c",
    " 中文 文本 ",
];

fn random_text(rng: &mut fastrand::Rng) -> String {
    let ws: Vec<char> = WHITESPACE_RANGES.iter().flat_map(|&(lo, hi)| lo..=hi).collect();
    let other = ['a', 'Z', '1', '.', '中', 'é', '🚀', '\u{200b}', '\u{feff}', '\u{180e}', '\u{1c}', '\u{ad}'];
    (0..rng.usize(0..30))
        .map(|_| match rng.u8(0..10) {
            0..=3 => ' ',
            4..=5 => ws[rng.usize(0..ws.len())],
            _ => other[rng.usize(0..other.len())],
        })
        .collect()
}

fn hf_splits(tok: &Tokenizer, s: &str) -> Vec<(usize, usize)> {
    let mut pts = PreTokenizedString::from(s);
    tok.get_pre_tokenizer().expect("pre_tokenizer").pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(_, offsets, _)| offsets)
        .collect()
}

fn native_splits(pretokenizer: Pretokenizer, s: &str) -> Vec<(usize, usize)> {
    let mut pos = 0;
    pretokenizer
        .segment_bytes(s.as_bytes())
        .into_iter()
        .map(|seg| {
            pos += seg.len();
            (pos - seg.len(), pos)
        })
        .collect()
}

#[test]
fn hf_split_regex_reproduces_both_pretokenizers_on_random_text() {
    let wc = word_counts(&corpus_docs());
    let mut rng = fastrand::Rng::with_seed(3);
    let random: Vec<String> = (0..20_000).map(|_| random_text(&mut rng)).collect();
    let mut probe = String::from("a");
    for &(lo, hi) in &WHITESPACE_RANGES {
        for c in lo..=hi {
            probe.extend([' ', c, 'b', c, ' ', 'b']);
        }
    }
    for p in [Pretokenizer::Whitespace, Pretokenizer::SpacePrefix] {
        let model = BottomUpTrainer::new(&BpeScorer, 300).with_pretokenizer(p).train(&wc).unwrap();
        let tok = hf(&HfTokenizerJson::from_bpe_model(&model, None));
        let texts = TESTS.iter().chain(SPACE_TESTS).map(|s| s.to_string()).chain([probe.clone()]).chain(random.clone());
        for s in texts {
            assert_eq!(hf_splits(&tok, &s), native_splits(p, &s), "{p:?} for {s:?}");
        }
    }
}

#[test]
fn space_prefix_bpe_family_export_matches_native_ids() {
    let wc = word_counts_in(&corpus_docs(), Pretokenizer::SpacePrefix);
    let train = |scorer: &dyn PairScorer| {
        BottomUpTrainer::new(scorer, 420).with_pretokenizer(Pretokenizer::SpacePrefix).train(&wc).unwrap()
    };
    let models = [
        ("bpe_frequency", train(&BpeScorer)),
        ("wordpiece", train(&WordPieceScorer::default())),
        ("wp_bpe_base", train(&WpBpeBaseScorer::default())),
        ("wp_bpe_ft", train(&WpBpeFtScorer::default())),
        ("wp_bpe_pig", train(&WpBpePigScorer::default())),
        ("wp_bpe_scp", train(&WpBpeScpScorer::default())),
        ("wp_bpe_npmi", train(&WpBpeNpmiScorer::default())),
        ("wp_bpe_ppig", train(&WpBpePpigScorer::default())),
        ("mdl", train(&MdlScorer::default())),
        ("length_penalized", train(&LengthPenalizedScorer::default())),
    ];
    let mut rng = fastrand::Rng::with_seed(4);
    let random: Vec<String> = (0..300).map(|_| random_text(&mut rng)).collect();
    let mut spaced = 0;
    for (name, model) in models {
        assert!(model.pretokenize && model.pretokenizer == Pretokenizer::SpacePrefix, "{name}");
        spaced += model.merges.iter().filter(|m| model.vocab.get_bytes(m.2).unwrap().starts_with(b" ")).count();
        let json = HfTokenizerJson::from_bpe_model(&model, None);
        assert_eq!(json.pretokenizer().unwrap(), Some(Pretokenizer::SpacePrefix));
        let reparsed: HfTokenizerJson = serde_json::from_str(&json.to_json_string().unwrap()).unwrap();
        assert_eq!(reparsed.pretokenizer().unwrap(), Some(Pretokenizer::SpacePrefix));
        let tok = hf(&json);
        assert_parity(name, &tok, |s| model.encode(s.as_bytes()));
        for s in SPACE_TESTS.iter().copied().chain(random.iter().map(String::as_str)) {
            assert_eq!(model.encode(s.as_bytes()), hf_ids(&tok, s), "[{name}] for {s:?}");
        }
    }
    assert!(spaced > 50, "the family must learn space-prefixed tokens: {spaced}");
}

#[test]
fn space_prefix_unigram_export_matches_native_ids() {
    let wc = word_counts_in(&corpus_docs(), Pretokenizer::SpacePrefix);
    let mut model = UnigramTrainer::new(500, 0.2).train(&wc).unwrap();
    model.pretokenizer = Pretokenizer::SpacePrefix;
    let json = HfTokenizerJson::from_unigram_model(&model, None);
    assert_eq!(json.pretokenizer().unwrap(), Some(Pretokenizer::SpacePrefix));
    let tok = hf(&json);
    assert_parity("unigram space_prefix", &tok, |s| model.encode(s.as_bytes()));
    let mut rng = fastrand::Rng::with_seed(5);
    for s in SPACE_TESTS.iter().map(|s| s.to_string()).chain((0..300).map(|_| random_text(&mut rng))) {
        assert_eq!(model.encode(s.as_bytes()), hf_ids(&tok, &s), "for {s:?}");
    }
    assert!(model.encode(b" the").len() < 4, "space-prefixed pieces must be learned");
}

#[test]
fn whitespace_export_is_unchanged_by_the_new_mode() {
    let model = &bpe_models()[0].1;
    assert_eq!(model.pretokenizer, Pretokenizer::Whitespace);
    let json = HfTokenizerJson::from_bpe_model(model, None);
    let v: serde_json::Value = serde_json::from_str(&json.to_json_string().unwrap()).unwrap();
    // The exact Split pattern every earlier export wrote.
    let expected = r"[\x{9}-\x{D}\x{20}\x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}-\x{2029}\x{202F}\x{205F}\x{3000}]+";
    assert_eq!(v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], expected);
    assert_eq!(json.pretokenizer().unwrap(), Some(Pretokenizer::Whitespace));
}
