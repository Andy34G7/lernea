use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::engine::bottom_up::BottomUpModel;
use crate::engine::unigram::UnigramModel;
use crate::error::{Result, TokeniserError};
use crate::pretokenizer::Pretokenizer;
use crate::vocab::{Vocabulary, BASE_VOCAB_SIZE};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
    Split {
        pattern: SplitPattern,
        behavior: String,
        invert: bool,
    },
    Sequence {
        pretokenizers: Vec<PreTokenizerConfig>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SplitPattern {
    String(String),
    Regex(String),
}

/// Code-point ranges of Unicode `White_Space`, i.e. exactly the chars where `char::is_whitespace`
/// is true. `Pretokenizer` segmentation splits on these.
pub const WHITESPACE_RANGES: [(char, char); 10] = [
    ('\u{9}', '\u{D}'),
    ('\u{20}', '\u{20}'),
    ('\u{85}', '\u{85}'),
    ('\u{A0}', '\u{A0}'),
    ('\u{1680}', '\u{1680}'),
    ('\u{2000}', '\u{200A}'),
    ('\u{2028}', '\u{2029}'),
    ('\u{202F}', '\u{202F}'),
    ('\u{205F}', '\u{205F}'),
    ('\u{3000}', '\u{3000}'),
];

/// The `WHITESPACE_RANGES` inside an Oniguruma character class.
fn whitespace_class() -> String {
    WHITESPACE_RANGES
        .iter()
        .map(|&(lo, hi)| {
            if lo == hi {
                format!("\\x{{{:X}}}", lo as u32)
            } else {
                format!("\\x{{{:X}}}-\\x{{{:X}}}", lo as u32, hi as u32)
            }
        })
        .collect()
}

/// Oniguruma regex matching maximal whitespace runs, as `Pretokenizer::Whitespace` defines them.
pub fn whitespace_run_regex() -> String {
    format!("[{}]+", whitespace_class())
}

/// Oniguruma regex whose matches are exactly the `Pretokenizer::SpacePrefix` segments (they tile
/// the text, so `Isolated` leaves no gaps). With W = whitespace and N = non-whitespace, at the
/// start of a segment it matches, in order:
/// - ` ?N+`: a word, with the space right before it if there is one;
/// - `W+?(?= N)`: the whitespace run before a word up to the space the word takes (lazy: the
///   shortest W+ followed by " N" ends right before the run's last char, a space);
/// - `W+`: any other whitespace run, whole (at the end of the text, or ending in a whitespace
///   char other than U+0020 before a word).
pub fn space_prefix_regex() -> String {
    let ws = whitespace_class();
    format!("\\x{{20}}?[^{ws}]+|[{ws}]+?(?=\\x{{20}}[^{ws}])|[{ws}]+")
}

/// The `Split` regex reproducing `pretokenizer`'s segmentation.
pub fn split_regex(pretokenizer: Pretokenizer) -> String {
    match pretokenizer {
        Pretokenizer::Whitespace => whitespace_run_regex(),
        Pretokenizer::SpacePrefix => space_prefix_regex(),
    }
}

/// Raw bytes -> GPT-2 byte-level chars on the whole input (no GPT-2 regex: native encoding has
/// no regex pre-tokenization).
fn byte_level() -> PreTokenizerConfig {
    PreTokenizerConfig::ByteLevel {
        add_prefix_space: false,
        trim_offsets: false,
        use_regex: false,
    }
}

/// HF equivalent of native pre-tokenization: split into `pretokenizer` segments, then byte-level
/// map. Without a pretokenizer (SuperBPE) the whole text is one word.
fn pre_tokenizer(pretokenizer: Option<Pretokenizer>) -> PreTokenizerConfig {
    let Some(pretokenizer) = pretokenizer else {
        return byte_level();
    };
    PreTokenizerConfig::Sequence {
        pretokenizers: vec![
            PreTokenizerConfig::Split {
                pattern: SplitPattern::Regex(split_regex(pretokenizer)),
                behavior: "Isolated".into(),
                invert: false,
            },
            byte_level(),
        ],
    }
}

fn added_tokens(vocab: &Vocabulary) -> Vec<AddedToken> {
    vocab
        .special_tokens()
        .iter()
        .map(|st| AddedToken {
            id: st.id,
            content: st.content.clone(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        })
        .collect()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum DecoderConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
}

/// Merges as `"u v"` strings, also accepting the `["u", "v"]` pairs `tokenizers` >= 0.20 saves
/// (byte-level token strings never contain a space).
fn merges_from_strings_or_pairs<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Merge {
        Joined(String),
        Pair(String, String),
    }
    Ok(Vec::<Merge>::deserialize(deserializer)?
        .into_iter()
        .map(|m| match m {
            Merge::Joined(s) => s,
            Merge::Pair(u, v) => format!("{u} {v}"),
        })
        .collect())
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum HfModelConfig {
    BPE {
        dropout: Option<f32>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
        vocab: BTreeMap<String, u32>,
        #[serde(deserialize_with = "merges_from_strings_or_pairs")]
        merges: Vec<String>,
    },
    Unigram {
        unk_id: Option<u32>,
        vocab: Vec<(String, f64)>,
        byte_fallback: bool,
    },
}

/// Canonical Hugging Face tokenizer.json representation with byte losslessness.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HfTokenizerJson {
    pub version: String,
    pub truncation: Option<serde_json::Value>,
    pub padding: Option<serde_json::Value>,
    pub added_tokens: Vec<AddedToken>,
    pub normalizer: Option<serde_json::Value>,
    pub pre_tokenizer: PreTokenizerConfig,
    pub post_processor: Option<serde_json::Value>,
    pub decoder: DecoderConfig,
    pub model: HfModelConfig,
}

impl HfTokenizerJson {
    /// Creates a BPE tokenizer.json whose `Tokenizer::encode(text, false)` ids equal native
    /// encoding: HF splits added (special) tokens out first, then pre-tokenizes exactly like
    /// `BottomUpModel::pretokenize` says, then applies the merges in rank order.
    pub fn from_bpe_model(model: &BottomUpModel, dropout: Option<f32>) -> Self {
        let vocab = &model.vocab;
        let printable = |id| vocab.bytes_to_printable(vocab.get_bytes(id).unwrap_or(&[]));
        let merges = model
            .merges
            .iter()
            .map(|&(u, v, _)| format!("{} {}", printable(u), printable(v)))
            .collect();

        let mut string_vocab: BTreeMap<String, u32> = vocab.to_string_map().into_iter().collect();
        // HF resolves an added token's id by looking its raw content up in the model vocab. Key
        // special tokens by raw content; keep the byte-level key too if merges can produce it.
        let merge_outputs: HashSet<u32> = model.merges.iter().map(|m| m.2).collect();
        for st in vocab.special_tokens() {
            let key = vocab.bytes_to_printable(&st.bytes);
            if key != st.content {
                if st.id as usize >= BASE_VOCAB_SIZE && !merge_outputs.contains(&st.id) {
                    string_vocab.remove(&key);
                }
                string_vocab.insert(st.content.clone(), st.id);
            }
        }

        Self {
            version: "1.0".into(),
            truncation: None,
            padding: None,
            added_tokens: added_tokens(vocab),
            normalizer: None,
            pre_tokenizer: pre_tokenizer(model.pretokenize.then_some(model.pretokenizer)),
            post_processor: None,
            decoder: DecoderConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: false,
            },
            model: HfModelConfig::BPE {
                dropout,
                unk_token: None,
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                ignore_merges: false,
                vocab: string_vocab,
                merges,
            },
        }
    }

    /// Creates a Unigram tokenizer.json whose ids equal `UnigramModel::encode` (segment-wise
    /// Viterbi; HF breaks score ties the same way: first-found, shortest start position).
    pub fn from_unigram_model(model: &UnigramModel, unk_id: Option<u32>) -> Self {
        let unigram_vocab = model
            .vocab
            .id_to_bytes_map()
            .iter()
            .enumerate()
            .map(|(id, bytes)| {
                let score = model.log_probs.get(id).copied().unwrap_or(-15.0);
                (model.vocab.bytes_to_printable(bytes), score)
            })
            .collect();

        Self {
            version: "1.0".into(),
            truncation: None,
            padding: None,
            added_tokens: added_tokens(&model.vocab),
            normalizer: None,
            pre_tokenizer: pre_tokenizer(Some(model.pretokenizer)),
            post_processor: None,
            decoder: DecoderConfig::ByteLevel {
                add_prefix_space: false,
                trim_offsets: false,
                use_regex: false,
            },
            model: HfModelConfig::Unigram {
                unk_id,
                vocab: unigram_vocab,
                byte_fallback: false,
            },
        }
    }

    /// `BottomUpModel::pretokenize` for a model loaded from this file (see `pretokenizer`).
    pub fn pretokenize(&self) -> Result<bool> {
        self.pretokenizer().map(|p| p.is_some())
    }

    /// The segmentation of a model loaded from this file: None for the bare whole-text byte-level
    /// pre-tokenizer (what SuperBPE exports), `Some(p)` for the `split_regex(p)` split + byte level
    /// that the other exports write. Native encoding implements nothing else, so
    /// any other pre-tokenizer (GPT-2 regex, `add_prefix_space`, other split patterns) or any
    /// normalizer is an error rather than a silently different encoding.
    pub fn pretokenizer(&self) -> Result<Option<Pretokenizer>> {
        let bare = |p: &PreTokenizerConfig| {
            matches!(
                p,
                PreTokenizerConfig::ByteLevel { add_prefix_space: false, use_regex: false, .. }
            )
        };
        let split = |re: &str| {
            [Pretokenizer::Whitespace, Pretokenizer::SpacePrefix]
                .into_iter()
                .find(|&p| split_regex(p) == re)
        };
        let flag = match &self.pre_tokenizer {
            p if bare(p) => Some(None),
            PreTokenizerConfig::Sequence { pretokenizers } => match pretokenizers.as_slice() {
                [PreTokenizerConfig::Split {
                    pattern: SplitPattern::Regex(re),
                    behavior,
                    invert: false,
                }, byte]
                    if split(re).is_some() && behavior == "Isolated" && bare(byte) =>
                {
                    Some(split(re))
                }
                _ => None,
            },
            _ => None,
        };
        match flag {
            Some(flag) if self.normalizer.is_none() => Ok(flag),
            _ => Err(TokeniserError::InvalidConfiguration(format!(
                "unsupported normalizer {:?} / pre_tokenizer {:?}: only the byte-level \
                 pre-tokenizers this crate exports encode like native",
                self.normalizer, self.pre_tokenizer
            ))),
        }
    }

    /// Exports the tokenizer configuration to a file or directory.
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let target_file = if path.is_dir() || path.extension().is_none() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };

        if let Some(parent) = target_file.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(target_file)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Serializes tokenizer config to formatted JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(TokeniserError::Json)
    }
}
