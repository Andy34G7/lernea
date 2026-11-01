use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, PreTokenizerWrapper};

use crate::error::{Result, TokeniserError};

/// GPT-2's pre-tokenization regex, verbatim from tiktoken 0.14.0 (`tiktoken_ext/openai_public.py`,
/// `r50k_pat_str`, the `pat_str` of its "gpt2" encoding). tiktoken documents it as equivalent to,
/// but faster than, the original GPT-2 release's
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, which is the
/// pattern Hugging Face's `ByteLevel(use_regex=true)` pre-tokenizer applies.
pub const GPT2_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

/// How text is split into segments (pre-tokens) before training and encoding. No learned token
/// spans a segment boundary, so the same segmentation must be used for training word counts, the
/// native encoder, Unigram's lattice and the Hugging Face export (`export::huggingface`).
///
/// Whitespace is Unicode `White_Space` (`char::is_whitespace`, the regex `\s`), so U+3000, U+00A0,
/// U+2028 etc. separate words just like ASCII space. Bytes that are not valid UTF-8 count as
/// non-whitespace in every mode; `Gpt2` reads each maximal invalid sequence as one U+FFFD, `Hf` each
/// invalid byte as one U+0000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Pretokenizer {
    /// Alternating maximal runs of whitespace and non-whitespace: every whitespace run is its own
    /// segment, so a space never joins a word.
    #[default]
    Whitespace,
    /// GPT-2-style: like `Whitespace`, except that a non-whitespace run takes the single U+0020
    /// SPACE immediately before it as its first byte ("a  b" -> "a", " ", " b"). Every other
    /// whitespace character (newlines, tabs, other Unicode spaces, all but the last space of a
    /// longer run) stays in its whitespace run. GPT-2's `\s+(?!\S)| ?\S+` idea without its
    /// letter / number / punctuation sub-splitting.
    SpacePrefix,
    /// GPT-2's exact pre-split: the matches of [`GPT2_PATTERN`] (fancy-regex, as tiktoken's Rust
    /// core runs it), which tile any text: English contractions, runs of letters, of numbers and
    /// of other non-whitespace, each with the one U+0020 before it if there is one, and whitespace
    /// runs ("it's  2024!\n" -> "it", "'s", " ", " 2024", "!", "\n").
    ///
    /// Invalid UTF-8 (only byte input such as `encode_bytes` has any; training text and the
    /// Hugging Face export are valid UTF-8 by type): each maximal invalid sequence
    /// (`Utf8Chunk::invalid`) is segmented exactly as one U+FFFD REPLACEMENT CHARACTER would be,
    /// i.e. as a symbol (non-whitespace, not a letter or number), and its segment holds the
    /// original bytes, so the segments still concatenate to the input.
    ///
    /// fancy-regex gives up (stack overflow) on a whitespace run of about a million chars followed
    /// by non-whitespace, where tiktoken panics. The split then goes on with the regex's own rule
    /// for that run (`\s+(?!\S)`: all of it but its last char), so the result is unchanged.
    Gpt2,
    /// A lossless Hugging Face `tokenizers` pre-tokenizer given as its `pre_tokenizer` JSON config
    /// and run by `tokenizers` itself (see [`HfPreTokenizer`]; made by [`Pretokenizer::hf`]).
    Hf(&'static HfPreTokenizer),
}

/// A Hugging Face pre-tokenizer that only splits: `Split` and `Punctuation` with any behavior but
/// `Removed`, `Digits`, `FixedLength` with length > 0 (0 panics in `tokenizers`), at most one
/// `ByteLevel` without `add_prefix_space`, and `Sequence`s of these. Every other component drops,
/// adds or rewrites text (`Whitespace`, `WhitespaceSplit`, `BertPreTokenizer` and
/// `CharDelimiterSplit` drop what they split on, `Metaspace` turns spaces into U+2581,
/// `UnicodeScripts` drops the U+0020 spaces a piece starts with) and is rejected. On top of that,
/// every new config must cut a probe text into segments that concatenate to it.
///
/// HF works on chars: its pieces are mapped back to bytes of the text, one byte per char when the
/// config has a `ByteLevel` (which turns every byte into one GPT-2 byte-level char), else as the
/// text itself. Pieces that do not tile the text are an error. A splitter after `ByteLevel` can
/// cut inside a char, as HF does (Falcon's `Digits`: "ス" is E3 82 B9, mapped to "ãĤ¹", and "¹" is
/// numeric; `Punctuation`: "ö" is "Ã¶"): such a segment is not valid UTF-8, and on its own need not
/// split as it did in context, so models trust their training words to be segments.
pub struct HfPreTokenizer {
    /// The `pre_tokenizer` config as given; the Hugging Face export writes it back unchanged.
    pub config: serde_json::Value,
    pre_tokenizer: PreTokenizerWrapper,
    /// GPT-2 byte-level char -> byte (`vocab::bytes_char`) when the config has a `ByteLevel`.
    byte_level: Option<[Option<u8>; 512]>,
}

impl PartialEq for HfPreTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
    }
}

impl Eq for HfPreTokenizer {}

impl std::hash::Hash for HfPreTokenizer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.config.to_string().hash(state); // objects serialise with sorted keys
    }
}

impl std::fmt::Debug for HfPreTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.config)
    }
}

/// Text of every kind a new `Hf` config must cut losslessly: scripts, digits in words, punctuation,
/// emoji, combining marks, whitespace runs, leading and trailing spaces, and chars with a byte that
/// GPT-2's byte-level map turns into punctuation (A1 ¡, A7 §, AB «, B6 ¶, B7 ·, BB », BF ¿).
const PROBE: &str = " Hello, wörld!! It's 2024-11-01: naïve café e\u{301}t\u{e9}, x²+½ ٣٤ १२ $1,299.00\n\n\t \
    \u{3000}中文文本。日本語のテキスト 한국어 Привет🚀🔥 👩\u{200d}👩\u{200d}👧 rock'n'roll   ab12cd \u{a0}\u{fffd}\u{200b}x \
    ¡olá! «ç» ¿ÿ? ·ŧ ";

/// Why the HF config component `config` (HF's own serialisation, so `type` is always set) is not
/// a lossless splitter, if it is not; counts its `ByteLevel`s.
fn check_splits_only(config: &serde_json::Value, byte_levels: &mut usize) -> std::result::Result<(), String> {
    let kind = config["type"].as_str().unwrap_or_default();
    let why = match kind {
        "Sequence" => {
            let parts = config["pretokenizers"].as_array().into_iter().flatten();
            return parts.into_iter().try_for_each(|p| check_splits_only(p, byte_levels));
        }
        "Split" | "Punctuation" if config["behavior"] == "Removed" => "with behavior Removed drops what it splits on",
        "FixedLength" if config["length"] == 0 => "with length 0 panics in tokenizers (chunk size 0)",
        "Split" | "Punctuation" | "Digits" | "FixedLength" => return Ok(()),
        "ByteLevel" if config["add_prefix_space"] != false => "with add_prefix_space adds a space",
        "ByteLevel" => {
            *byte_levels += 1;
            return Ok(());
        }
        "Whitespace" | "WhitespaceSplit" | "BertPreTokenizer" => "drops whitespace",
        "CharDelimiterSplit" => "drops the delimiter",
        "Metaspace" => "replaces spaces with U+2581 (and can add one)",
        "UnicodeScripts" => "drops the U+0020 spaces a piece starts with (\" ab\" -> \"ab\")",
        _ => "is not a known lossless splitter",
    };
    Err(format!(
        "{kind} {why}; only lossless splitters are allowed: Split and Punctuation (behavior not \
         Removed), Digits, FixedLength (length > 0), ByteLevel (add_prefix_space=false) and \
         Sequences of these"
    ))
}

impl HfPreTokenizer {
    fn new(config: serde_json::Value) -> std::result::Result<Self, String> {
        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_value(config.clone())
            .map_err(|e| check_splits_only(&config, &mut 0).err().unwrap_or_else(|| e.to_string()))?;
        let mut byte_levels = 0;
        check_splits_only(&serde_json::to_value(&pre_tokenizer).map_err(|e| e.to_string())?, &mut byte_levels)?;
        if byte_levels > 1 {
            return Err(format!("{byte_levels} ByteLevels would map the bytes {byte_levels} times; use one"));
        }
        let byte_level = (byte_levels == 1).then(|| crate::vocab::bytes_char().1);
        let hf = Self { config, pre_tokenizer, byte_level };
        hf.segments(PROBE.as_bytes())?;
        Ok(hf)
    }

    /// The config has a `ByteLevel`, so HF's pieces are byte-level chars.
    pub fn byte_level(&self) -> bool {
        self.byte_level.is_some()
    }

    /// Segments of `bytes`. HF takes text, so each byte of invalid UTF-8 reads as U+0000, a symbol
    /// (not whitespace, a letter, a digit or punctuation), one char per byte: HF's cuts are then
    /// offsets in `bytes`, also where a splitter after `ByteLevel` cuts inside a char.
    fn segments<'a>(&self, bytes: &'a [u8]) -> std::result::Result<Vec<&'a [u8]>, String> {
        let text = match std::str::from_utf8(bytes) {
            Ok(text) => std::borrow::Cow::Borrowed(text),
            Err(_) => bytes
                .utf8_chunks()
                .map(|chunk| chunk.valid().to_owned() + &"\0".repeat(chunk.invalid().len()))
                .collect(),
        };
        Ok(cut(bytes, &self.cuts(&text)?))
    }

    /// End offsets of the segments of `text` (ascending, the last one `text.len()`): HF's pieces
    /// mapped back to `text`. An error unless they concatenate to `text` exactly.
    // ponytail: HF's PreTokenizedString (strings, alignments, one NormalizedString per piece) peaks
    // at ~50 bytes of RAM per byte of text, ~90 with a ByteLevel (measured: 22 MB of mixed text,
    // 4.9M pieces, ~2 GB; `gpt2` mode needs ~1/10 of that): a 100 MB document needs ~9 GB. Chunk
    // huge inputs if that matters (no cut point is safe for every config: Split regexes span any).
    fn cuts(&self, text: &str) -> std::result::Result<Vec<usize>, String> {
        let mut pre = PreTokenizedString::from(text);
        self.pre_tokenizer.pre_tokenize(&mut pre).map_err(|e| e.to_string())?;
        let bytes = text.as_bytes();
        let (mut cuts, mut pos) = (Vec::new(), 0);
        for (piece, _, _) in pre.get_splits(OffsetReferential::Normalized, OffsetType::Byte) {
            let tiles = match &self.byte_level {
                Some(to_byte) => piece.chars().all(|c| {
                    let byte = to_byte.get(c as usize).copied().flatten();
                    let hit = byte.is_some() && bytes.get(pos).copied() == byte;
                    pos += hit as usize;
                    hit
                }),
                None => {
                    let hit = bytes[pos..].starts_with(piece.as_bytes());
                    pos += if hit { piece.len() } else { 0 };
                    hit
                }
            };
            if !tiles {
                return Err(format!("not lossless: piece {piece:?} is not the text at byte {pos}"));
            }
            cuts.push(pos);
        }
        if pos != bytes.len() {
            return Err(format!("not lossless: the pieces end at byte {pos} of {}", bytes.len()));
        }
        Ok(cuts)
    }
}

thread_local! {
    // One compiled copy per thread, as in tiktoken: threads searching with one shared fancy-regex
    // `Regex` contend on its scratch space.
    static GPT2_REGEX: fancy_regex::Regex =
        fancy_regex::Regex::new(GPT2_PATTERN).expect("GPT2_PATTERN compiles");
}

/// End offsets of the `GPT2_PATTERN` segments of `text`, ascending, the last one `text.len()`.
fn gpt2_cuts(text: &str) -> Vec<usize> {
    let mut cuts = Vec::new();
    // Each search starts at a match boundary. The pattern has no lookbehind and `$` is the end of
    // `text` either way, so searching `text[start..]` finds the same matches as searching `text`.
    let mut start = 0;
    GPT2_REGEX.with(|re| loop {
        if re.find_iter(&text[start..]).all(|m| m.map(|m| cuts.push(start + m.end())).is_ok()) {
            break;
        }
        // The search failed at `end`: a huge whitespace run before a non-whitespace char.
        let end = cuts.last().copied().unwrap_or(0).max(start);
        let run = text[end..].find(|c: char| !c.is_whitespace()).map_or(text.len(), |n| end + n);
        let last = text[end..run].char_indices().last().map_or(end, |(i, _)| end + i);
        start = if run > end && run == text.len() {
            run // `\s++$`: the whole run
        } else if last > end {
            last // `\s+(?!\S)`: the run up to its last char
        } else {
            // Never seen: go on one char at a time (the split stays lossless).
            end + text[end..].chars().next().map_or(text.len() - end, char::len_utf8)
        };
        cuts.push(start);
    });
    if cuts.last().is_some_and(|&c| c < text.len()) {
        cuts.push(text.len()); // the matches tile the text, so never needed
    }
    cuts
}

/// `bytes` cut at `cuts` (ascending end offsets, the last one `bytes.len()`).
fn cut<'a>(bytes: &'a [u8], cuts: &[usize]) -> Vec<&'a [u8]> {
    let mut start = 0;
    cuts.iter()
        .map(|&end| {
            let seg = &bytes[start..end];
            start = end;
            seg
        })
        .collect()
}

/// [`Pretokenizer::Gpt2`] segments of `bytes`.
fn gpt2_segments(bytes: &[u8]) -> Vec<&[u8]> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return cut(bytes, &gpt2_cuts(text));
    }
    // Each invalid sequence becomes one U+FFFD; `orig[i]` is the input offset of byte `i` of the
    // substituted text, exact at every char start (where matches end) and at its end.
    let mut text = String::with_capacity(bytes.len() + 16);
    let mut orig = Vec::with_capacity(bytes.len() + 16);
    let mut pos = 0;
    for chunk in bytes.utf8_chunks() {
        text.push_str(chunk.valid());
        orig.extend(pos..pos + chunk.valid().len());
        pos += chunk.valid().len();
        if !chunk.invalid().is_empty() {
            text.push(char::REPLACEMENT_CHARACTER);
            orig.extend([pos; 3]);
            pos += chunk.invalid().len();
        }
    }
    orig.push(pos);
    let cuts: Vec<usize> = gpt2_cuts(&text).into_iter().map(|c| orig[c]).collect();
    cut(bytes, &cuts)
}

impl Pretokenizer {
    /// Every named mode (`Hf` takes a config instead).
    pub const ALL: [Self; 3] = [Self::Whitespace, Self::SpacePrefix, Self::Gpt2];

    /// The name used in the Python API and in result metadata ("hf" for every `Hf` config).
    pub fn name(self) -> &'static str {
        match self {
            Self::Whitespace => "whitespace",
            Self::SpacePrefix => "space_prefix",
            Self::Gpt2 => "gpt2",
            Self::Hf(_) => "hf",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name).ok_or_else(|| {
            TokeniserError::InvalidConfiguration(format!(
                "pretokenizer must be \"whitespace\", \"space_prefix\", \"gpt2\" or {{\"hf\": <pre_tokenizer \
                 config>}}, got {name:?}"
            ))
        })
    }

    /// `Hf` mode for a Hugging Face `pre_tokenizer` JSON config, e.g. Llama-style single digits
    /// `{"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": true},
    /// {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}]}`.
    /// An error names the component that is not a lossless splitter (see [`HfPreTokenizer`]).
    pub fn hf(config: serde_json::Value) -> Result<Self> {
        // ponytail: each distinct config is built and leaked once, for the process lifetime, so
        // `Pretokenizer` stays `Copy`; an Arc and a non-Copy enum if configs ever get per-request.
        static INTERNED: parking_lot::Mutex<Vec<&'static HfPreTokenizer>> = parking_lot::Mutex::new(Vec::new());
        let mut interned = INTERNED.lock();
        if let Some(&hf) = interned.iter().find(|hf| hf.config == config) {
            return Ok(Self::Hf(hf));
        }
        let hf = HfPreTokenizer::new(config.clone()).map_err(|why| {
            TokeniserError::InvalidConfiguration(format!("HF pre-tokenizer {config}: {why}"))
        })?;
        let hf: &'static HfPreTokenizer = Box::leak(Box::new(hf));
        interned.push(hf);
        Ok(Self::Hf(hf))
    }

    /// Segments of `text` (see [`Self::segment_bytes`]).
    pub fn segment_text(self, text: &str) -> Vec<Vec<u8>> {
        self.segment_bytes(text.as_bytes())
            .into_iter()
            .map(<[u8]>::to_vec)
            .collect()
    }

    /// Non-empty segments of `bytes`; concatenated they give back `bytes`.
    pub fn segment_bytes(self, bytes: &[u8]) -> Vec<&[u8]> {
        match self {
            Self::Gpt2 => return gpt2_segments(bytes),
            // `HfPreTokenizer::new` only accepts splitters, so this never fails.
            Self::Hf(hf) => return hf.segments(bytes).unwrap_or_else(|e| panic!("HF pre-tokenizer {hf:?}: {e}")),
            _ => {}
        }
        // (is_whitespace, byte length) per char; each invalid UTF-8 run is one non-ws unit.
        let units = bytes.utf8_chunks().flat_map(|chunk| {
            let invalid = chunk.invalid().len();
            chunk
                .valid()
                .chars()
                .map(|c| (c.is_whitespace(), c.len_utf8()))
                .chain((invalid > 0).then_some((false, invalid)))
        });
        // Maximal runs as (end offset, is_whitespace).
        let mut runs: Vec<(usize, bool)> = Vec::new();
        let mut pos = 0;
        for (ws, len) in units {
            pos += len;
            match runs.last_mut() {
                Some(last) if last.1 == ws => last.0 = pos,
                _ => runs.push((pos, ws)),
            }
        }
        let mut segments = Vec::with_capacity(runs.len());
        let mut start = 0;
        for (i, &(end, ws)) in runs.iter().enumerate() {
            // A whitespace run followed by a word gives that word its last byte if it is a space
            // (the last byte of a whitespace char is 0x20 only for U+0020).
            let takes_space =
                self == Self::SpacePrefix && ws && i + 1 < runs.len() && bytes[end - 1] == b' ';
            let cut = if takes_space { end - 1 } else { end };
            if cut > start {
                segments.push(&bytes[start..cut]);
                start = cut;
            }
        }
        segments
    }
}
