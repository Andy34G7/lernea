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
/// non-whitespace in every mode; `Gpt2` reads each maximal invalid sequence as one U+FFFD.
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
    /// Every mode.
    pub const ALL: [Self; 3] = [Self::Whitespace, Self::SpacePrefix, Self::Gpt2];

    /// The name used in the Python API and in result metadata.
    pub fn name(self) -> &'static str {
        match self {
            Self::Whitespace => "whitespace",
            Self::SpacePrefix => "space_prefix",
            Self::Gpt2 => "gpt2",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name).ok_or_else(|| {
            TokeniserError::InvalidConfiguration(format!(
                "pretokenizer must be \"whitespace\", \"space_prefix\" or \"gpt2\", got {name:?}"
            ))
        })
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
        if self == Self::Gpt2 {
            return gpt2_segments(bytes);
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
