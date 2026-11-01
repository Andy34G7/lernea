use crate::error::{Result, TokeniserError};

/// How text is split into segments (pre-tokens) before training and encoding. No learned token
/// spans a segment boundary, so the same segmentation must be used for training word counts, the
/// native encoder, Unigram's lattice and the Hugging Face export (`export::huggingface`).
///
/// Whitespace is Unicode `White_Space` (`char::is_whitespace`), so U+3000, U+00A0, U+2028 etc.
/// separate words just like ASCII space. Bytes that are not valid UTF-8 count as non-whitespace.
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
}

impl Pretokenizer {
    /// The name used in the Python API and in result metadata.
    pub fn name(self) -> &'static str {
        match self {
            Self::Whitespace => "whitespace",
            Self::SpacePrefix => "space_prefix",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "whitespace" => Ok(Self::Whitespace),
            "space_prefix" => Ok(Self::SpacePrefix),
            _ => Err(TokeniserError::InvalidConfiguration(format!(
                "pretokenizer must be \"whitespace\" or \"space_prefix\", got {name:?}"
            ))),
        }
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
