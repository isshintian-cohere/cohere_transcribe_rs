use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Special token IDs (static constants derived from the model's tokenizer config).
pub struct SpecialTokens {
    pub unk: i64,
    pub nospeech: i64,
    pub pad: i64,
    pub eos: i64,
    pub startoftranscript: i64,
    pub pnc: i64,
    pub nopnc: i64,
    pub startofcontext: i64,
    pub noitn: i64,
    pub notimestamp: i64,
    pub nodiarize: i64,
    pub emo_undefined: i64,
    pub lang_ids: HashMap<String, i64>,
    pub special_ids: HashSet<i64>,
}

impl SpecialTokens {
    pub fn from_tokenizer_config(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("tokenizer_config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read tokenizer_config.json at {:?}", path))?;

        #[derive(Deserialize)]
        struct TokenEntry {
            content: String,
            special: Option<bool>,
        }
        #[derive(Deserialize)]
        struct TokenizerConfig {
            added_tokens_decoder: HashMap<String, TokenEntry>,
        }

        let cfg: TokenizerConfig =
            serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;

        let mut token_to_id: HashMap<String, i64> = HashMap::new();
        let mut special_ids = HashSet::new();

        for (id_str, entry) in &cfg.added_tokens_decoder {
            let id: i64 = id_str.parse().context("Invalid token ID")?;
            token_to_id.insert(entry.content.clone(), id);
            if entry.special.unwrap_or(false) {
                special_ids.insert(id);
            }
        }

        let get = |name: &str| -> Result<i64> {
            token_to_id
                .get(name)
                .copied()
                .with_context(|| format!("Special token '{}' not found", name))
        };

        let lang_codes = [
            "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
        ];
        let mut lang_ids = HashMap::new();
        for code in &lang_codes {
            let token = format!("<|{}|>", code);
            if let Ok(id) = get(&token) {
                lang_ids.insert(code.to_string(), id);
                special_ids.insert(id);
            }
        }

        Ok(Self {
            unk: get("<unk>").unwrap_or(0),
            nospeech: get("<|nospeech|>").unwrap_or(1),
            pad: token_to_id.get("<pad>").copied().unwrap_or(2),
            eos: get("<|endoftext|>").unwrap_or(3),
            startoftranscript: get("<|startoftranscript|>").unwrap_or(4),
            pnc: get("<|pnc|>").unwrap_or(5),
            nopnc: get("<|nopnc|>").unwrap_or(6),
            startofcontext: get("<|startofcontext|>").unwrap_or(7),
            noitn: get("<|noitn|>").unwrap_or(9),
            notimestamp: get("<|notimestamp|>").unwrap_or(11),
            nodiarize: get("<|nodiarize|>").unwrap_or(13),
            emo_undefined: get("<|emo:undefined|>").unwrap_or(16),
            lang_ids,
            special_ids,
        })
    }

    /// Build the decoder prompt token IDs for the given language and punctuation setting.
    ///
    /// Prompt format:
    ///   <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
    ///   <|{lang}|> <|{lang}|> <|pnc|or|nopnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
    pub fn build_prompt(&self, language: &str, punctuation: bool) -> Result<Vec<i64>> {
        let lang_id = self
            .lang_ids
            .get(language)
            .copied()
            .with_context(|| format!("Unsupported language: '{}'", language))?;
        let pnc_id = if punctuation { self.pnc } else { self.nopnc };
        Ok(vec![
            self.startofcontext,
            self.startoftranscript,
            self.emo_undefined,
            lang_id,
            lang_id,
            pnc_id,
            self.noitn,
            self.notimestamp,
            self.nodiarize,
        ])
    }
}

/// Vocabulary loaded from `vocab.json` produced by `tools/extract_vocab.py`.
/// Maps token ID → piece string.
pub struct Vocab {
    pub id_to_piece: HashMap<i64, String>,
}

impl Vocab {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("vocab.json");
        let content = std::fs::read_to_string(&path).with_context(|| {
            format!(
                "Cannot read vocab.json at {:?}. \
                 Please run `python tools/extract_vocab.py --model_dir <path>` first.",
                path
            )
        })?;
        let raw: HashMap<String, String> =
            serde_json::from_str(&content).context("Failed to parse vocab.json")?;
        let id_to_piece = raw
            .into_iter()
            .map(|(k, v)| -> Result<(i64, String)> { Ok((k.parse()?, v)) })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { id_to_piece })
    }
}

pub struct Tokenizer {
    vocab: Vocab,
    pub special: SpecialTokens,
}

impl Tokenizer {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let vocab = Vocab::load(model_dir.as_ref())?;
        let special = SpecialTokens::from_tokenizer_config(model_dir.as_ref())?;
        Ok(Self { vocab, special })
    }

    /// Decode a list of token IDs to a string, skipping all special tokens.
    /// Handles SentencePiece ▁ (U+2581) word-boundary markers and <0xXX> byte tokens.
    pub fn decode(&self, ids: &[i64]) -> String {
        let skip = &self.special.special_ids;

        let mut result = String::new();
        for &id in ids {
            if skip.contains(&id) {
                continue;
            }
            let piece = match self.vocab.id_to_piece.get(&id) {
                Some(p) => p.as_str(),
                None => continue,
            };

            if piece.starts_with('\u{2581}') {
                // ▁ marks a word boundary — replace with space
                if !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(&piece['\u{2581}'.len_utf8()..]);
            } else if piece.starts_with("<0x") && piece.ends_with('>') {
                // Raw byte token (e.g. <0xE3>)
                if let Ok(byte) = u8::from_str_radix(&piece[3..piece.len() - 1], 16) {
                    result.push(byte as char);
                }
            } else if piece == "<unk>" || piece == "<s>" || piece == "</s>" {
                // Skip sentence-piece meta tokens
            } else {
                result.push_str(piece);
            }
        }
        result
    }
}
