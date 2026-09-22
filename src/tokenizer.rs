//! Wrapper over the Rust Hugging Face tokenizer shipped with a Laya checkpoint.

use std::path::Path;

use tokenizers::Tokenizer as RustTokenizer;

pub struct Tokenizer {
    inner: RustTokenizer,
    pub mask_token: String,
    pub mask_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub pad_token_id: i32,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let inner = RustTokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| error.to_string())?;
        let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .map_err(|error| error.to_string())?;
        let config: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| error.to_string())?;
        let special = |key: &str| -> Result<String, String> {
            config
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("tokenizer_config.json has no {key}"))
        };
        let mask_token = special("mask_token")?;
        let cls_token = special("cls_token")?;
        let sep_token = special("sep_token")?;
        let pad_token = special("pad_token")?;
        let id = |token: &str| -> Result<u32, String> {
            inner
                .token_to_id(token)
                .ok_or_else(|| format!("tokenizer has no id for {token}"))
        };
        Ok(Self {
            mask_token_id: id(&mask_token)?,
            cls_token_id: id(&cls_token)?,
            sep_token_id: id(&sep_token)?,
            pad_token_id: id(&pad_token)? as i32,
            mask_token,
            inner,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }
}
