use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub transf_decoder: TransfDecoderConfig,
    pub head: HeadConfig,
    pub preprocessor: PreprocessorConfig,
    pub vocab_size: usize,
    pub max_audio_clip_s: f64,
    pub overlap_chunk_second: f64,
    pub sample_rate: usize,
    pub supported_languages: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ff_expansion_factor: usize,
    pub conv_kernel_size: usize,
    pub dropout: f64,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize,
    pub feat_in: usize,
    pub pos_emb_max_len: usize,
}

#[derive(Debug, Deserialize)]
pub struct TransfDecoderConfig {
    pub config_dict: DecoderConfigDict,
}

#[derive(Debug, Deserialize)]
pub struct DecoderConfigDict {
    pub hidden_size: usize,
    pub inner_size: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub max_sequence_length: usize,
    pub hidden_act: String,
}

#[derive(Debug, Deserialize)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub num_classes: usize,
}

#[derive(Debug, Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub features: usize,
    pub n_fft: usize,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub normalize: String,
    pub dither: f64,
    pub log: bool,
    pub frame_splicing: usize,
}

impl ModelConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read config.json at {:?}", path))?;
        serde_json::from_str(&content).context("Failed to parse config.json")
    }
}
