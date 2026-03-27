use cohere_transcribe_rs::{audio, config, tokenizer};

#[cfg(feature = "tch-backend")]
use cohere_transcribe_rs::{decoder, encoder, inference, weights};

#[cfg(feature = "mlx")]
use cohere_transcribe_rs::mlx;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

#[derive(Parser, Debug)]
#[command(
    name = "transcribe",
    about = "Cohere Transcribe ASR — Rust implementation",
    long_about = "Transcribe audio files using the Cohere Transcribe model.\n\n\
                  Model weights must be downloaded from HuggingFace:\n\
                  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026\n\n\
                  Run `python tools/extract_vocab.py --model_dir <model_dir>` once \
                  to generate vocab.json before using this tool."
)]
struct Args {
    /// Path to the model directory (must contain config.json, model.safetensors, vocab.json)
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Audio file(s) to transcribe
    #[arg(required = true)]
    audio_files: Vec<PathBuf>,

    /// Language code (en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko)
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Disable punctuation in the output
    #[arg(long)]
    no_punctuation: bool,

    /// Maximum number of tokens to generate per audio segment
    #[arg(long, default_value = "448")]
    max_tokens: usize,

    /// Log verbosity (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Logging
    let log_level = match args.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    // Validate model directory
    let model_dir = &args.model_dir;
    anyhow::ensure!(
        model_dir.exists(),
        "Model directory does not exist: {:?}",
        model_dir
    );
    for required in &["config.json", "model.safetensors", "vocab.json"] {
        anyhow::ensure!(
            model_dir.join(required).exists(),
            "Missing required file '{}' in {:?}. \
             Run `python tools/extract_vocab.py --model_dir {:?}` to generate vocab.json.",
            required,
            model_dir,
            model_dir,
        );
    }

    // Load config
    tracing::info!("Loading model config...");
    let cfg = config::ModelConfig::load(model_dir)?;

    // Load tokenizer
    tracing::info!("Loading tokenizer...");
    let tokenizer = tokenizer::Tokenizer::load(model_dir)?;

    // Validate language
    anyhow::ensure!(
        cfg.supported_languages.contains(&args.language),
        "Language '{}' is not supported. Supported: {:?}",
        args.language,
        cfg.supported_languages
    );

    // Audio preprocessing config
    let mel_cfg = audio::MelConfig::from_model_config(&cfg);

    run_backend(&args, model_dir, &cfg, &tokenizer, &mel_cfg)
}

#[cfg(feature = "tch-backend")]
fn run_backend(
    args: &Args,
    model_dir: &std::path::Path,
    cfg: &config::ModelConfig,
    tokenizer: &tokenizer::Tokenizer,
    mel_cfg: &audio::MelConfig,
) -> Result<()> {
    tracing::info!("Backend: libtorch (tch)");

    // Load weights
    tracing::info!("Loading model weights (this may take a minute)...");
    let weights = weights::Weights::load(model_dir.join("model.safetensors"))?;

    // Build model components
    tracing::info!("Building encoder...");
    let encoder = encoder::ConformerEncoder::load(&weights, cfg)
        .context("Failed to load ConformerEncoder")?;

    tracing::info!("Building decoder...");
    let decoder = decoder::TransformerDecoder::load(&weights, cfg)
        .context("Failed to load TransformerDecoder")?;

    for audio_path in &args.audio_files {
        if args.audio_files.len() > 1 {
            eprintln!("[{}]", audio_path.display());
        }
        let transcript = process_audio_tch(
            audio_path,
            mel_cfg,
            &encoder,
            &decoder,
            tokenizer,
            &args.language,
            !args.no_punctuation,
            args.max_tokens,
            cfg.max_audio_clip_s,
            cfg.overlap_chunk_second,
        )?;
        println!("{}", transcript);
    }
    Ok(())
}

#[cfg(feature = "mlx")]
fn run_backend(
    args: &Args,
    model_dir: &std::path::Path,
    cfg: &config::ModelConfig,
    tokenizer: &tokenizer::Tokenizer,
    mel_cfg: &audio::MelConfig,
) -> Result<()> {
    tracing::info!("Backend: MLX (Apple Metal)");

    // Initialise the MLX runtime (GPU by default)
    mlx::stream::init_mlx(true);

    // Load weights as MLX arrays
    tracing::info!("Loading model weights (this may take a minute)...");
    let weights = mlx::weights::MlxWeights::load(model_dir.join("model.safetensors"))?;

    // Build model components
    tracing::info!("Building encoder...");
    let encoder = mlx::encoder::ConformerEncoder::load(&weights, cfg)
        .context("Failed to load ConformerEncoder")?;

    tracing::info!("Building decoder...");
    let decoder = mlx::decoder::TransformerDecoder::load(&weights, cfg)
        .context("Failed to load TransformerDecoder")?;

    for audio_path in &args.audio_files {
        if args.audio_files.len() > 1 {
            eprintln!("[{}]", audio_path.display());
        }
        let transcript = process_audio_mlx(
            audio_path,
            mel_cfg,
            &encoder,
            &decoder,
            tokenizer,
            &args.language,
            !args.no_punctuation,
            args.max_tokens,
            cfg.max_audio_clip_s,
            cfg.overlap_chunk_second,
        )?;
        println!("{}", transcript);
    }
    Ok(())
}

#[cfg(not(any(feature = "tch-backend", feature = "mlx")))]
fn run_backend(
    _args: &Args,
    _model_dir: &std::path::Path,
    _cfg: &config::ModelConfig,
    _tokenizer: &tokenizer::Tokenizer,
    _mel_cfg: &audio::MelConfig,
) -> Result<()> {
    anyhow::bail!(
        "No backend selected. Build with `--features tch-backend` or `--no-default-features --features mlx`."
    )
}

// ---------------------------------------------------------------------------
// tch backend audio processing
// ---------------------------------------------------------------------------

#[cfg(feature = "tch-backend")]
fn process_audio_tch(
    audio_path: &std::path::Path,
    mel_cfg: &audio::MelConfig,
    encoder: &encoder::ConformerEncoder,
    decoder: &decoder::TransformerDecoder,
    tokenizer: &tokenizer::Tokenizer,
    language: &str,
    punctuation: bool,
    max_tokens: usize,
    max_clip_s: f64,
    overlap_s: f64,
) -> Result<String> {
    tracing::info!("Loading audio: {:?}", audio_path);
    let samples = audio::load_audio(audio_path, mel_cfg.sample_rate)
        .with_context(|| format!("Failed to load audio: {:?}", audio_path))?;

    tracing::info!(
        "Audio loaded: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / mel_cfg.sample_rate as f64
    );

    let max_samples = (max_clip_s * mel_cfg.sample_rate as f64) as usize;
    let overlap_samples = (overlap_s * mel_cfg.sample_rate as f64) as usize;

    if samples.len() <= max_samples {
        return transcribe_chunk_tch(
            &samples,
            mel_cfg,
            encoder,
            decoder,
            tokenizer,
            language,
            punctuation,
            max_tokens,
        );
    }

    let mut transcripts: Vec<String> = Vec::new();
    let step = max_samples - overlap_samples;
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + max_samples).min(samples.len());
        let chunk = &samples[pos..end];
        let t = transcribe_chunk_tch(
            chunk,
            mel_cfg,
            encoder,
            decoder,
            tokenizer,
            language,
            punctuation,
            max_tokens,
        )?;
        transcripts.push(t);
        if end >= samples.len() {
            break;
        }
        pos += step;
    }

    Ok(transcripts.join(" "))
}

#[cfg(feature = "tch-backend")]
fn transcribe_chunk_tch(
    samples: &[f32],
    mel_cfg: &audio::MelConfig,
    encoder: &encoder::ConformerEncoder,
    decoder: &decoder::TransformerDecoder,
    tokenizer: &tokenizer::Tokenizer,
    language: &str,
    punctuation: bool,
    max_tokens: usize,
) -> Result<String> {
    tracing::debug!("Computing mel features for {} samples...", samples.len());
    let dithered = add_dither(samples, mel_cfg.dither as f32, samples.len() as u64);
    let mel = audio::compute_mel_features(&dithered, mel_cfg);
    let (flat, shape) = audio::mel_to_tensor_data(&mel);
    tracing::debug!("Mel features shape: {:?}", shape);
    let mel_tensor = Tensor::from_slice(&flat)
        .reshape(&shape)
        .to_kind(Kind::Float);
    inference::transcribe(
        &mel_tensor,
        encoder,
        decoder,
        tokenizer,
        language,
        punctuation,
        max_tokens,
    )
}

// ---------------------------------------------------------------------------
// MLX backend audio processing
// ---------------------------------------------------------------------------

#[cfg(feature = "mlx")]
fn process_audio_mlx(
    audio_path: &std::path::Path,
    mel_cfg: &audio::MelConfig,
    encoder: &mlx::encoder::ConformerEncoder,
    decoder: &mlx::decoder::TransformerDecoder,
    tokenizer: &tokenizer::Tokenizer,
    language: &str,
    punctuation: bool,
    max_tokens: usize,
    max_clip_s: f64,
    overlap_s: f64,
) -> Result<String> {
    tracing::info!("Loading audio: {:?}", audio_path);
    let samples = audio::load_audio(audio_path, mel_cfg.sample_rate)
        .with_context(|| format!("Failed to load audio: {:?}", audio_path))?;

    tracing::info!(
        "Audio loaded: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / mel_cfg.sample_rate as f64
    );

    let max_samples = (max_clip_s * mel_cfg.sample_rate as f64) as usize;
    let overlap_samples = (overlap_s * mel_cfg.sample_rate as f64) as usize;

    if samples.len() <= max_samples {
        return transcribe_chunk_mlx(
            &samples,
            mel_cfg,
            encoder,
            decoder,
            tokenizer,
            language,
            punctuation,
            max_tokens,
        );
    }

    let mut transcripts: Vec<String> = Vec::new();
    let step = max_samples - overlap_samples;
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + max_samples).min(samples.len());
        let chunk = &samples[pos..end];
        let t = transcribe_chunk_mlx(
            chunk,
            mel_cfg,
            encoder,
            decoder,
            tokenizer,
            language,
            punctuation,
            max_tokens,
        )?;
        transcripts.push(t);
        if end >= samples.len() {
            break;
        }
        pos += step;
    }

    Ok(transcripts.join(" "))
}

#[cfg(feature = "mlx")]
fn transcribe_chunk_mlx(
    samples: &[f32],
    mel_cfg: &audio::MelConfig,
    encoder: &mlx::encoder::ConformerEncoder,
    decoder: &mlx::decoder::TransformerDecoder,
    tokenizer: &tokenizer::Tokenizer,
    language: &str,
    punctuation: bool,
    max_tokens: usize,
) -> Result<String> {
    tracing::debug!("Computing mel features for {} samples...", samples.len());
    let dithered = add_dither(samples, mel_cfg.dither as f32, samples.len() as u64);
    let mel = audio::compute_mel_features(&dithered, mel_cfg);
    let (flat, shape) = audio::mel_to_tensor_data(&mel);
    tracing::debug!("Mel features shape: {:?}", shape);

    // Convert shape from i64 (tch convention in audio.rs) to i32 (MLX)
    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let mel_array = mlx::array::Array::from_data_f32(&flat, &shape_i32);

    mlx::inference::transcribe(
        &mel_array,
        encoder,
        decoder,
        tokenizer,
        language,
        punctuation,
        max_tokens,
    )
}

/// Simple deterministic dithering (matches the Python model's seeded approach).
fn add_dither(samples: &[f32], dither: f32, seed: u64) -> Vec<f32> {
    if dither == 0.0 {
        return samples.to_vec();
    }
    // Simple LCG pseudo-random for reproducibility
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = samples.to_vec();
    for s in out.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng >> 33) as f32 / (u32::MAX as f32);
        // Box-Muller transform for Gaussian noise
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (rng >> 33) as f32 / (u32::MAX as f32);
        let noise = (-2.0 * u.max(1e-38).ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        *s += dither * noise;
    }
    out
}
