//! OpenAI-compatible transcription API server.
//!
//! Loads the model once at startup, then serves:
//!   POST /v1/audio/transcriptions   — OpenAI-compatible multipart upload
//!   GET  /health                    — liveness probe
//!
//! Usage:
//!   transcribe-server --model-dir models/cohere-transcribe-03-2026
//!   transcribe-server --model-dir models/... --host 127.0.0.1 --port 8080
//!
//! Compatible with OpenAI clients — drop-in replacement for the Whisper API.

use std::io::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use cohere_transcribe_rs::{audio, config, tokenizer};

#[cfg(feature = "tch-backend")]
use cohere_transcribe_rs::{decoder, encoder, inference, weights};
#[cfg(feature = "tch-backend")]
use tch::{Kind, Tensor};

#[cfg(feature = "mlx")]
use cohere_transcribe_rs::mlx;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "transcribe-server",
    about = "OpenAI-compatible transcription API server",
    long_about = "Serves the Cohere Transcribe model as an OpenAI-compatible REST API.\n\n\
                  Compatible endpoint: POST /v1/audio/transcriptions\n\n\
                  Drop-in replacement for OpenAI Whisper API — works with any OpenAI client."
)]
struct Args {
    /// Path to the model directory
    #[arg(short, long)]
    model_dir: std::path::PathBuf,

    /// Host address to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Default language when the client does not specify one
    #[arg(short, long, default_value = "en")]
    language: String,

    /// Maximum tokens to generate per audio segment
    #[arg(long, default_value = "448")]
    max_tokens: usize,

    /// Log verbosity (-v for info, -vv for debug)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared model state
// ─────────────────────────────────────────────────────────────────────────────

/// All model components needed to run inference.
/// Wrapped in Arc<Mutex<_>> so concurrent requests serialize through one model.
struct ModelState {
    cfg: config::ModelConfig,
    tokenizer: tokenizer::Tokenizer,
    mel_cfg: audio::MelConfig,
    default_language: String,
    max_tokens: usize,

    #[cfg(feature = "tch-backend")]
    encoder: encoder::ConformerEncoder,
    #[cfg(feature = "tch-backend")]
    decoder: decoder::TransformerDecoder,

    #[cfg(feature = "mlx")]
    encoder: mlx::encoder::ConformerEncoder,
    #[cfg(feature = "mlx")]
    decoder: mlx::decoder::TransformerDecoder,
}

type SharedState = Arc<Mutex<ModelState>>;

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI API types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Serialize)]
struct VerboseTranscriptionResponse {
    task: &'static str,
    language: String,
    duration: f64,
    text: String,
    segments: Vec<Segment>,
}

#[derive(Serialize)]
struct Segment {
    id: u32,
    seek: u32,
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<u32>,
    temperature: f64,
    avg_logprob: f64,
    compression_ratio: f64,
    no_speech_prob: f64,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

/// Wraps anyhow errors as HTTP 500 responses.
struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error: {:#}", self.0),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────────────────────

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// POST /v1/audio/transcriptions
///
/// Accepts multipart/form-data with these fields (OpenAI compatible):
///   file            — audio bytes (required)
///   model           — model name (required by spec; any string accepted)
///   language        — ISO-639-1 code, e.g. "en" (optional)
///   response_format — "json" | "text" | "verbose_json" | "srt" | "vtt" (optional, default "json")
///   temperature     — float (optional, ignored — always greedy)
///   prompt          — string (optional, ignored)
async fn transcribe_audio(
    State(state): State<SharedState>,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();
    let mut punctuation = true;

    // Parse multipart fields
    while let Some(field) = multipart.next_field().await? {
        match field.name() {
            Some("file") => {
                filename = field.file_name().map(|s| s.to_string());
                audio_bytes = Some(field.bytes().await?.to_vec());
            }
            Some("language") => {
                language = Some(field.text().await?);
            }
            Some("response_format") => {
                response_format = field.text().await?;
            }
            Some("temperature") | Some("prompt") | Some("model") => {
                // Consume and ignore
                let _ = field.bytes().await?;
            }
            _ => {
                let _ = field.bytes().await?;
            }
        }
    }

    let audio_data = audio_bytes.ok_or_else(|| {
        AppError(anyhow::anyhow!(
            "Missing required field 'file' in multipart form"
        ))
    })?;

    // Write audio bytes to a temporary file so the audio loader can read it
    let ext = filename
        .as_deref()
        .and_then(|f| f.rsplit('.').next())
        .unwrap_or("wav");
    let mut tmp = tempfile::Builder::new()
        .suffix(&format!(".{}", ext))
        .tempfile()?;
    tmp.write_all(&audio_data)?;
    let tmp_path = tmp.path().to_path_buf();

    // Run inference (acquires the model lock)
    let state = state.lock().await;
    let lang = language
        .as_deref()
        .unwrap_or(&state.default_language)
        .to_string();

    // Validate language
    if !state.cfg.supported_languages.contains(&lang) {
        return Err(AppError(anyhow::anyhow!(
            "Unsupported language '{}'. Supported: {:?}",
            lang,
            state.cfg.supported_languages
        )));
    }

    let samples = audio::load_audio(&tmp_path, state.mel_cfg.sample_rate)
        .with_context(|| "Failed to decode audio")?;

    let duration_s = samples.len() as f64 / state.mel_cfg.sample_rate as f64;
    let max_samples = (state.cfg.max_audio_clip_s * state.mel_cfg.sample_rate as f64) as usize;
    let overlap_samples =
        (state.cfg.overlap_chunk_second * state.mel_cfg.sample_rate as f64) as usize;

    let text = if samples.len() <= max_samples {
        transcribe_chunk(&samples, &state, &lang, punctuation)?
    } else {
        let mut parts = Vec::new();
        let step = max_samples - overlap_samples;
        let mut pos = 0;
        while pos < samples.len() {
            let end = (pos + max_samples).min(samples.len());
            parts.push(transcribe_chunk(&samples[pos..end], &state, &lang, punctuation)?);
            if end >= samples.len() {
                break;
            }
            pos += step;
        }
        parts.join(" ")
    };

    drop(state); // release model lock before building response

    let response: Response = match response_format.as_str() {
        "text" => (StatusCode::OK, text).into_response(),

        "verbose_json" => {
            let segment = Segment {
                id: 0,
                seek: 0,
                start: 0.0,
                end: duration_s,
                text: text.clone(),
                tokens: vec![],
                temperature: 0.0,
                avg_logprob: 0.0,
                compression_ratio: 1.0,
                no_speech_prob: 0.0,
            };
            Json(VerboseTranscriptionResponse {
                task: "transcribe",
                language: lang,
                duration: duration_s,
                text,
                segments: vec![segment],
            })
            .into_response()
        }

        "srt" => {
            let srt = format!(
                "1\n{} --> {}\n{}\n",
                format_srt_time(0.0),
                format_srt_time(duration_s),
                text
            );
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                srt,
            )
                .into_response()
        }

        "vtt" => {
            let vtt = format!(
                "WEBVTT\n\n{} --> {}\n{}\n",
                format_vtt_time(0.0),
                format_vtt_time(duration_s),
                text
            );
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/vtt; charset=utf-8")],
                vtt,
            )
                .into_response()
        }

        // Default: "json"
        _ => Json(TranscriptionResponse { text }).into_response(),
    };

    Ok(response)
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend-specific inference
// ─────────────────────────────────────────────────────────────────────────────

fn transcribe_chunk(
    samples: &[f32],
    state: &ModelState,
    language: &str,
    punctuation: bool,
) -> Result<String> {
    let dithered = add_dither(samples, state.mel_cfg.dither as f32, samples.len() as u64);
    let mel = audio::compute_mel_features(&dithered, &state.mel_cfg);
    let (flat, shape) = audio::mel_to_tensor_data(&mel);

    #[cfg(feature = "tch-backend")]
    {
        let mel_tensor = Tensor::from_slice(&flat)
            .reshape(&shape)
            .to_kind(Kind::Float);
        return inference::transcribe(
            &mel_tensor,
            &state.encoder,
            &state.decoder,
            &state.tokenizer,
            language,
            punctuation,
            state.max_tokens,
        );
    }

    #[cfg(feature = "mlx")]
    {
        let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mel_array = mlx::array::Array::from_data_f32(&flat, &shape_i32);
        return mlx::inference::transcribe(
            &mel_array,
            &state.encoder,
            &state.decoder,
            &state.tokenizer,
            language,
            punctuation,
            state.max_tokens,
        );
    }

    #[allow(unreachable_code)]
    anyhow::bail!("No backend compiled — build with --features tch-backend or mlx")
}

// ─────────────────────────────────────────────────────────────────────────────
// Time formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

fn format_srt_time(seconds: f64) -> String {
    let h = (seconds / 3600.0) as u64;
    let m = ((seconds % 3600.0) / 60.0) as u64;
    let s = (seconds % 60.0) as u64;
    let ms = ((seconds.fract()) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

fn format_vtt_time(seconds: f64) -> String {
    let h = (seconds / 3600.0) as u64;
    let m = ((seconds % 3600.0) / 60.0) as u64;
    let s = (seconds % 60.0) as u64;
    let ms = ((seconds.fract()) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dithering (shared with CLI — small noise for numerical stability)
// ─────────────────────────────────────────────────────────────────────────────

fn add_dither(samples: &[f32], dither: f32, seed: u64) -> Vec<f32> {
    if dither == 0.0 {
        return samples.to_vec();
    }
    let mut rng = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = samples.to_vec();
    for s in out.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng >> 33) as f32 / (u32::MAX as f32);
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (rng >> 33) as f32 / (u32::MAX as f32);
        let noise =
            (-2.0 * u.max(1e-38).ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
        *s += dither * noise;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Model loading
// ─────────────────────────────────────────────────────────────────────────────

fn load_model(args: &Args) -> Result<ModelState> {
    let model_dir = &args.model_dir;

    anyhow::ensure!(model_dir.exists(), "Model directory not found: {:?}", model_dir);
    for f in &["config.json", "model.safetensors", "vocab.json"] {
        anyhow::ensure!(
            model_dir.join(f).exists(),
            "Missing '{}' in {:?}",
            f,
            model_dir
        );
    }

    tracing::info!("Loading model config…");
    let cfg = config::ModelConfig::load(model_dir)?;

    tracing::info!("Loading tokenizer…");
    let tokenizer = tokenizer::Tokenizer::load(model_dir)?;

    let mel_cfg = audio::MelConfig::from_model_config(&cfg);

    anyhow::ensure!(
        cfg.supported_languages.contains(&args.language),
        "Default language '{}' is not supported. Supported: {:?}",
        args.language,
        cfg.supported_languages
    );

    #[cfg(feature = "tch-backend")]
    {
        tracing::info!("Loading weights (tch-backend)…");
        let w = weights::Weights::load(model_dir.join("model.safetensors"))?;
        tracing::info!("Building encoder…");
        let enc = encoder::ConformerEncoder::load(&w, &cfg)
            .context("Failed to build ConformerEncoder")?;
        tracing::info!("Building decoder…");
        let dec = decoder::TransformerDecoder::load(&w, &cfg)
            .context("Failed to build TransformerDecoder")?;
        tracing::info!("Model ready.");
        return Ok(ModelState {
            cfg,
            tokenizer,
            mel_cfg,
            default_language: args.language.clone(),
            max_tokens: args.max_tokens,
            encoder: enc,
            decoder: dec,
        });
    }

    #[cfg(feature = "mlx")]
    {
        mlx::stream::init_mlx(true);
        tracing::info!("Loading weights (mlx)…");
        let w = mlx::weights::MlxWeights::load(model_dir.join("model.safetensors"))?;
        tracing::info!("Building encoder…");
        let enc = mlx::encoder::ConformerEncoder::load(&w, &cfg)
            .context("Failed to build ConformerEncoder")?;
        tracing::info!("Building decoder…");
        let dec = mlx::decoder::TransformerDecoder::load(&w, &cfg)
            .context("Failed to build TransformerDecoder")?;
        tracing::info!("Model ready.");
        return Ok(ModelState {
            cfg,
            tokenizer,
            mel_cfg,
            default_language: args.language.clone(),
            max_tokens: args.max_tokens,
            encoder: enc,
            decoder: dec,
        });
    }

    #[allow(unreachable_code)]
    anyhow::bail!("No backend compiled — build with --features tch-backend or mlx")
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = match args.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    // Load model synchronously before accepting requests
    let model_state = load_model(&args)?;
    let state: SharedState = Arc::new(Mutex::new(model_state));

    let app = Router::new()
        .route("/v1/audio/transcriptions", post(transcribe_audio))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("Invalid host/port")?;

    tracing::warn!("Listening on http://{}", addr);
    tracing::warn!("Endpoint: POST http://{}/v1/audio/transcriptions", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
