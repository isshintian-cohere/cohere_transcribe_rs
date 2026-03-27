use anyhow::{Context, Result};
use std::path::Path;

/// Mel filterbank parameters matching the model's preprocessor config.
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub preemph: f64,
    pub dither: f64,
    pub log_zero_guard: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 128,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            fmin: 0.0,
            fmax: 8000.0,
            preemph: 0.97,
            dither: 1e-5,
            log_zero_guard: 2.0f64.powi(-24),
        }
    }
}

impl MelConfig {
    pub fn from_model_config(cfg: &crate::config::ModelConfig) -> Self {
        let pp = &cfg.preprocessor;
        let win_length = (pp.window_size * pp.sample_rate as f64).round() as usize;
        let hop_length = (pp.window_stride * pp.sample_rate as f64).round() as usize;
        Self {
            sample_rate: pp.sample_rate,
            n_mels: pp.features,
            n_fft: pp.n_fft,
            win_length,
            hop_length,
            fmin: 0.0,
            fmax: pp.sample_rate as f64 / 2.0,
            preemph: 0.97,
            dither: pp.dither,
            log_zero_guard: 2.0f64.powi(-24),
        }
    }
}

/// Load audio from a file, resample to the target sample rate, and return mono f32 samples.
pub fn load_audio(path: impl AsRef<Path>, target_sr: usize) -> Result<Vec<f32>> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path.as_ref())
        .with_context(|| format!("Cannot open audio file {:?}", path.as_ref()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found")?;

    let src_sr = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")? as usize;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create audio decoder")?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;

        let buf_f32: Vec<f32> = match &decoded {
            AudioBufferRef::F32(buf) => {
                let n = buf.frames();
                let mut out = Vec::with_capacity(n);
                for frame in 0..n {
                    let mut sample = 0.0f32;
                    for ch in 0..channels {
                        sample += buf.chan(ch)[frame];
                    }
                    out.push(sample / channels as f32);
                }
                out
            }
            AudioBufferRef::S16(buf) => {
                let n = buf.frames();
                let mut out = Vec::with_capacity(n);
                for frame in 0..n {
                    let mut sample = 0.0f32;
                    for ch in 0..channels {
                        sample += buf.chan(ch)[frame] as f32 / 32768.0;
                    }
                    out.push(sample / channels as f32);
                }
                out
            }
            AudioBufferRef::S32(buf) => {
                let n = buf.frames();
                let mut out = Vec::with_capacity(n);
                for frame in 0..n {
                    let mut sample = 0.0f32;
                    for ch in 0..channels {
                        sample += buf.chan(ch)[frame] as f32 / 2147483648.0;
                    }
                    out.push(sample / channels as f32);
                }
                out
            }
            AudioBufferRef::U8(buf) => {
                let n = buf.frames();
                let mut out = Vec::with_capacity(n);
                for frame in 0..n {
                    let mut sample = 0.0f32;
                    for ch in 0..channels {
                        sample += (buf.chan(ch)[frame] as f32 - 128.0) / 128.0;
                    }
                    out.push(sample / channels as f32);
                }
                out
            }
            _ => {
                // Convert via interleaved float for other formats
                let mut tmp = decoded.make_equivalent::<f32>();
                decoded.convert(&mut tmp);
                let n = tmp.frames();
                let mut out = Vec::with_capacity(n);
                for frame in 0..n {
                    let mut sample = 0.0f32;
                    for ch in 0..channels {
                        sample += tmp.chan(ch)[frame];
                    }
                    out.push(sample / channels as f32);
                }
                out
            }
        };

        samples.extend(buf_f32);
    }

    // Resample if needed
    if src_sr != target_sr {
        samples = resample(&samples, src_sr, target_sr)?;
    }

    Ok(samples)
}

fn resample(input: &[f32], src_sr: usize, dst_sr: usize) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};

    let ratio = dst_sr as f64 / src_sr as f64;
    let chunk_size = 4096;

    let mut resampler = FftFixedIn::<f32>::new(src_sr, dst_sr, chunk_size, 2, 1)
        .context("Failed to create resampler")?;

    let mut output = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let mut chunk = input[pos..end].to_vec();
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let out = resampler
            .process(&[chunk], None)
            .context("Resampling failed")?;
        output.extend_from_slice(&out[0]);
        pos += chunk_size;
    }

    let expected = (input.len() as f64 * ratio).ceil() as usize;
    output.truncate(expected);

    Ok(output)
}

/// Compute mel filterbank matrix: (n_mels, n_fft/2+1).
/// Follows librosa.filters.mel with norm='slaney'.
pub fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    let hz_to_mel = |f: f64| -> f64 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f64| -> f64 { 700.0 * (10.0f64.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 center frequencies in mel space
    let mel_pts: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    // FFT bin center frequencies
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let mut fb = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let f_lower = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_upper = hz_pts[m + 2];
        // Slaney normalization factor
        let norm = 2.0 / (f_upper - f_lower);

        for k in 0..n_freqs {
            let f = fft_freqs[k];
            let w = if f >= f_lower && f <= f_center {
                (f - f_lower) / (f_center - f_lower)
            } else if f > f_center && f <= f_upper {
                (f_upper - f) / (f_upper - f_center)
            } else {
                0.0
            };
            fb[m * n_freqs + k] = (w * norm) as f32;
        }
    }

    fb
}

/// Compute mel log-filterbank features from raw 16kHz mono audio.
/// Returns a (n_mels, T) matrix of features.
pub fn compute_mel_features(audio: &[f32], cfg: &MelConfig) -> Vec<Vec<f32>> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = audio.len();

    // 1. Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1]
    let mut preemphasized = Vec::with_capacity(n);
    preemphasized.push(audio[0]);
    for i in 1..n {
        preemphasized.push(audio[i] - (cfg.preemph as f32) * audio[i - 1]);
    }

    // 2. Hann window (non-periodic, win_length)
    let win_length = cfg.win_length;
    let hann: Vec<f32> = (0..win_length)
        .map(|i| {
            let pi = std::f32::consts::PI;
            0.5 * (1.0 - (2.0 * pi * i as f32 / (win_length - 1) as f32).cos())
        })
        .collect();

    // 3. STFT with center padding (pad n_fft/2 on each side with zeros, "constant" mode)
    let pad = cfg.n_fft / 2;
    let mut padded = vec![0.0f32; n + 2 * pad];
    padded[pad..pad + n].copy_from_slice(&preemphasized);

    let padded_n = padded.len();
    let n_frames = 1 + (padded_n.saturating_sub(cfg.n_fft)) / cfg.hop_length;

    let n_freqs = cfg.n_fft / 2 + 1;
    let mut power_spectrum = vec![vec![0.0f32; n_frames]; n_freqs];

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cfg.n_fft);
    let mut buf: Vec<Complex<f32>> = vec![Complex::default(); cfg.n_fft];

    for frame_idx in 0..n_frames {
        let start = frame_idx * cfg.hop_length;
        let _end = start + cfg.n_fft;

        // Zero-fill frame buffer
        for c in buf.iter_mut() {
            *c = Complex::default();
        }

        // Copy windowed samples (win_length centered in n_fft)
        let offset = (cfg.n_fft - win_length) / 2;
        for i in 0..win_length {
            let sig_pos = start + i;
            if sig_pos < padded_n {
                buf[offset + i].re = padded[sig_pos] * hann[i];
            }
        }

        fft.process(&mut buf);

        for k in 0..n_freqs {
            let re = buf[k].re;
            let im = buf[k].im;
            power_spectrum[k][frame_idx] = re * re + im * im;
        }
    }

    // 4. Mel filterbank: (n_mels, n_freqs) x power_spectrum
    let mel_fb = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

    let log_guard = cfg.log_zero_guard as f32;
    let mut mel_features: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_mels);

    for m in 0..cfg.n_mels {
        let mut mel_row = vec![0.0f32; n_frames];
        for k in 0..n_freqs {
            let fb_val = mel_fb[m * n_freqs + k];
            if fb_val == 0.0 {
                continue;
            }
            for t in 0..n_frames {
                mel_row[t] += fb_val * power_spectrum[k][t];
            }
        }

        // 5. Log
        for v in mel_row.iter_mut() {
            *v = (*v + log_guard).ln();
        }

        mel_features.push(mel_row);
    }

    // 6. Per-feature normalization: (x - mean) / (std + 1e-5)
    let eps = 1e-5f32;
    for row in mel_features.iter_mut() {
        let n_t = row.len() as f32;
        let mean: f32 = row.iter().sum::<f32>() / n_t;
        let var: f32 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (n_t - 1.0).max(1.0);
        let std = var.sqrt() + eps;
        for v in row.iter_mut() {
            *v = (*v - mean) / std;
        }
    }

    mel_features
}

/// Convert mel features Vec<Vec<f32>> of shape (n_mels, T) to a flat f32 Vec.
pub fn mel_to_tensor_data(mel: &[Vec<f32>]) -> (Vec<f32>, Vec<i64>) {
    let n_mels = mel.len();
    let n_frames = if n_mels > 0 { mel[0].len() } else { 0 };
    let mut flat = Vec::with_capacity(n_mels * n_frames);
    for row in mel {
        flat.extend_from_slice(row);
    }
    // Shape: (1, n_mels, n_frames) for batch_size=1
    (flat, vec![1, n_mels as i64, n_frames as i64])
}
