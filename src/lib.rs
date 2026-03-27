//! cohere_transcribe_rs — library root.
//!
//! Exposes all shared modules so that both the CLI (`transcribe`) and
//! the API server (`transcribe-server`) binaries can reuse the same code.

pub mod audio;
pub mod config;
pub mod tokenizer;

#[cfg(feature = "tch-backend")]
pub mod decoder;
#[cfg(feature = "tch-backend")]
pub mod encoder;
#[cfg(feature = "tch-backend")]
pub mod inference;
#[cfg(feature = "tch-backend")]
pub mod weights;

#[cfg(feature = "mlx")]
pub mod mlx;
