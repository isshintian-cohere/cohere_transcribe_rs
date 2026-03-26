//! MLX backend — macOS/Apple Silicon only.
//!
//! Enabled with: cargo build --features mlx
//! Requires libmlxc and Metal frameworks (see build.rs and README.md).

// The tch-backend and mlx features are mutually exclusive.
#[cfg(all(feature = "tch-backend", feature = "mlx"))]
compile_error!(
    "Features 'tch-backend' and 'mlx' are mutually exclusive. \
     Build with `--no-default-features --features mlx` for the MLX backend."
);

pub mod array;
pub mod decoder;
pub mod encoder;
pub mod ffi;
pub mod inference;
pub mod ops;
pub mod stream;
pub mod weights;
