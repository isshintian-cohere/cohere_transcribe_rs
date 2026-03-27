//! MLX weight loader — reads SafeTensors and materialises weights as MLX arrays.
//!
//! Reuses the same BF16→F32 conversion as the tch backend; weights are loaded
//! into CPU memory first, then transferred to the Metal device via `Array::from_data_f32`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

use super::array::Array;

/// Convert a BFloat16 bit pattern to f32 (same as in src/weights.rs).
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convert a Float16 bit pattern to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) as u32) << 31;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;
    if exp == 0 && mant == 0 {
        return f32::from_bits(sign);
    }
    if exp == 31 {
        return if mant == 0 {
            f32::from_bits(sign | 0x7F800000)
        } else {
            f32::NAN
        };
    }
    f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (mant << 13))
}

pub struct MlxWeights {
    // Keep the raw bytes alive — MLX arrays may hold pointers into this buffer.
    _data: Vec<u8>,
    tensors: HashMap<String, Array>,
}

impl MlxWeights {
    /// Load `model.safetensors` from `model_dir`, decode every tensor to F32,
    /// and upload it to the default MLX stream (GPU).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        tracing::info!("Loading MLX weights from {:?}", path.as_ref());
        let data = std::fs::read(path.as_ref())
            .with_context(|| format!("Cannot read safetensors at {:?}", path.as_ref()))?;

        let st = SafeTensors::deserialize(&data)?;
        let mut tensors = HashMap::new();

        for (name, view) in st.tensors() {
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let raw = view.data();

            let f32_vals: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => raw
                    .chunks_exact(2)
                    .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
                safetensors::Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
                _ => continue, // skip unsupported dtypes (e.g. i64 scalars)
            };

            tensors.insert(name.to_string(), Array::from_data_f32(&f32_vals, &shape));
        }

        tracing::info!("Loaded {} MLX tensors", tensors.len());
        Ok(Self {
            _data: data,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.tensors
            .get(name)
            .with_context(|| format!("Missing weight: '{}'", name))
    }
}
