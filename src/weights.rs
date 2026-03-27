use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use tch::Tensor;

/// Convert a BFloat16 u16 bit pattern to f32.
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub struct Weights {
    _data: Vec<u8>,
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        tracing::info!("Loading weights from {:?}", path.as_ref());
        let data = std::fs::read(path.as_ref())
            .with_context(|| format!("Cannot read safetensors at {:?}", path.as_ref()))?;

        let st = SafeTensors::deserialize(&data)?;
        let mut tensors = HashMap::new();

        for (name, view) in st.tensors() {
            let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
            let raw = view.data();

            let tensor = match view.dtype() {
                safetensors::Dtype::BF16 => {
                    let f32_vals: Vec<f32> = raw
                        .chunks_exact(2)
                        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                        .collect();
                    Tensor::from_slice(&f32_vals).reshape(&shape)
                }
                safetensors::Dtype::F32 => {
                    let f32_vals: Vec<f32> = raw
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Tensor::from_slice(&f32_vals).reshape(&shape)
                }
                safetensors::Dtype::F16 => {
                    let f32_vals: Vec<f32> = raw
                        .chunks_exact(2)
                        .map(|b| {
                            let half = u16::from_le_bytes([b[0], b[1]]);
                            half_to_f32(half)
                        })
                        .collect();
                    Tensor::from_slice(&f32_vals).reshape(&shape)
                }
                safetensors::Dtype::I64 => {
                    let i64_vals: Vec<i64> = raw
                        .chunks_exact(8)
                        .map(|b| {
                            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        })
                        .collect();
                    Tensor::from_slice(&i64_vals).reshape(&shape)
                }
                _ => {
                    // Skip unsupported types (e.g. num_batches_tracked as i64 is fine above)
                    continue;
                }
            };

            tensors.insert(name.to_string(), tensor);
        }

        tracing::info!("Loaded {} tensors", tensors.len());
        Ok(Weights { _data: data, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("Missing weight: '{}'", name))
    }

    pub fn get_opt(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}

fn half_to_f32(half: u16) -> f32 {
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
    let exp_f32 = exp + (127 - 15);
    f32::from_bits(sign | (exp_f32 << 23) | (mant << 13))
}
