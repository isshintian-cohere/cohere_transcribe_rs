/// Conformer encoder implementing the architecture from modeling_cohere_asr.py.
///
/// Key components:
///   ConvSubsampling  → downsamples mel frames by factor 8
///   RelPositionalEncoding  → relative sinusoidal positional embeddings
///   ConformerLayer × 48  → FF + RelPosAttn + ConvModule + FF
use anyhow::{Context, Result};
use tch::{Kind, Tensor};

use crate::config::ModelConfig;
use crate::weights::Weights;

// ---------------------------------------------------------------------------
// Helper: Linear (y = x W^T + b)
// ---------------------------------------------------------------------------
fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.matmul(&w.tr()) + b
}

// ---------------------------------------------------------------------------
// Helper: Layer norm
// ---------------------------------------------------------------------------
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.layer_norm(&[w.size()[0]], Some(w), Some(b), 1e-5, false)
}

// ---------------------------------------------------------------------------
// Helper: Batch norm 1d (eval mode)
// x: (B, C, T)
// ---------------------------------------------------------------------------
fn batch_norm_eval(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
) -> Tensor {
    let eps = 1e-5f64;
    let inv_std = (running_var + eps).sqrt().reciprocal();
    let x_norm = (x - running_mean.view([1, -1, 1])) * inv_std.view([1, -1, 1]);
    x_norm * weight.view([1, -1, 1]) + bias.view([1, -1, 1])
}

// ---------------------------------------------------------------------------
// ConvSubsampling
//   conv[0]: Conv2d(1, 256, 3, stride=2, pad=1)
//   conv[2]: Conv2d(256, 256, 3, stride=2, pad=1, groups=256)  [depthwise]
//   conv[3]: Conv2d(256, 256, 1)                               [pointwise]
//   conv[5]: Conv2d(256, 256, 3, stride=2, pad=1, groups=256)  [depthwise]
//   conv[6]: Conv2d(256, 256, 1)                               [pointwise]
//   out: Linear(256 * (n_mels / subsampling_factor), d_model)
// ---------------------------------------------------------------------------
struct ConvSubsampling {
    c0_w: Tensor,
    c0_b: Tensor,
    c2_w: Tensor,
    c2_b: Tensor,
    c3_w: Tensor,
    c3_b: Tensor,
    c5_w: Tensor,
    c5_b: Tensor,
    c6_w: Tensor,
    c6_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    conv_channels: i64,
}

impl ConvSubsampling {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            c0_w: w("conv.0.weight")?,
            c0_b: w("conv.0.bias")?,
            c2_w: w("conv.2.weight")?,
            c2_b: w("conv.2.bias")?,
            c3_w: w("conv.3.weight")?,
            c3_b: w("conv.3.bias")?,
            c5_w: w("conv.5.weight")?,
            c5_b: w("conv.5.bias")?,
            c6_w: w("conv.6.weight")?,
            c6_b: w("conv.6.bias")?,
            out_w: w("out.weight")?,
            out_b: w("out.bias")?,
            conv_channels: 256,
        })
    }

    /// Forward pass.
    /// x: (1, n_mels, T)  → (1, T', d_model)
    /// Returns (out, T')
    fn forward(&self, x: &Tensor) -> (Tensor, i64) {
        // Reshape: (1, n_mels, T) → (1, 1, T, n_mels)
        let x = x.transpose(1, 2).unsqueeze(1);

        // Three stride-2 Conv2d passes with ReLU
        let x = x
            .conv2d(&self.c0_w, Some(&self.c0_b), &[2, 2], &[1, 1], &[1, 1], 1)
            .relu();
        let x = x
            .conv2d(
                &self.c2_w,
                Some(&self.c2_b),
                &[2, 2],
                &[1, 1],
                &[1, 1],
                self.conv_channels,
            )
            .conv2d(&self.c3_w, Some(&self.c3_b), &[1, 1], &[0, 0], &[1, 1], 1)
            .relu();
        let x = x
            .conv2d(
                &self.c5_w,
                Some(&self.c5_b),
                &[2, 2],
                &[1, 1],
                &[1, 1],
                self.conv_channels,
            )
            .conv2d(&self.c6_w, Some(&self.c6_b), &[1, 1], &[0, 0], &[1, 1], 1)
            .relu();

        // x shape: (1, 256, T', n_mels/8)
        let sz = x.size();
        let (b, _c, t, _f) = (sz[0], sz[1], sz[2], sz[3]);
        // Transpose and reshape: (1, T', 256 * (n_mels/8))
        let x = x.transpose(1, 2).reshape(&[b, t, -1]);
        // Linear projection
        let out = linear(&x, &self.out_w, &self.out_b);
        (out, t)
    }
}

// ---------------------------------------------------------------------------
// Relative Positional Encoding (RelPositionalEncoding)
//   Produces positional embeddings for positions [L-1, L-2, ..., 0, ..., -(L-1)]
//   where L = sequence length.
// ---------------------------------------------------------------------------
fn rel_positional_encoding(length: usize, d_model: usize) -> Tensor {
    let n_pos = 2 * length - 1; // Total positions
    let mut pe = vec![0.0f32; n_pos * d_model];

    for i in 0..n_pos {
        // Position value: from (length-1) down to -(length-1)
        let pos = (length as i64 - 1 - i as i64) as f64;
        for k in (0..d_model).step_by(2) {
            let div = ((k as f64) * -(10000.0f64.ln()) / d_model as f64).exp();
            pe[i * d_model + k] = (pos * div).sin() as f32;
            if k + 1 < d_model {
                pe[i * d_model + k + 1] = (pos * div).cos() as f32;
            }
        }
    }

    // Shape: (1, 2L-1, d_model)
    Tensor::from_slice(&pe).reshape(&[1, n_pos as i64, d_model as i64])
}

// ---------------------------------------------------------------------------
// ConformerFeedForward (½-scaled residual)
//   Linear → SiLU → Linear
// ---------------------------------------------------------------------------
struct FeedForward {
    l1_w: Tensor,
    l1_b: Tensor,
    l2_w: Tensor,
    l2_b: Tensor,
}

impl FeedForward {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            l1_w: w("linear1.weight")?,
            l1_b: w("linear1.bias")?,
            l2_w: w("linear2.weight")?,
            l2_b: w("linear2.bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h = linear(x, &self.l1_w, &self.l1_b).silu();
        linear(&h, &self.l2_w, &self.l2_b)
    }
}

// ---------------------------------------------------------------------------
// ConformerConvolution
//   pointwise_conv1 → GLU → depthwise_conv → BatchNorm → SiLU → pointwise_conv2
// ---------------------------------------------------------------------------
struct ConformerConv {
    pw1_w: Tensor,
    pw1_b: Tensor,
    dw_w: Tensor,
    dw_b: Tensor,
    bn_w: Tensor,
    bn_b: Tensor,
    bn_rm: Tensor,
    bn_rv: Tensor,
    pw2_w: Tensor,
    pw2_b: Tensor,
    d_model: i64,
}

impl ConformerConv {
    fn load(weights: &Weights, prefix: &str, d_model: i64) -> Result<Self> {
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            pw1_w: w("pointwise_conv1.weight")?,
            pw1_b: w("pointwise_conv1.bias")?,
            dw_w: w("depthwise_conv.weight")?,
            dw_b: w("depthwise_conv.bias")?,
            bn_w: w("batch_norm.weight")?,
            bn_b: w("batch_norm.bias")?,
            bn_rm: w("batch_norm.running_mean")?,
            bn_rv: w("batch_norm.running_var")?,
            pw2_w: w("pointwise_conv2.weight")?,
            pw2_b: w("pointwise_conv2.bias")?,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: (B, T, d_model) → transpose → (B, d_model, T)
        let x = x.transpose(1, 2);

        // Pointwise conv 1: (B, 2*d_model, T)
        let kernel_size = self.dw_w.size()[2]; // kernel_size from depthwise weight
        let x = x.conv1d(&self.pw1_w, Some(&self.pw1_b), &[1], &[0], &[1], 1);

        // GLU along dim 1: x[:, :d] * sigmoid(x[:, d:])
        let (a, b) = (
            x.narrow(1, 0, self.d_model),
            x.narrow(1, self.d_model, self.d_model),
        );
        let x = a * b.sigmoid();

        // Depthwise conv
        let pad = (kernel_size - 1) / 2;
        let x = x.conv1d(
            &self.dw_w,
            Some(&self.dw_b),
            &[1],
            &[pad],
            &[1],
            self.d_model,
        );

        // BatchNorm (eval mode)
        let x = batch_norm_eval(&x, &self.bn_w, &self.bn_b, &self.bn_rm, &self.bn_rv);

        // SiLU
        let x = x.silu();

        // Pointwise conv 2: (B, d_model, T)
        let x = x.conv1d(&self.pw2_w, Some(&self.pw2_b), &[1], &[0], &[1], 1);

        // Transpose back: (B, T, d_model)
        x.transpose(1, 2)
    }
}

// ---------------------------------------------------------------------------
// RelPositionMultiHeadAttention
// ---------------------------------------------------------------------------
struct RelPosAttn {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    pos_w: Tensor, // no bias
    out_w: Tensor,
    out_b: Tensor,
    pos_bias_u: Tensor, // (n_heads, d_k)
    pos_bias_v: Tensor, // (n_heads, d_k)
    n_heads: i64,
    d_k: i64,
    scale: f64,
}

impl RelPosAttn {
    fn load(weights: &Weights, prefix: &str, n_heads: i64, d_model: i64) -> Result<Self> {
        let d_k = d_model / n_heads;
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            q_w: w("linear_q.weight")?,
            q_b: w("linear_q.bias")?,
            k_w: w("linear_k.weight")?,
            k_b: w("linear_k.bias")?,
            v_w: w("linear_v.weight")?,
            v_b: w("linear_v.bias")?,
            pos_w: w("linear_pos.weight")?,
            out_w: w("linear_out.weight")?,
            out_b: w("linear_out.bias")?,
            pos_bias_u: w("pos_bias_u")?,
            pos_bias_v: w("pos_bias_v")?,
            n_heads,
            d_k,
            scale: (d_k as f64).powf(-0.5),
        })
    }

    /// Relative shift: x (B, H, T, 2T-1) → (B, H, T, T)
    fn rel_shift(&self, x: &Tensor) -> Tensor {
        let (b, h, t, _) = x.size4().unwrap();
        // Pad one column on the left
        let x = x.constant_pad_nd(&[1, 0]);
        // View as (B, H, 2T, T) then slice
        let x = x.view([b, h, -1, t]);
        x.narrow(2, 1, t)
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Tensor {
        let (b, t, _) = x.size3().unwrap();

        let reshape =
            |z: &Tensor| -> Tensor { z.view([b, t, self.n_heads, self.d_k]).transpose(1, 2) };

        let q = reshape(&linear(x, &self.q_w, &self.q_b));
        let k = reshape(&linear(x, &self.k_w, &self.k_b));
        let v = reshape(&linear(x, &self.v_w, &self.v_b));

        // pos_emb: (1, 2T-1, d_model) → (1, 2T-1, H, d_k) → (1, H, 2T-1, d_k)
        let n_pos = pos_emb.size()[1];
        let p = linear(
            pos_emb,
            &self.pos_w,
            &Tensor::zeros(&[1], (Kind::Float, x.device())),
        )
        .view([1, n_pos, self.n_heads, self.d_k])
        .transpose(1, 2);

        // pos_bias_u/v: (n_heads, d_k) → (1, n_heads, 1, d_k) for broadcasting
        let u = self.pos_bias_u.view([1, self.n_heads, 1, self.d_k]);
        let v_bias = self.pos_bias_v.view([1, self.n_heads, 1, self.d_k]);

        let q_with_u = &q + &u;
        let q_with_v = &q + &v_bias;

        // matrix_ac: (B, H, T, T)
        let matrix_ac = q_with_u.matmul(&k.transpose(-2, -1));
        // matrix_bd: (B, H, T, 2T-1) → rel_shift → (B, H, T, T)
        let matrix_bd = q_with_v.matmul(&p.transpose(-2, -1));
        let matrix_bd = self.rel_shift(&matrix_bd);

        let scores = (matrix_ac + matrix_bd) * self.scale;
        let attn = scores.softmax(-1, Kind::Float);
        let out = attn.matmul(&v);

        // Reshape back: (B, H, T, d_k) → (B, T, d_model)
        let out = out
            .transpose(1, 2)
            .contiguous()
            .view([b, t, self.n_heads * self.d_k]);
        linear(&out, &self.out_w, &self.out_b)
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------
struct ConformerLayer {
    norm_ff1: (Tensor, Tensor),
    ff1: FeedForward,
    norm_self_att: (Tensor, Tensor),
    self_attn: RelPosAttn,
    norm_conv: (Tensor, Tensor),
    conv: ConformerConv,
    norm_ff2: (Tensor, Tensor),
    ff2: FeedForward,
    norm_out: (Tensor, Tensor),
}

impl ConformerLayer {
    fn load(weights: &Weights, prefix: &str, n_heads: i64, d_model: i64) -> Result<Self> {
        let norm = |n: &str| -> Result<(Tensor, Tensor)> {
            let key = format!("{}{}", prefix, n);
            Ok((
                weights.get(&format!("{}.weight", key))?.shallow_clone(),
                weights.get(&format!("{}.bias", key))?.shallow_clone(),
            ))
        };
        Ok(Self {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: FeedForward::load(weights, &format!("{}feed_forward1.", prefix))?,
            norm_self_att: norm("norm_self_att")?,
            self_attn: RelPosAttn::load(
                weights,
                &format!("{}self_attn.", prefix),
                n_heads,
                d_model,
            )?,
            norm_conv: norm("norm_conv")?,
            conv: ConformerConv::load(weights, &format!("{}conv.", prefix), d_model)?,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: FeedForward::load(weights, &format!("{}feed_forward2.", prefix))?,
            norm_out: norm("norm_out")?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Tensor {
        // FF1 (½-scaled)
        let x = x + 0.5
            * self
                .ff1
                .forward(&layer_norm(x, &self.norm_ff1.0, &self.norm_ff1.1));
        // Self-attention
        let x = &x
            + self.self_attn.forward(
                &layer_norm(&x, &self.norm_self_att.0, &self.norm_self_att.1),
                pos_emb,
            );
        // Conformer conv
        let x = &x
            + self
                .conv
                .forward(&layer_norm(&x, &self.norm_conv.0, &self.norm_conv.1));
        // FF2 (½-scaled)
        let x = &x
            + 0.5
                * self
                    .ff2
                    .forward(&layer_norm(&x, &self.norm_ff2.0, &self.norm_ff2.1));
        // Final norm
        layer_norm(&x, &self.norm_out.0, &self.norm_out.1)
    }
}

// ---------------------------------------------------------------------------
// ConformerEncoder (public)
// ---------------------------------------------------------------------------
pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    layers: Vec<ConformerLayer>,
    enc_dec_proj_w: Option<Tensor>,
    enc_dec_proj_b: Option<Tensor>,
    d_model: i64,
}

impl ConformerEncoder {
    pub fn load(weights: &Weights, cfg: &ModelConfig) -> Result<Self> {
        let enc = &cfg.encoder;
        let d_model = enc.d_model as i64;
        let n_heads = enc.n_heads as i64;

        let pre_encode = ConvSubsampling::load(weights, "encoder.pre_encode.")?;

        let mut layers = Vec::with_capacity(enc.n_layers);
        for i in 0..enc.n_layers {
            let prefix = format!("encoder.layers.{}.", i);
            let layer = ConformerLayer::load(weights, &prefix, n_heads, d_model)
                .with_context(|| format!("Loading ConformerLayer {}", i))?;
            layers.push(layer);
        }

        // Encoder→decoder projection (Linear 1280 → 1024)
        let (enc_dec_proj_w, enc_dec_proj_b) = if let (Ok(w), Ok(b)) = (
            weights.get("encoder_decoder_proj.weight"),
            weights.get("encoder_decoder_proj.bias"),
        ) {
            (Some(w.shallow_clone()), Some(b.shallow_clone()))
        } else {
            (None, None)
        };

        Ok(Self {
            pre_encode,
            layers,
            enc_dec_proj_w,
            enc_dec_proj_b,
            d_model,
        })
    }

    /// Encode mel features.
    /// x: (1, n_mels, T) → encoder_hs: (1, T', dec_hidden)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Conv subsampling
        let (x, t_prime) = self.pre_encode.forward(x);

        // Relative positional encoding: (1, 2T'-1, d_model)
        let pos_emb = rel_positional_encoding(t_prime as usize, self.d_model as usize)
            .to_kind(Kind::Float)
            .to_device(x.device());

        // Conformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb);
        }

        // Optional projection to decoder hidden size
        if let (Some(w), Some(b)) = (&self.enc_dec_proj_w, &self.enc_dec_proj_b) {
            x = linear(&x, w, b);
        }

        x
    }
}
