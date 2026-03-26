/// Transformer decoder implementing the architecture from modeling_cohere_asr.py.
///
/// Architecture:
///   TransformerDecoderEmbedding  → token embed + fixed pos enc + layer norm
///   TransformerDecoderLayer × 8  → pre-LN self-attn + cross-attn + FFN
///   FinalLayerNorm
///   TokenClassifierHead  → Linear(1024, 16384)
///
/// Greedy decoding uses a simple KV cache:
///   - Self-attention: cache grows by 1 per step
///   - Cross-attention: K/V from encoder are fixed for the whole utterance
use anyhow::Result;
use tch::{Kind, Tensor};

use crate::config::ModelConfig;
use crate::weights::Weights;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.matmul(&w.tr()) + b
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.layer_norm(&[w.size()[0]], Some(w), Some(b), 1e-5, false)
}

// Scaled dot-product attention (no KV cache involvement here; caller manages cache).
// q: (B, H, Tq, d_k), k: (B, H, Tk, d_k), v: (B, H, Tk, d_k)
// mask: optional additive mask (B, 1, Tq, Tk) or (1, 1, Tq, Tk)
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
    let scale = 1.0 / (q.size().last().copied().unwrap() as f64).sqrt();
    let scores = q.matmul(&k.transpose(-2, -1)) * scale;
    let scores = match mask {
        Some(m) => scores + m,
        None => scores,
    };
    scores.softmax(-1, Kind::Float).matmul(v)
}

// ---------------------------------------------------------------------------
// Decoder attention (used for both self-attn and cross-attn)
// ---------------------------------------------------------------------------
struct DecoderAttn {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    n_heads: i64,
    head_dim: i64,
    hidden: i64,
}

impl DecoderAttn {
    fn load(weights: &Weights, prefix: &str, n_heads: i64, hidden: i64) -> Result<Self> {
        let head_dim = hidden / n_heads;
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            q_w: w("query_net.weight")?,
            q_b: w("query_net.bias")?,
            k_w: w("key_net.weight")?,
            k_b: w("key_net.bias")?,
            v_w: w("value_net.weight")?,
            v_b: w("value_net.bias")?,
            out_w: w("out_projection.weight")?,
            out_b: w("out_projection.bias")?,
            n_heads,
            head_dim,
            hidden,
        })
    }

    fn project_qkv(&self, hidden_states: &Tensor, source: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (b, t, _) = hidden_states.size3().unwrap();
        let s = source.size()[1];

        let reshape_q = |z: &Tensor| -> Tensor {
            z.view([b, t, self.n_heads, self.head_dim]).transpose(1, 2)
        };
        let reshape_kv = |z: &Tensor, seq: i64| -> Tensor {
            z.view([b, seq, self.n_heads, self.head_dim]).transpose(1, 2)
        };

        let q = reshape_q(&linear(hidden_states, &self.q_w, &self.q_b));
        let k = reshape_kv(&linear(source, &self.k_w, &self.k_b), s);
        let v = reshape_kv(&linear(source, &self.v_w, &self.v_b), s);
        (q, k, v)
    }

    fn forward_with_kv(
        &self,
        hidden_states: &Tensor,
        source: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let (b, t, _) = hidden_states.size3().unwrap();
        let (q, k, v) = self.project_qkv(hidden_states, source);
        let out = sdpa(&q, &k, &v, mask);
        let out = out
            .transpose(1, 2)
            .contiguous()
            .view([b, t, self.hidden]);
        linear(&out, &self.out_w, &self.out_b)
    }
}

// ---------------------------------------------------------------------------
// Decoder FFN (relu activation)
// ---------------------------------------------------------------------------
struct DecoderFFN {
    dense_in_w: Tensor,
    dense_in_b: Tensor,
    dense_out_w: Tensor,
    dense_out_b: Tensor,
}

impl DecoderFFN {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        let w = |n: &str| -> Result<Tensor> {
            Ok(weights.get(&format!("{}{}", prefix, n))?.shallow_clone())
        };
        Ok(Self {
            dense_in_w: w("dense_in.weight")?,
            dense_in_b: w("dense_in.bias")?,
            dense_out_w: w("dense_out.weight")?,
            dense_out_b: w("dense_out.bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        linear(&linear(x, &self.dense_in_w, &self.dense_in_b).relu(), &self.dense_out_w, &self.dense_out_b)
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoderLayer
// ---------------------------------------------------------------------------
pub struct DecoderLayer {
    norm1: (Tensor, Tensor), // self-attn pre-norm
    self_attn: DecoderAttn,
    norm2: (Tensor, Tensor), // cross-attn pre-norm
    cross_attn: DecoderAttn,
    norm3: (Tensor, Tensor), // FFN pre-norm
    ffn: DecoderFFN,
}

impl DecoderLayer {
    fn load(weights: &Weights, prefix: &str, n_heads: i64, hidden: i64) -> Result<Self> {
        let norm = |n: &str| -> Result<(Tensor, Tensor)> {
            let key = format!("{}{}", prefix, n);
            Ok((
                weights.get(&format!("{}.weight", key))?.shallow_clone(),
                weights.get(&format!("{}.bias", key))?.shallow_clone(),
            ))
        };
        Ok(Self {
            norm1: norm("layer_norm_1")?,
            self_attn: DecoderAttn::load(
                weights,
                &format!("{}first_sub_layer.", prefix),
                n_heads,
                hidden,
            )?,
            norm2: norm("layer_norm_2")?,
            cross_attn: DecoderAttn::load(
                weights,
                &format!("{}second_sub_layer.", prefix),
                n_heads,
                hidden,
            )?,
            norm3: norm("layer_norm_3")?,
            ffn: DecoderFFN::load(weights, &format!("{}third_sub_layer.", prefix))?,
        })
    }

    /// Single-token forward (during greedy decoding with KV cache).
    ///
    /// hidden: (1, 1, hidden)  — just the new token's embedding
    /// self_k_cache / self_v_cache: accumulated K/V from previous steps (1, H, T_prev, d_k)
    /// cross_k / cross_v: encoder K/V (1, H, T_enc, d_k) — pre-computed once
    ///
    /// Returns (out, new_self_k, new_self_v).
    fn forward_cached(
        &self,
        hidden: &Tensor,
        self_k_cache: Option<&Tensor>,
        self_v_cache: Option<&Tensor>,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_attn_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor, Tensor) {
        // Self-attention with KV cache
        let normed = layer_norm(hidden, &self.norm1.0, &self.norm1.1);
        let (b, t, _) = normed.size3().unwrap();
        let (q_new, k_new, v_new) = self.self_attn.project_qkv(&normed, &normed);

        let (k_full, v_full) = match (self_k_cache, self_v_cache) {
            (Some(kc), Some(vc)) => (
                Tensor::cat(&[kc, &k_new], 2),
                Tensor::cat(&[vc, &v_new], 2),
            ),
            _ => (k_new.shallow_clone(), v_new.shallow_clone()),
        };

        let self_out = sdpa(&q_new, &k_full, &v_full, self_attn_mask);
        let self_out = self_out
            .transpose(1, 2)
            .contiguous()
            .view([b, t, self.self_attn.hidden]);
        let self_out = linear(&self_out, &self.self_attn.out_w, &self.self_attn.out_b);
        let hidden = hidden + self_out;

        // Cross-attention: Q from decoder, K/V from encoder (pre-computed)
        let normed2 = layer_norm(&hidden, &self.norm2.0, &self.norm2.1);
        let cross_q = linear(&normed2, &self.cross_attn.q_w, &self.cross_attn.q_b)
            .view([b, t, self.cross_attn.n_heads, self.cross_attn.head_dim])
            .transpose(1, 2);
        let cross_out = sdpa(&cross_q, cross_k, cross_v, None);
        let cross_out = cross_out
            .transpose(1, 2)
            .contiguous()
            .view([b, t, self.cross_attn.hidden]);
        let cross_out = linear(&cross_out, &self.cross_attn.out_w, &self.cross_attn.out_b);
        let hidden = hidden + cross_out;

        // FFN
        let normed3 = layer_norm(&hidden, &self.norm3.0, &self.norm3.1);
        let ffn_out = self.ffn.forward(&normed3);
        let hidden = hidden + ffn_out;

        (hidden, k_full, v_full)
    }
}

// ---------------------------------------------------------------------------
// Fixed Positional Encoding (stored in weights)
// pos_enc: (max_seq_len, d_model), divided by sqrt(d_model) during model init
// ---------------------------------------------------------------------------
struct FixedPosEnc {
    pos_enc: Tensor, // (max_seq_len, d_model)
}

impl FixedPosEnc {
    fn load(weights: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            pos_enc: weights
                .get(&format!("{}position_embedding.pos_enc", prefix))?
                .shallow_clone(),
        })
    }

    fn forward(&self, position_ids: &[i64]) -> Tensor {
        let indices = Tensor::from_slice(position_ids);
        self.pos_enc.index_select(0, &indices)
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoder (public)
// ---------------------------------------------------------------------------
pub struct TransformerDecoder {
    token_emb: Tensor,           // (vocab, hidden)
    pos_enc: FixedPosEnc,
    emb_norm_w: Tensor,
    emb_norm_b: Tensor,
    pub layers: Vec<DecoderLayer>,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
    head_w: Tensor,
    head_b: Tensor,
    n_heads: i64,
    head_dim: i64,
    hidden: i64,
}

impl TransformerDecoder {
    pub fn load(weights: &Weights, cfg: &ModelConfig) -> Result<Self> {
        let dec = &cfg.transf_decoder.config_dict;
        let hidden = dec.hidden_size as i64;
        let n_heads = dec.num_attention_heads as i64;
        let head_dim = hidden / n_heads;

        let emb_prefix = "transf_decoder._embedding.";
        let dec_prefix = "transf_decoder._decoder.";

        let token_emb = weights
            .get(&format!("{}token_embedding.weight", emb_prefix))?
            .shallow_clone();
        let pos_enc = FixedPosEnc::load(weights, emb_prefix)?;
        let emb_norm_w = weights
            .get(&format!("{}layer_norm.weight", emb_prefix))?
            .shallow_clone();
        let emb_norm_b = weights
            .get(&format!("{}layer_norm.bias", emb_prefix))?
            .shallow_clone();

        let mut layers = Vec::with_capacity(dec.num_layers);
        for i in 0..dec.num_layers {
            let prefix = format!("{}layers.{}.", dec_prefix, i);
            layers.push(DecoderLayer::load(weights, &prefix, n_heads, hidden)?);
        }

        let final_ln_w = weights
            .get(&format!("{}final_layer_norm.weight", dec_prefix))?
            .shallow_clone();
        let final_ln_b = weights
            .get(&format!("{}final_layer_norm.bias", dec_prefix))?
            .shallow_clone();

        // Head weights (tied with embedding)
        let head_w = weights
            .get("log_softmax.mlp.layer0.weight")?
            .shallow_clone();
        let head_b = weights
            .get("log_softmax.mlp.layer0.bias")?
            .shallow_clone();

        Ok(Self {
            token_emb,
            pos_enc,
            emb_norm_w,
            emb_norm_b,
            layers,
            final_ln_w,
            final_ln_b,
            head_w,
            head_b,
            n_heads,
            head_dim,
            hidden,
        })
    }

    /// Pre-compute the cross-attention K and V for each decoder layer from encoder hidden states.
    /// Returns Vec of (K, V) pairs, one per layer.
    pub fn precompute_cross_kv(&self, encoder_hs: &Tensor) -> Vec<(Tensor, Tensor)> {
        let (b, s, _) = encoder_hs.size3().unwrap();
        self.layers
            .iter()
            .map(|layer| {
                let k = linear(encoder_hs, &layer.cross_attn.k_w, &layer.cross_attn.k_b)
                    .view([b, s, self.n_heads, self.head_dim])
                    .transpose(1, 2);
                let v = linear(encoder_hs, &layer.cross_attn.v_w, &layer.cross_attn.v_b)
                    .view([b, s, self.n_heads, self.head_dim])
                    .transpose(1, 2);
                (k, v)
            })
            .collect()
    }

    /// One greedy-decoding step.
    ///
    /// token_id: the latest token (i64)
    /// position: the position index of this token
    /// self_kv_cache: per-layer (K, V) for previous self-attention steps
    /// cross_kv: per-layer (K, V) from encoder (pre-computed)
    ///
    /// Returns (logits: Vec<f32> of shape vocab_size, updated self_kv_cache).
    pub fn step(
        &self,
        token_id: i64,
        position: i64,
        self_kv_cache: &[(Option<Tensor>, Option<Tensor>)],
        cross_kv: &[(Tensor, Tensor)],
    ) -> (Vec<f32>, Vec<(Option<Tensor>, Option<Tensor>)>) {
        let ids = Tensor::from_slice(&[token_id]);
        let emb = self.token_emb.index_select(0, &ids).unsqueeze(0); // (1, 1, hidden)
        let pe = self.pos_enc.forward(&[position]).unsqueeze(0);      // (1, 1, hidden)
        let x = layer_norm(&(emb + pe), &self.emb_norm_w, &self.emb_norm_b);

        let mut new_kv: Vec<(Option<Tensor>, Option<Tensor>)> = Vec::with_capacity(self.layers.len());
        let mut hidden = x;

        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = &self_kv_cache[i];
            let (ck, cv) = &cross_kv[i];

            let (new_hidden, k_full, v_full) = layer.forward_cached(
                &hidden,
                kc.as_ref(),
                vc.as_ref(),
                ck,
                cv,
                None, // No causal mask needed for single-token input
            );

            hidden = new_hidden;
            new_kv.push((Some(k_full), Some(v_full)));
        }

        // Final layer norm + head
        let hidden = layer_norm(&hidden, &self.final_ln_w, &self.final_ln_b);
        let logits = linear(&hidden.squeeze_dim(1), &self.head_w, &self.head_b); // (1, vocab)
        let logits_squeezed = logits.squeeze();
        let logits_vec: Vec<f32> = Vec::try_from(logits_squeezed).expect("logits to Vec<f32>");

        (logits_vec, new_kv)
    }
}
