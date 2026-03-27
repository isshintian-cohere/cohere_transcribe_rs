//! MLX greedy decoding loop — mirrors src/inference.rs using Array.

use anyhow::Result;

use super::array::Array;
use super::decoder::TransformerDecoder;
use super::encoder::ConformerEncoder;
use crate::tokenizer::Tokenizer;

/// Run a full transcription: encode → greedy decode → detokenize.
pub fn transcribe(
    mel: &Array,
    encoder: &ConformerEncoder,
    decoder: &TransformerDecoder,
    tokenizer: &Tokenizer,
    language: &str,
    punctuation: bool,
    max_new_tokens: usize,
) -> Result<String> {
    // 1. Encode
    tracing::debug!("Running MLX encoder...");
    let encoder_hs = encoder.forward(mel); // (1, T', dec_hidden)
    tracing::debug!("Encoder output shape: {:?}", encoder_hs.shape());

    // Force evaluation before cross-attention pre-computation
    encoder_hs.eval();

    // 2. Pre-compute cross-attention K/V — reused every decoder step
    let cross_kv = decoder.precompute_cross_kv(&encoder_hs);

    // 3. Build prompt token IDs
    let prompt = tokenizer.special.build_prompt(language, punctuation)?;
    let n_prompt = prompt.len();
    tracing::debug!("Prompt IDs: {:?}", prompt);

    // 4. Initialize self-attention KV cache (empty)
    let mut self_kv_cache: Vec<(Option<Array>, Option<Array>)> =
        (0..decoder.layers.len()).map(|_| (None, None)).collect();

    // 5. Prime decoder with prompt tokens
    let mut next_token = 0i32;
    for (i, &token_id) in prompt.iter().enumerate() {
        let (token, new_kv) = decoder.step(token_id as i32, i as i32, &self_kv_cache, &cross_kv);
        self_kv_cache = new_kv;
        next_token = token;
    }

    // 6. Greedy decode until EOS or max_new_tokens
    //    Argmax is computed on GPU inside decoder.step() — only a single i32
    //    is transferred per step instead of the full 16,384-element logits vector.
    let eos_id = tokenizer.special.eos as i32;
    let nospeech_id = tokenizer.special.nospeech as i32;
    let mut generated: Vec<i64> = Vec::new();
    let mut position = n_prompt as i32;

    while generated.len() < max_new_tokens {
        if next_token == eos_id || next_token == nospeech_id {
            break;
        }
        generated.push(next_token as i64);

        let (token, new_kv) = decoder.step(next_token, position, &self_kv_cache, &cross_kv);
        self_kv_cache = new_kv;
        next_token = token;
        position += 1;
    }

    tracing::debug!("Generated token IDs: {:?}", generated);

    // 7. Decode tokens to text
    Ok(tokenizer.decode(&generated))
}
