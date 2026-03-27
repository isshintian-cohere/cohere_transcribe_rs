/// Greedy decoding loop for CohereTranscribe ASR.
use anyhow::Result;
use tch::Tensor;

use crate::decoder::TransformerDecoder;
use crate::encoder::ConformerEncoder;
use crate::tokenizer::Tokenizer;

/// Run a full transcription: encode → greedy decode → detokenize.
pub fn transcribe(
    mel: &Tensor,
    encoder: &ConformerEncoder,
    decoder: &TransformerDecoder,
    tokenizer: &Tokenizer,
    language: &str,
    punctuation: bool,
    max_new_tokens: usize,
) -> Result<String> {
    // 1. Encode
    tracing::debug!("Running encoder...");
    let encoder_hs = encoder.forward(mel); // (1, T', dec_hidden)
    tracing::debug!("Encoder output shape: {:?}", encoder_hs.size());

    // 2. Pre-compute cross-attention K/V (encoder side) — reused every decoder step
    let cross_kv = decoder.precompute_cross_kv(&encoder_hs);

    // 3. Build prompt token IDs
    let prompt = tokenizer.special.build_prompt(language, punctuation)?;
    let n_prompt = prompt.len();
    tracing::debug!("Prompt IDs: {:?}", prompt);

    // 4. Initialize self-attention KV cache (empty at start)
    let mut self_kv_cache: Vec<(Option<Tensor>, Option<Tensor>)> =
        (0..decoder.layers.len()).map(|_| (None, None)).collect();

    // 5. Prime the decoder by feeding all prompt tokens, updating KV cache
    //    We run each prompt token through the decoder to build the cache.
    let mut last_logits: Vec<f32> = Vec::new();

    for (i, &token_id) in prompt.iter().enumerate() {
        let position = i as i64;
        let (logits, new_kv) = decoder.step(token_id, position, &self_kv_cache, &cross_kv);
        self_kv_cache = new_kv;
        last_logits = logits;
    }

    // 6. Greedy decode until EOS or max_new_tokens
    let eos_id = tokenizer.special.eos;
    let nospeech_id = tokenizer.special.nospeech;
    let mut generated: Vec<i64> = Vec::new();

    let mut next_token = argmax(&last_logits);
    let mut position = n_prompt as i64;

    while generated.len() < max_new_tokens {
        if next_token == eos_id || next_token == nospeech_id {
            break;
        }
        generated.push(next_token);

        let (logits, new_kv) = decoder.step(next_token, position, &self_kv_cache, &cross_kv);
        self_kv_cache = new_kv;
        last_logits = logits;
        position += 1;
        next_token = argmax(&last_logits);
    }

    tracing::debug!("Generated token IDs: {:?}", generated);

    // 7. Decode tokens to text
    let text = tokenizer.decode(&generated);
    Ok(text)
}

fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}
