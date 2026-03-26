# Test fixtures

Audio files borrowed from [second-state/qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs/tree/main/test_audio).

| File | Duration | Reference transcript |
|------|----------|---------------------|
| `sample1.wav` | ~11 s | Thank you for your contribution to the most recent issue of Computer! We sincerely appreciate your work, and we hope you enjoy the entire issue. |
| `sample2.wav` | ~6 s | The quick brown fox jumps over the lazy dog. |

Both files are 16 kHz mono WAV — the native format for the Cohere Transcribe model.

## Usage

```bash
# CLI
./target/release/transcribe \
  --model-dir models/cohere-transcribe-03-2026 \
  tests/fixtures/sample2.wav

# API server (server must be running)
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@tests/fixtures/sample2.wav;type=audio/wav" \
  -F "model=cohere-transcribe"
```
