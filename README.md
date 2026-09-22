# laya-mlx-rs

A native Rust + MLX implementation of the [Laya](https://huggingface.co/convaiinnovations/laya)
non-autoregressive decision model for Apple Silicon.

It runs the ModernBERT encoder and typed decision head directly on MLX/Metal, with no Python,
PyTorch, or Transformers runtime dependency. It supports the English, multilingual, and
typed-decisions checkpoints, automatic language routing, a CLI, and a localhost HTTP service.

## Accuracy

Compared with the upstream torch FP32 implementation:

| Mode | Worst probability difference | Changed decisions |
|---|---:|---:|
| FP32 | 0.0 | 0 |
| FP16 | 0.0003 | 0 |

Two compatibility details are intentional: JSON object order is preserved because choice order is
part of the model input, and state JSON uses Python-compatible separators because the model reads
serialized state as text.

## Performance

Measured on an M4 Pro with 48 GB, three questions at sequence length 96:

| Stage | Python MLX | Rust MLX |
|---|---:|---:|
| English encoder, FP16 | 48.5 ms | 42.5 ms |
| typed-decisions encoder, FP16 | 48.4 ms | 42.9 ms |
| typed-decisions full forward, FP16 | 52.0 ms | 47.7 ms |

The main optimization is a dtype-aware, shapeless-compiled GELU. Generic mlx-rs GELU constants
are F32; in an F16 graph that inserts two casts per layer. Reusing one compiled GELU with constants
created in the input dtype reduces the matching 28-layer microbenchmark from 9.34 ms to 7.49–7.55
ms, equal to Python MLX.

## Requirements

- Apple Silicon Mac
- Rust toolchain
- CMake (required while building MLX)

## Build and test

```bash
cargo build --release
cargo test
```

## CLI

The first run downloads the selected files from `convaiinnovations/laya` into
`$XDG_CACHE_HOME/laya-rs` or `~/.cache/laya-rs`.

```bash
# Start the local service
LAYA_DTYPE=float16 cargo run --release -- --serve

# Run a JSON batch
cargo run --release -- --batch cases.json typed-decisions float16

# Benchmark the first case in a batch
cargo run --release -- --bench cases.json 20 typed-decisions float16
```

Batch input:

```json
{
  "cases": [
    {
      "name": "refund",
      "state": {"message": "I was charged twice. Please refund the duplicate."},
      "questions": {
        "refund": {
          "type": "noul",
          "instructions": "Does the customer ask for money back?"
        }
      }
    }
  ]
}
```

## HTTP service

The service binds to `127.0.0.1:8765` by default and keeps all configured checkpoints resident.

| Variable | Default |
|---|---|
| `LAYA_HOST` | `127.0.0.1` |
| `LAYA_PORT` | `8765` |
| `LAYA_MODELS` | `english,multilingual,typed-decisions` |
| `LAYA_DTYPE` | `float16` |
| `LAYA_MODEL` | `convaiinnovations/laya` |
| `LAYA_TOKEN` | unset; optional bearer token |
| `HF_TOKEN` | unset; only needed for gated repositories |
| `LAYA_EVAL_EVERY` | `0`; evaluate every N encoder layers to trade speed for lower peak memory |

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/predict \
  -H 'content-type: application/json' \
  -d '{"state":{"message":"Please refund the duplicate charge."},"questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}}'
```

## Architecture

- `model.rs`: ModernBERT encoder and decision head
- `agent.rs`: checkpoint loading, batching, calibration, and result schema
- `router.rs`: checkpoint selection
- `lang.rs`: script and language detection
- `text.rs`: input rendering and probability post-processing
- `server.rs`: axum service with one dedicated inference thread

## Acknowledgements

This implementation targets the Apache-2.0-licensed
[Convai Innovations Laya](https://github.com/NandhaKishorM/laya) checkpoints and behavior. It uses
[mlx-rs](https://github.com/oxideai/mlx-rs) and Apple MLX.

## License

Apache-2.0
