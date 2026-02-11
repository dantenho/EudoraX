# EudoraX Backend (Python 3.14)

A modern async backend scaffold using **FastAPI + uvicorn + uvloop** with integrations for:

- **PyArrow + Polars + Numba** data processing
- **Transformers + Tokenizers + HuggingFace Hub** tokenization/inference-ready flow
- **Qdrant + Milvus ("Malvius")** vector storage health checks
- **OpenCV + upscaling stub** image pipeline foundation
- **Torch CUDA 13.1 nightly** optional dependency configuration for UV

## Quick start (with uv)

```bash
uv venv --python 3.14
source .venv/bin/activate
uv sync
uv run python -m eudorax_backend
```

Server starts on `http://0.0.0.0:8000`.

## Torch nightly CUDA 13.1

Install optional dependencies from the configured nightly index:

```bash
uv sync --extra torch-nightly-cu131
```

## API endpoints

- `GET /health`
- `POST /tokenize`
- `POST /normalize`
- `POST /upscale`

## Example request

```bash
curl -X POST http://localhost:8000/tokenize \
  -H 'content-type: application/json' \
  -d '{"text":"hello future backend"}'
```
