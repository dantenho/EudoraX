# EudoraX

## Python 3.14 cutting-edge backend scaffold

This repo now includes a modern async backend built with:

- **FastAPI + Uvicorn + asyncio + uvloop**
- **PyArrow + Polars** for high-performance analytics data frames
- **Transformers + Tokenizers** (Hugging Face stack)
- **Numba** support for JIT acceleration
- **Torch nightly CUDA 13.1** index configured via `uv`
- **Qdrant + Milvus clients** for vector search connectivity
- **OpenCV + Real-ESRGAN** for CV/upscaler workflows

## Quick start

```bash
uv sync
uv run uvicorn eudorax_backend.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` basic liveness
- `GET /stack` runtime import and CUDA capability report
- `GET /analytics/demo` Arrow → Polars demo analytics

## Project layout

- `pyproject.toml` dependency and build configuration
- `src/eudorax_backend/main.py` FastAPI application
