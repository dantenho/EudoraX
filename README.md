# EudoraX

EudoraX now includes a Python backend module designed for Python **3.14 no-GIL**
runtimes with modern AI/CV acceleration paths.

## Included backend functions

`backend/functions.py` adds production-friendly entry points with optional,
lazy-loaded dependencies:

- Async loop tuning with **uvloop + asyncio**
- CPU capability probing for **AVX512 / Zen5 class flags**
- DataFrame conversion with **Polars + PyArrow**
- Text tokenization via **Tokenizers** or **Transformers**
- LLM generation via **vLLM** with Transformers fallback
- CUDA vector math via **Numba + CUDA 13.1 compatible drivers**
- Computer-vision tools:
  - Denoising (OpenCV; NVIDIA denoiser pre-processing path)
  - Foreground segmentation (GrabCut baseline)
  - Depth estimation via Transformers depth pipeline
  - Super-resolution with **Real-ESRGAN**

## Dependency strategy

The backend is capability-driven: it starts even if some optional packages are
missing, and raises targeted runtime errors when you call a function that needs
an unavailable library.

Core dependencies are declared in `pyproject.toml`, with GPU-heavy libraries in
an optional `gpu` extra.

## Quick start

```bash
python -c "from backend.functions import get_backend_capabilities; print(get_backend_capabilities())"
```

Install with GPU extras:

```bash
pip install -e .[gpu]
```
