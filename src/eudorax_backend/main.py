from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.util import find_spec
from typing import Any

import pyarrow as pa
import polars as pl
from fastapi import FastAPI


def _try_enable_uvloop() -> bool:
    if find_spec("uvloop") is None:
        return False

    import uvloop

    uvloop.install()
    return True


def _safe_import_status(module_name: str) -> dict[str, Any]:
    available = find_spec(module_name) is not None
    return {"module": module_name, "available": available}


def _torch_status() -> dict[str, Any]:
    if find_spec("torch") is None:
        return {"available": False}

    import torch

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    uvloop_enabled = _try_enable_uvloop()
    print(f"uvloop enabled: {uvloop_enabled}")
    yield


app = FastAPI(
    title="EudoraX Backend",
    version="0.1.0",
    lifespan=lifespan,
    summary="FastAPI + asyncio backend with Arrow/Polars/HF/GPU/vector-db ready integrations.",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/stack")
async def stack() -> dict[str, Any]:
    modules = [
        "pyarrow",
        "polars",
        "numba",
        "tokenizers",
        "transformers",
        "qdrant_client",
        "pymilvus",
        "cv2",
        "realesrgan",
        "uvicorn",
    ]
    return {
        "python_target": "3.14",
        "framework": "fastapi",
        "event_loop": "uvloop",
        "modules": [_safe_import_status(module_name) for module_name in modules],
        "torch": _torch_status(),
    }


@app.get("/analytics/demo")
async def analytics_demo() -> dict[str, Any]:
    table = pa.table({"id": [1, 2, 3], "latency_ms": [10.5, 20.2, 8.9]})
    frame = pl.from_arrow(table).with_columns(
        (pl.col("latency_ms") * 1_000).alias("latency_us")
    )
    return {
        "rows": frame.to_dicts(),
        "mean_latency_ms": frame["latency_ms"].mean(),
        "max_latency_us": frame["latency_us"].max(),
    }
