from __future__ import annotations

import uvloop
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse

from .config import settings
from .services import DataService, InferenceService, UpscaleService, VectorService

uvloop.install()

app = FastAPI(title=settings.app_name, default_response_class=ORJSONResponse)

inference_service = InferenceService(model_id=settings.huggingface_model)
vector_service = VectorService(qdrant_url=settings.qdrant_url, milvus_uri=settings.milvus_uri)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "vectors": await vector_service.health(),
    }


@app.post("/tokenize")
async def tokenize(payload: dict[str, str]) -> dict[str, object]:
    text = payload.get("text", "")
    return {
        "transformers": inference_service.tokenize(text),
        "tokenizers": inference_service.tokenize_fast(text),
    }


@app.post("/normalize")
async def normalize(payload: dict[str, list[float]]) -> dict[str, object]:
    values = payload.get("values", [])
    return DataService.normalize(values)


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = 2) -> dict[str, int]:
    content = await file.read()
    return UpscaleService.upscale_stub(content, scale=scale)
