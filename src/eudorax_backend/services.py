from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
import pyarrow as pa
from fastapi import HTTPException
from numba import njit
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pymilvus import MilvusClient
from tokenizers import Tokenizer
from transformers import AutoTokenizer


@njit(cache=True)
def _l2_norm(arr: np.ndarray) -> float:
    return float(np.sqrt(np.sum(arr * arr)))


@dataclass
class InferenceService:
    model_id: str

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.fast_tokenizer = Tokenizer.from_pretrained(self.model_id)

    def tokenize(self, text: str) -> dict[str, list[int]]:
        encoded = self.tokenizer(text, truncation=True, max_length=512)
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def tokenize_fast(self, text: str) -> dict[str, list[int]]:
        encoded = self.fast_tokenizer.encode(text)
        return {"ids": encoded.ids, "tokens": encoded.tokens}


class VectorService:
    def __init__(self, qdrant_url: str, milvus_uri: str) -> None:
        self.qdrant = AsyncQdrantClient(url=qdrant_url)
        self.milvus = MilvusClient(uri=milvus_uri)

    async def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        await self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    async def health(self) -> dict[str, str]:
        qdrant_ok = "ok"
        milvus_ok = "ok"
        try:
            await self.qdrant.get_collections()
        except Exception as exc:  # noqa: BLE001
            qdrant_ok = f"error: {exc!s}"

        try:
            await asyncio.to_thread(self.milvus.list_collections)
        except Exception as exc:  # noqa: BLE001
            milvus_ok = f"error: {exc!s}"

        return {"qdrant": qdrant_ok, "malvius_milvus": milvus_ok}


class DataService:
    @staticmethod
    def normalize(values: list[float]) -> dict[str, object]:
        if not values:
            raise HTTPException(status_code=400, detail="values cannot be empty")

        series = pl.Series(name="values", values=values)
        norm = _l2_norm(series.to_numpy())

        table = pa.table({"values": values})
        return {
            "l2_norm": norm,
            "rows": table.num_rows,
            "columns": table.num_columns,
            "mean": float(series.mean()),
        }


class UpscaleService:
    @staticmethod
    def upscale_stub(image_bytes: bytes, scale: int = 2) -> dict[str, int]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        h, w, _ = img.shape
        resized = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        return {"original_width": w, "original_height": h, "upscaled_width": resized.shape[1], "upscaled_height": resized.shape[0]}
