
"""
@file backend/models.py
@description Pydantic V2 models for EudoraX protocol.
@jules_hint Align these strictly with types.ts on the frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PIXEL = "pixel"

class SynthesisRequest(BaseModel):
    prompt: str = Field(..., description="The neural command for synthesis")
    modality: Modality
    aspect_ratio: str = "1:1"
    negative_prompt: Optional[str] = None
    style_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class SynthesisResponse(BaseModel):
    asset_id: str
    url: str
    modality: Modality
    metadata: Dict[str, Any]
    latency_ms: float
    compute_node: str = "NODE_0x01"
