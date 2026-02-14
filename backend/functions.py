"""High-performance backend helpers for AI + computer vision workloads.

The module targets Python 3.14 no-GIL deployments and tries to use modern
accelerators when available (CUDA, Numba, vLLM, Torch nightly, etc.).
All third-party imports are optional and resolved lazily.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BackendCapabilities:
    """Runtime capability snapshot for optional high-performance dependencies."""

    python: str
    uvloop: bool
    numba: bool
    pyarrow: bool
    polars: bool
    torch: bool
    vllm: bool
    transformers: bool
    tokenizers: bool
    diffusers: bool
    cv2: bool
    realesrgan: bool
    cuda_available: bool
    avx512_available: bool


def _optional_import(module_name: str) -> Any | None:
    """Import a module if installed, otherwise return ``None``."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def cpu_flags() -> set[str]:
    """Return lowercase CPU flags detected from /proc/cpuinfo when available."""
    flags: set[str] = set()
    cpuinfo_path = "/proc/cpuinfo"
    if not os.path.exists(cpuinfo_path):
        return flags

    with open(cpuinfo_path, "r", encoding="utf-8") as cpuinfo:
        for line in cpuinfo:
            if line.lower().startswith("flags"):
                _, value = line.split(":", 1)
                flags.update(flag.strip().lower() for flag in value.split())
    return flags


def configure_event_loop() -> str:
    """Set uvloop as default loop policy when available and return active policy."""
    uvloop = _optional_import("uvloop")
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        return "uvloop"
    return "asyncio-default"


def get_backend_capabilities() -> BackendCapabilities:
    """Inspect which optional dependencies are available at runtime."""
    torch = _optional_import("torch")
    numba = _optional_import("numba")
    flags = cpu_flags()

    cuda_available = False
    if torch is not None and hasattr(torch, "cuda"):
        cuda_available = bool(torch.cuda.is_available())
    elif numba is not None and hasattr(numba, "cuda"):
        cuda_available = bool(numba.cuda.is_available())

    return BackendCapabilities(
        python=f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        uvloop=_optional_import("uvloop") is not None,
        numba=numba is not None,
        pyarrow=_optional_import("pyarrow") is not None,
        polars=_optional_import("polars") is not None,
        torch=torch is not None,
        vllm=_optional_import("vllm") is not None,
        transformers=_optional_import("transformers") is not None,
        tokenizers=_optional_import("tokenizers") is not None,
        diffusers=_optional_import("diffusers") is not None,
        cv2=_optional_import("cv2") is not None,
        realesrgan=_optional_import("realesrgan") is not None,
        cuda_available=cuda_available,
        avx512_available=any(flag.startswith("avx512") for flag in flags),
    )


def polars_to_arrow(dataframe: Any) -> Any:
    """Convert a Polars DataFrame into a PyArrow Table."""
    pl = _optional_import("polars")
    pa = _optional_import("pyarrow")
    if pl is None or pa is None:
        raise RuntimeError("polars_to_arrow requires both polars and pyarrow")
    if not isinstance(dataframe, pl.DataFrame):
        raise TypeError("dataframe must be a polars.DataFrame")
    return dataframe.to_arrow()


def tokenize_text(text: str, model_name: str = "bert-base-uncased") -> list[int]:
    """Tokenize text using Hugging Face tokenizers fast path, then transformers fallback."""
    tokenizers = _optional_import("tokenizers")
    if tokenizers is not None:
        tokenizer = tokenizers.Tokenizer.from_pretrained(model_name)
        return tokenizer.encode(text).ids

    transformers = _optional_import("transformers")
    if transformers is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return tokenizer(text, add_special_tokens=True)["input_ids"]

    raise RuntimeError("tokenizers or transformers must be installed")


def llm_generate(prompt: str, model_name: str, max_tokens: int = 128) -> str:
    """Generate text with vLLM, with transformers pipeline as fallback."""
    vllm = _optional_import("vllm")
    if vllm is not None:
        sampling = vllm.SamplingParams(max_tokens=max_tokens, temperature=0.2)
        engine = vllm.LLM(model=model_name)
        outputs = engine.generate([prompt], sampling_params=sampling)
        return outputs[0].outputs[0].text

    transformers = _optional_import("transformers")
    if transformers is not None:
        generator = transformers.pipeline("text-generation", model=model_name)
        output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.2)
        return output[0]["generated_text"]

    raise RuntimeError("vllm or transformers must be installed")


def cuda_vector_add(a: Any, b: Any) -> Any:
    """Add two vectors on CUDA using Numba (CUDA 13.1 compatible driver expected)."""
    np = _optional_import("numpy")
    numba = _optional_import("numba")
    if np is None or numba is None:
        raise RuntimeError("cuda_vector_add requires numpy and numba")

    if not hasattr(numba, "cuda") or not numba.cuda.is_available():
        raise RuntimeError("CUDA is not available for Numba")

    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        raise ValueError("input vectors must share the same shape")

    @numba.cuda.jit
    def _kernel(x, y, out):
        idx = numba.cuda.grid(1)
        if idx < out.size:
            out[idx] = x[idx] + y[idx]

    out = np.empty_like(arr_a)
    threads_per_block = 256
    blocks = (out.size + threads_per_block - 1) // threads_per_block
    _kernel[blocks, threads_per_block](arr_a, arr_b, out)
    numba.cuda.synchronize()
    return out


def denoise_image(image: Any) -> Any:
    """Run CV denoising using OpenCV; intended for NVIDIA denoiser pre-processing."""
    cv2 = _optional_import("cv2")
    if cv2 is None:
        raise RuntimeError("denoise_image requires opencv-python")
    return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)


def segment_foreground(image: Any) -> Any:
    """Segment foreground mask using OpenCV GrabCut baseline."""
    np = _optional_import("numpy")
    cv2 = _optional_import("cv2")
    if np is None or cv2 is None:
        raise RuntimeError("segment_foreground requires numpy and opencv-python")

    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")


def estimate_depth(image: Any, model_name: str = "Intel/dpt-large") -> Any:
    """Estimate depth map using a transformers depth-estimation pipeline."""
    transformers = _optional_import("transformers")
    if transformers is None:
        raise RuntimeError("estimate_depth requires transformers")
    estimator = transformers.pipeline("depth-estimation", model=model_name)
    return estimator(image)


def upscale_image(image_path: str, model_path: str) -> Any:
    """Upscale image with Real-ESRGAN when available."""
    realesrgan = _optional_import("realesrgan")
    if realesrgan is None:
        raise RuntimeError("upscale_image requires realesrgan")

    cv2 = _optional_import("cv2")
    if cv2 is None:
        raise RuntimeError("upscale_image requires opencv-python")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    upsampler = realesrgan.RealESRGANer(
        scale=4,
        model_path=model_path,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    output, _ = upsampler.enhance(image)
    return output
