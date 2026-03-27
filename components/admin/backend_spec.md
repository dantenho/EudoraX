
# EudoraX Extreme Backend Spec v4.7 (2026 Structural Synthesis)

## Structural & Conditional Synthesis
- **ControlNet**: Mandatory support for multi-adapter conditioning (Canny, Depth, OpenPose).
- **OpenPose**: Extraction of 18-keypoint skeleton vectors for human-centric motion and pose control.
- **MiDaS Depth**: High-resolution monocular depth estimation for geometric structural fidelity.

## Post-Processing & Upscaling
- **Real-ESRGAN**: Optimized GAN-based upscaling with support for 4x and 8x tiles.
- **OptiX Denoiser**: Integration with NVIDIA AI denoise kernels for artifact-free tensor outputs.

## Performance flags (v4.7 Updates)
- `-Dcontrolnet_parallel=1`: Enable concurrent multi-adapter loading.
- `-Dgan_fp16=1`: Half-precision GAN execution.
- `-mavx512vnni`: Vector neural network instructions for quantized depth passes.

## Middleware Runtime: Node.js 25 (Alpha)
The frontend communicates via a Node.js 25 BFF (Backend for Frontend) layer to leverage specific 2026 capabilities:
- **`node:ai` Integration**: Direct binding to server-side NPU tensors before passing to Python.
- **Zero-Copy Streams**: Uses Node 25's `Transferable` streams to pass binary data from Python `uvloop` to React `SharedArrayBuffer` without serialization overhead.
- **Native TypeScript**: The BFF runs `.ts` files natively using Node 25's `--experimental-strip-types` flag, removing build steps.
- **Vim Runtime Bridge**: Exposes a neovim-lua-v5 instance via WebSockets to `GeminiCode.tsx`, enabling native modal editing in the browser.

## Jules Architecture Notes
The `orchestrator.py` now supports a conditional dispatch path. If a `control_mode` is detected in the request payload, the system will prioritize structural extraction (OpenPose/DepthMap) before triggering the primary diffusion pass via `ControlNetTool`.
