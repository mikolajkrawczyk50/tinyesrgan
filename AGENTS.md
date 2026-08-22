# Development Notes

## Philosophy

tinyesrgan is a minimal, KISS (Keep It Simple, Stupid) super-resolution project built on tinygrad. Keep dependencies minimal, code simple and readable, and avoid unnecessary abstractions.

## Recommended Runtime Environment

For best performance on AMD GPUs, use:
```
RUSTICL_ENABLE=radeonsi DEV=CL
```

This enables Rusticl OpenCL backend with RadeonSI driver.

## Model Conversion

Convert PyTorch `.pth` checkpoints to `.safetensors`:
```bash
python -c "
import torch
from safetensors.torch import save_file

for pth in ['realesr-animevideov3.pth', 'RealESRGAN_x4plus.pth']:
    ckpt = torch.load(pth, map_location='cpu', weights_only=True)
    params = ckpt['params_ema'] if 'params_ema' in ckpt else ckpt['params']
    save_file({k: v.cpu() for k, v in params.items()}, pth.replace('.pth', '.safetensors'))
"
```

## Testing

Prefer `realesr-animevideov3.safetensors` (~2.5MB) for testing — smaller, faster. `RealESRGAN_x4plus.safetensors` (~67MB) is heavier and may timeout on GPU.

## Precision

FP16 not worth it — minimal speedup, visible quality loss on super-resolution. Use FP32.