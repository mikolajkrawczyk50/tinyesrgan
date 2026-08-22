# tinyesrgan - RealESRGAN Super-Resolution (tinygrad)

**Minimal KISS project** — super-resolution in tinygrad with minimal dependencies and clean architecture.

## Project Structure

```
tinyesrgan/
├── pyproject.toml       # Build configuration & tinyesrgan CLI entrypoint
├── src/
│   ├── model.py         # Core model definitions (SRVGGNetCompact, RRDBNet, ResidualDenseBlock, RRDB)
│   └── weights.py       # Safetensors weight loader (load_pth)
├── tinyesrgan.py        # Main CLI entry point (single/dir mode, tiling, TTA, verbose, timing)
├── tests/
│   ├── test_tinyesrgan.py     # Unit & integration tests (components, tiling, TTA, CLI helpers)
│   └── test_against_torch.py  # Numerical parity tests against PyTorch reference
├── models/              # Pretrained models (.safetensors)
│   ├── realesr-animevideov3.safetensors
│   └── realesrgan-x4plus.safetensors
└── demo/                # Demo/test images
    └── i0.png
```

## Environment & Hardware

Prefer running with OpenCL on Radeon via RustiCL:
```bash
DEV=CL RUSTICL_ENABLE=radeonsi
```
(e.g., `DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan ...`)

### Environment Variable Note (`CPU`)

If your shell or build environment sets `CPU` to an architecture string (e.g., `CPU=x86_64`), tinygrad may attempt to parse it as a device index. If this occurs, unset `CPU` before running (`unset CPU` or `env -u CPU tinyesrgan ...`).

## Entry Point

**Main CLI:** `tinyesrgan` (or `python tinyesrgan.py`)

```bash
# Single image mode
DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan -i input.png -o output.png -m models/realesr-animevideov3.safetensors

# Directory mode (batch super-resolution)
DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan -i input_dir/ -o output_dir/ -m models/realesr-animevideov3.safetensors

# With spatial tiling for large images
DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan -i input.png -o output.png -t 128 --tile_pad 16

# With Test-Time Augmentation (8x inference, D4 dihedral group)
DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan -i input.png -o output.png -x
```

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | Models (`SRVGGNetCompact`, `RRDBNet`), custom PReLU & PixelShuffle |
| `src/weights.py` | Loads `.safetensors` checkpoints into tinygrad state dicts |
| `tinyesrgan.py` | CLI entry point, preprocessing/postprocessing, tiling, TTA |
| `tests/test_tinyesrgan.py` | Pytest suite for model layers, tiling, TTA, and CLI helpers |
| `tests/test_against_torch.py` | Reference parity test against PyTorch |

## Dependencies

- `tinygrad` (inference engine)
- `safetensors` (weight loading)
- `pillow` (image I/O)
- `numpy`

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

## Demo Data

Test images stored under `demo/`:
- `demo/i0.png`

## Commands

```bash
# Run CLI help
DEV=CL RUSTICL_ENABLE=radeonsi tinyesrgan -h

# Run full test suite
DEV=CL RUSTICL_ENABLE=radeonsi pytest -v
```

## Notes

- **Models**:
  - `SRVGGNetCompact` (`realesr-animevideov3`, ~2.5MB): 16-conv compact VGG network with PReLU and pixel shuffle. Recommended for fast testing and anime video.
  - `RRDBNet` (`realesrgan-x4plus`, ~67MB): 23 Residual-in-Residual Dense Blocks. High quality for general images.
- **Input/Output**: RGB images normalized to `[0, 1]`, output scaled 4x.
- **Tiling**: Reflect-padded overlapping patch tiling (default `tile = 128`, `tile_pad = 16`). Setting `-t 0` disables tiling.
- **TTA**: 8-variant dihedral $D_4$ group augmentation (rotations and flips) with inverse transform averaging.
- **Precision**: FP16 provides minimal speedup with visible quality loss on super-resolution. Use FP32.