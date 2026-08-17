# realesrgan-tinygrad

RealESRGAN inference in [tinygrad](https://github.com/tinygrad/tinygrad), no PyTorch at inference time (PyTorch only used to read the `.pth` checkpoint).

Supported models (auto-detected from the state dict):

| model | arch | `model.py` class |
|-------|------|------------------|
| `realesr-animevideov3` | SRVGGNetCompact (18 convs) | `SRVGGNetCompact` |
| `RealESRGAN_x4plus` | RRDBNet (23× RRDB) | `RRDBNet` |

## Architecture

`SRVGGNetCompact` mirrors `realesrgan/archs/srvgg_arch.py`:

```
conv(3->64) + PReLU
16x (conv(64->64) + PReLU)
conv(64 -> 3*upscale^2) + PixelShuffle(upscale)
+ nearest-upsampled input (residual)
```

`RRDBNet` mirrors `basicsr/archs/rrdbnet_arch.py`:

```
conv_first(3->64) -> 23x RRDB -> conv_body (residual add) ->
2x (nearest x2 + conv(64->64) + LeakyReLU(0.2)) ->
conv_hr(64->64) + LeakyReLU(0.2) -> conv_last(64->3)
```

Weights load straight from the torch state dict.

## Files

| file | purpose |
|------|---------|
| `model.py` | SRVGGNetCompact + RRDBNet in tinygrad |
| `weights.py` | `.pth` -> dict of tinygrad tensors |
| `realesrgan.py` | CLI: pre/post-process (reflect pad), tiling, fp16 |
| `test_against_torch.py` | numerical check vs torch references |

## Usage

```bash
pip install -r requirements.txt

curl -L -o realesr-animevideov3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
curl -L -o RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

python realesrgan.py -i input.png -o output.png -m realesr-animevideov3.pth
python realesrgan.py -i input.png -o output.png -m RealESRGAN_x4plus.pth
```

Options: `--fp16` half precision, `--tile N` tile processing (lower memory), `--pre_pad N` (default 10, mirror Real-ESRGAN), `--scale 4`.

## Verify

```bash
python test_against_torch.py realesr-animevideov3.pth RealESRGAN_x4plus.pth
```

Both outputs match their torch references to ~3e-6 (fp32).

## Performance

`realesr-animevideov3` (18 convs) is fast. `RealESRGAN_x4plus` (345 RRDB convs) is slow on CPU — expect ~2 min per 16×16 tile; use `--tile` and/or `--fp16` for larger images.

## Note

tinygrad auto-selects a device by reading env vars named after backends (e.g. `CPU`). If your shell exports `CPU=x86_64` (common on SUSE), it breaks device detection; the scripts strip a non-numeric `CPU` env var on import.