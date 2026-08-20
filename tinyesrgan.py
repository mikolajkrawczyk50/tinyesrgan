#!/usr/bin/env python3
"""
tinyesrgan - RealESRGAN super-resolution CLI (tinygrad)
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
from PIL import Image
from tinygrad import Tensor

from src.model import RRDBNet, SRVGGNetCompact
from src.weights import load_pth


SCALE = 4


def build_model(state: dict):
    if "conv_first.weight" in state:
        return RRDBNet().load_state_dict(state)
    return SRVGGNetCompact().load_state_dict(state)


def load_image(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    return img


def save_image(img_rgb: np.ndarray, path: str):
    Image.fromarray(img_rgb).save(path)


def get_image_files(dir_path: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    files.sort()
    return files


def pre_process(img_rgb: np.ndarray, pre_pad: int):
    x = Tensor(img_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    if pre_pad:
        x = x.pad((0, pre_pad, 0, pre_pad), mode="reflect")
    return x


def post_process(y: Tensor, pre_pad: int) -> np.ndarray:
    if pre_pad:
        y = y[:, :, : y.shape[2] - pre_pad * SCALE, : y.shape[3] - pre_pad * SCALE]
    out = np.clip(y.clip(0.0, 1.0).numpy()[0] * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return out.transpose(1, 2, 0)


def process(model, img_rgb: np.ndarray, pre_pad: int) -> np.ndarray:
    x = pre_process(img_rgb, pre_pad)
    y = model(x)
    return post_process(y, pre_pad)


def _get_tta_variants(img: np.ndarray) -> list[np.ndarray]:
    """Generate 8 TTA variants (D4 dihedral group) matching realesrgan-ncnn-vulkan."""
    h, w = img.shape[:2]
    variants = [
        img,
        np.flip(img, axis=1).copy(),  # hflip
        np.flip(img, axis=(0, 1)).copy(),  # rotate 180 (both flip)
        np.flip(img, axis=0).copy(),  # vflip
        img.transpose(1, 0, 2).copy(),  # transpose (rotate 90)
        np.flip(img.transpose(1, 0, 2).copy(), axis=1),  # transpose + hflip
        np.flip(img.transpose(1, 0, 2).copy(), axis=(0, 1)),  # transpose + both flip
        np.flip(img.transpose(1, 0, 2).copy(), axis=0),  # transpose + vflip
    ]
    return variants


def _inverse_tta_variants(outputs: list[np.ndarray]) -> list[np.ndarray]:
    """Inverse transform 8 TTA variants back to original orientation."""
    h, w = outputs[0].shape[:2]
    inversed = [
        outputs[0],
        np.flip(outputs[1], axis=1).copy(),  # inverse hflip
        np.flip(outputs[2], axis=(0, 1)).copy(),  # inverse rotate 180
        np.flip(outputs[3], axis=0).copy(),  # inverse vflip
        outputs[4].transpose(1, 0, 2).copy(),  # inverse transpose
        np.flip(outputs[5], axis=1).transpose(1, 0, 2).copy(),  # inverse transpose + hflip
        np.flip(outputs[6], axis=(0, 1)).transpose(1, 0, 2).copy(),  # inverse transpose + both flip
        np.flip(outputs[7], axis=0).transpose(1, 0, 2).copy(),  # inverse transpose + vflip
    ]
    return inversed


def process_tta(model, img_rgb: np.ndarray, pre_pad: int) -> np.ndarray:
    """Test Time Augmentation: average 8 predictions (D4 dihedral group) matching realesrgan-ncnn-vulkan."""
    variants = _get_tta_variants(img_rgb)
    results = [process(model, v, pre_pad) for v in variants]
    inversed = _inverse_tta_variants(results)
    return np.clip(np.mean(inversed, axis=0) + 0.5, 0, 255).round().astype(np.uint8)


def tile_process_tta(
    model,
    img_rgb: np.ndarray,
    tile: int = 128,
    tile_pad: int | None = None,
    pre_pad: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    """Tiled TTA processing with 8 variants (D4 dihedral group)."""
    variants = _get_tta_variants(img_rgb)
    results = [tile_process(model, v, tile, tile_pad, pre_pad, verbose) for v in variants]
    inversed = _inverse_tta_variants(results)
    return np.clip(np.mean(inversed, axis=0) + 0.5, 0, 255).round().astype(np.uint8)


def tile_process(
    model,
    img_rgb: np.ndarray,
    tile: int = 128,
    tile_pad: int | None = None,
    pre_pad: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    import math
    import time
    from tinygrad import TinyJit

    if tile_pad is None:
        tile_pad = tile // 8 if tile > 0 else 0
    base = tile - 2 * tile_pad
    assert base > 0, f"tile size ({tile}) must be greater than 2 * tile_pad ({2 * tile_pad})"

    x = pre_process(img_rgb, pre_pad)
    _, c, height, width = x.shape
    out = np.zeros((1, c, height * SCALE, width * SCALE), dtype=np.float32)

    tiles_x = math.ceil(width / base)
    tiles_y = math.ceil(height / base)

    pad_left = tile_pad
    pad_top = tile_pad
    pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
    pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

    x_np = x.numpy()
    x_padded = np.pad(
        x_np,
        ((0, 0), (0, 0), (pad_top, max(0, pad_bottom)), (pad_left, max(0, pad_right))),
        mode="reflect",
    )

    model_jit = TinyJit(model)

    n_tiles = tiles_x * tiles_y
    tile_times = np.zeros(n_tiles, dtype=np.float64) if verbose else None
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            sx = tx * base
            sy = ty * base
            tile_np = x_padded[:, :, sy : sy + tile, sx : sx + tile]

            t_start = time.perf_counter()
            tile_t = Tensor(tile_np)
            y_tile = model_jit(tile_t).numpy()

            if verbose:
                tile_times[ty * tiles_x + tx] = time.perf_counter() - t_start
                print(f"    tile {ty * tiles_x + tx + 1}/{n_tiles} (x={tx}, y={ty}): {tile_times[ty * tiles_x + tx]:.3f}s")

            in_x = tx * base
            in_y = ty * base
            w_valid = min(base, width - in_x)
            h_valid = min(base, height - in_y)

            oy_t = tile_pad * SCALE
            ox_t = tile_pad * SCALE
            h_t = h_valid * SCALE
            w_t = w_valid * SCALE

            oy = in_y * SCALE
            ox = in_x * SCALE

            out[:, :, oy : oy + h_t, ox : ox + w_t] = y_tile[:, :, oy_t : oy_t + h_t, ox_t : ox_t + w_t]

    if verbose:
        print(
            f"  Tiles: {n_tiles} total, avg {tile_times.mean() * 1000:.1f}ms, "
            f"min {tile_times.min() * 1000:.1f}ms, max {tile_times.max() * 1000:.1f}ms, "
            f"total {tile_times.sum():.3f}s"
        )

    y = Tensor(out)
    return post_process(y, pre_pad)


def main():
    parser = argparse.ArgumentParser(description="RealESRGAN super-resolution (tinygrad)")
    parser.add_argument("-i", "--input", required=True, help="Input image path or directory")
    parser.add_argument("-o", "--output", required=True, help="Output image path (file mode) or directory (dir mode)")
    parser.add_argument("-m", "--model", default="models/realesr-animevideov3.safetensors", help="Path to .safetensors model (default: models/realesr-animevideov3.safetensors)")
    parser.add_argument("-t", "--tile", type=int, default=128, help="Tile size for processing, 0 disables tiling (default: 128)")
    parser.add_argument("--tile_pad", type=int, default=None, help="Pad around each tile (default: tile/8, 0 if tiling disabled)")
    parser.add_argument("--pre_pad", type=int, default=10, help="Reflect padding before inference (default: 10)")
    parser.add_argument("-x", action="store_true", help="Enable Test Time Augmentation (8x inference, D4 dihedral group)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.tile_pad is None:
        args.tile_pad = args.tile // 8 if args.tile > 0 else 0

    if not os.path.exists(args.input):
        parser.error(f"Input not found: {args.input}")

    is_dir = os.path.isdir(args.input)

    if is_dir:
        input_files = get_image_files(args.input)
        if not input_files:
            sys.exit(f"Error: No images found in {args.input}")
        os.makedirs(args.output, exist_ok=True)
    else:
        input_files = [args.input]

    if args.verbose:
        print(f"Loading model from {args.model}...")
        if args.tile > 0:
            print(f"Tiling enabled: tile={args.tile}, tile_pad={args.tile_pad}, base={args.tile - 2 * args.tile_pad}")
        if args.x:
            print("TTA enabled: 8x inference (D4 dihedral group)")

    start_time = time.time()
    state = load_pth(args.model)
    model = build_model(state)
    if args.verbose:
        print(f"Model loaded in {time.time() - start_time:.2f}s")

    if args.verbose:
        print(f"Processing {len(input_files)} image(s)...")

    for idx, in_path in enumerate(input_files):
        if args.verbose:
            print(f"  [{idx+1}/{len(input_files)}] {os.path.basename(in_path)}")

        img_rgb = load_image(in_path)

        t0 = time.time()
        if args.x:
            if args.tile > 0:
                out = tile_process_tta(model, img_rgb, tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, verbose=args.verbose)
            else:
                out = process_tta(model, img_rgb, pre_pad=args.pre_pad)
        else:
            if args.tile > 0:
                out = tile_process(model, img_rgb, tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, verbose=args.verbose)
            else:
                out = process(model, img_rgb, pre_pad=args.pre_pad)

        if args.verbose:
            print(f"    Inference took {time.time() - t0:.2f}s, output {out.shape[1]}x{out.shape[0]}")

        if is_dir:
            out_name = os.path.splitext(os.path.basename(in_path))[0] + "_realesrgan.png"
            out_path = os.path.join(args.output, out_name)
        else:
            out_path = args.output

        save_image(out, out_path)

        if args.verbose:
            print(f"    Saved: {out_path}")

    if args.verbose:
        print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()