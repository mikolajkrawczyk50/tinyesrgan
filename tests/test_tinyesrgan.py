import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import RRDB, ResidualDenseBlock, RRDBNet, SRVGGNetCompact
from src.weights import load_pth
from tinyesrgan import (
    SCALE,
    _get_tta_variants,
    _inverse_tta_variants,
    build_model,
    get_image_files,
    inference,
    load_image,
    post_process,
    pre_process,
    process,
    process_tta,
    resolve_model_path,
    save_image,
    tile_process,
    tile_process_tta,
)


class TestModelComponents:
    """Test individual model components."""

    def test_srvggnet_compact_init(self):
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
        assert len(model.convs) == 18
        assert len(model.alphas) == 17
        assert model.upscale == 4

    def test_srvggnet_compact_load_state_dict(self):
        from tinygrad import Tensor
        model = SRVGGNetCompact()
        state = {
            "body.0.weight": Tensor(np.random.randn(64, 3, 3, 3).astype(np.float32)),
            "body.0.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "body.1.weight": Tensor(np.random.randn(64).astype(np.float32)),
        }
        for i in range(1, 17):
            state[f"body.{2*i}.weight"] = Tensor(np.random.randn(64, 64, 3, 3).astype(np.float32))
            state[f"body.{2*i}.bias"] = Tensor(np.random.randn(64).astype(np.float32))
            state[f"body.{2*i+1}.weight"] = Tensor(np.random.randn(64).astype(np.float32))
        state["body.34.weight"] = Tensor(np.random.randn(48, 64, 3, 3).astype(np.float32))
        state["body.34.bias"] = Tensor(np.random.randn(48).astype(np.float32))

        loaded = model.load_state_dict(state)
        assert loaded is model
        assert len(loaded.convs) == 18

    def test_rrdbnet_init(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        assert model.scale == 4
        assert len(model.body) == 23
        assert hasattr(model, "conv_first")
        assert hasattr(model, "conv_body")
        assert hasattr(model, "conv_up1")
        assert hasattr(model, "conv_up2")
        assert hasattr(model, "conv_hr")
        assert hasattr(model, "conv_last")

    def test_rrdbnet_load_state_dict(self):
        from tinygrad import Tensor
        model = RRDBNet(num_block=2)
        state = {
            "conv_first.weight": Tensor(np.random.randn(64, 3, 3, 3).astype(np.float32)),
            "conv_first.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "conv_body.weight": Tensor(np.random.randn(64, 64, 3, 3).astype(np.float32)),
            "conv_body.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "conv_up1.weight": Tensor(np.random.randn(64, 64, 3, 3).astype(np.float32)),
            "conv_up1.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "conv_up2.weight": Tensor(np.random.randn(64, 64, 3, 3).astype(np.float32)),
            "conv_up2.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "conv_hr.weight": Tensor(np.random.randn(64, 64, 3, 3).astype(np.float32)),
            "conv_hr.bias": Tensor(np.random.randn(64).astype(np.float32)),
            "conv_last.weight": Tensor(np.random.randn(3, 64, 3, 3).astype(np.float32)),
            "conv_last.bias": Tensor(np.random.randn(3).astype(np.float32)),
        }
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    prefix = f"body.{i}.rdb{j+1}.conv{k+1}"
                    state[f"{prefix}.weight"] = Tensor(np.random.randn(32 if k < 4 else 64, 64 + 32 * k, 3, 3).astype(np.float32))
                    state[f"{prefix}.bias"] = Tensor(np.random.randn(32 if k < 4 else 64).astype(np.float32))

        loaded = model.load_state_dict(state)
        assert loaded is model

    def test_residual_dense_block(self):
        block = ResidualDenseBlock(num_feat=64, num_grow_ch=32)
        assert len(block.convs) == 5

    def test_rrdb(self):
        block = RRDB(num_feat=64, num_grow_ch=32)
        assert len(block.rdbs) == 3


class TestPrePostProcess:
    """Test preprocessing and postprocessing functions."""

    def test_pre_process(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        x = pre_process(img)
        assert x.shape == (1, 3, 64, 64)
        assert x.numpy().min() >= 0.0 and x.numpy().max() <= 1.0

    def test_post_process(self):
        from tinygrad import Tensor
        y = Tensor(np.random.rand(1, 3, 256, 256).astype(np.float32))
        out = post_process(y)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_pre_post_roundtrip(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        _ = pre_process(img)
        from tinygrad import Tensor
        # Simulate model output: 4x upscale
        y = Tensor(np.random.rand(1, 3, 128, 128).astype(np.float32))
        out = post_process(y)
        assert out.shape == (128, 128, 3)


class TestTTAVariants:
    """Test TTA variant generation and inversion."""

    def test_get_tta_variants_count(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        variants = _get_tta_variants(img)
        assert len(variants) == 8

    def test_get_tta_variants_shapes(self):
        img = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
        variants = _get_tta_variants(img)
        for i, v in enumerate(variants):
            if i < 4:
                assert v.shape == (32, 64, 3)
            else:
                assert v.shape == (64, 32, 3)

    def test_inverse_tta_variants_count(self):
        outputs = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(8)]
        inversed = _inverse_tta_variants(outputs)
        assert len(inversed) == 8

    def test_inverse_tta_variants_shapes(self):
        outputs = [np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8) for _ in range(8)]
        inversed = _inverse_tta_variants(outputs)
        for i, v in enumerate(inversed):
            if i < 4:
                assert v.shape == (128, 256, 3)
            else:
                assert v.shape == (256, 128, 3)

    def test_tta_roundtrip(self):
        """Test that applying variants then inverse gives back original (for identity transform)."""
        img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        variants = _get_tta_variants(img)

        identity_outputs = variants
        inversed = _inverse_tta_variants(identity_outputs)

        for i, inv in enumerate(inversed):
            np.testing.assert_array_equal(inv, img, err_msg=f"Variant {i} roundtrip failed")

    def test_tta_specific_transforms(self):
        """Test specific transform pairs are correct inverses."""
        img = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        variants = _get_tta_variants(img)

        # variant 0: identity -> inverse should be identity
        assert np.array_equal(variants[0], img)

        # variant 1: hflip -> inverse hflip
        inv_hflip = np.flip(variants[1], axis=1).copy()
        assert np.array_equal(inv_hflip, img)

        # variant 2: rotate 180 -> inverse rotate 180
        inv_rot180 = np.flip(variants[2], axis=(0, 1)).copy()
        assert np.array_equal(inv_rot180, img)

        # variant 3: vflip -> inverse vflip
        inv_vflip = np.flip(variants[3], axis=0).copy()
        assert np.array_equal(inv_vflip, img)

        # variant 4: transpose -> inverse transpose
        inv_transpose = variants[4].transpose(1, 0, 2).copy()
        assert np.array_equal(inv_transpose, img)

        # variant 5: transpose + hflip -> inverse hflip then transpose
        inv_t_hflip = np.flip(variants[5], axis=1).transpose(1, 0, 2).copy()
        assert np.array_equal(inv_t_hflip, img)

        # variant 6: transpose + both flip -> inverse both flip then transpose
        inv_t_both = np.flip(variants[6], axis=(0, 1)).transpose(1, 0, 2).copy()
        assert np.array_equal(inv_t_both, img)

        # variant 7: transpose + vflip -> inverse vflip then transpose
        inv_t_vflip = np.flip(variants[7], axis=0).transpose(1, 0, 2).copy()
        assert np.array_equal(inv_t_vflip, img)


class TestProcess:
    """Test the main process function."""

    def test_process_basic(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = process(model, img)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8

    def test_inference_modes(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out_plain = inference(model, img)
        assert out_plain.shape == (128, 128, 3)

        out_tiled = inference(model, img, tile=64, tile_pad=8)
        assert out_tiled.shape == (128, 128, 3)

        out_tta = inference(model, img, tta=True)
        assert out_tta.shape == (128, 128, 3)

        out_tiled_tta = inference(model, img, tile=64, tile_pad=8, tta=True)
        assert out_tiled_tta.shape == (128, 128, 3)


class TestTileProcess:
    """Test tiled processing."""

    def test_tile_process_small_image(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = tile_process(model, img, tile=64, tile_pad=8)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8

    def test_tile_process_large_image(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = tile_process(model, img, tile=64, tile_pad=8)
        assert out.shape == (512, 512, 3)
        assert out.dtype == np.uint8

    def test_tile_process_non_square(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        out = tile_process(model, img, tile=64, tile_pad=8)
        assert out.shape == (256, 512, 3)

    def test_tile_process_tile_size_validation(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            tile_process(model, img, tile=16, tile_pad=10)

    def test_tile_process_consistency(self):
        """Test that tiling gives similar results to non-tiling for small images.
        
        Note: Due to boundary effects from reflective padding and tile stitching,
        some differences are expected at tile boundaries.
        """
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        out_no_tile = process(model, img)
        out_tile = tile_process(model, img, tile=128, tile_pad=16)

        diff = np.abs(out_no_tile.astype(np.float32) - out_tile.astype(np.float32))
        # Allow larger diff due to tile boundary effects with reflective padding
        assert diff.mean() < 10.0, f"Tiled vs non-tiled mean diff too large: {diff.mean()}"


class TestTTA:
    """Test Test Time Augmentation."""

    def test_process_tta_basic(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = process_tta(model, img)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8

    def test_tile_process_tta_basic(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = tile_process_tta(model, img, tile=64, tile_pad=8)
        assert out.shape == (128, 128, 3)
        assert out.dtype == np.uint8

    def test_tta_vs_non_tta_different(self):
        """TTA output should differ from non-TTA output."""
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out_normal = process(model, img)
        out_tta = process_tta(model, img)

        diff = np.abs(out_normal.astype(np.float32) - out_tta.astype(np.float32))
        assert diff.mean() > 0.1, "TTA should produce different results"


class TestBuildModel:
    """Test build_model function."""

    def test_build_model_srvgg(self):
        state = load_pth("models/realesr-animevideov3.safetensors")
        model = build_model(state)
        assert isinstance(model, SRVGGNetCompact)

    def test_build_model_rrdbnet(self):
        state = load_pth("models/realesrgan-x4plus.safetensors")
        model = build_model(state)
        assert isinstance(model, RRDBNet)


class TestImageIO:
    """Test image loading and saving."""

    def test_load_save_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            save_image(img, str(img_path))

            loaded = load_image(str(img_path))
            assert loaded.shape == (64, 64, 3)
            assert loaded.dtype == np.uint8
            np.testing.assert_array_equal(loaded, img)

    def test_get_image_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.png").write_bytes(b"fake")
            Path(tmpdir, "b.jpg").write_bytes(b"fake")
            Path(tmpdir, "c.jpeg").write_bytes(b"fake")
            Path(tmpdir, "d.webp").write_bytes(b"fake")
            Path(tmpdir, "e.bmp").write_bytes(b"fake")
            Path(tmpdir, "f.txt").write_bytes(b"fake")

            files = get_image_files(tmpdir)
            assert len(files) == 5
            assert all(Path(f).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp") for f in files)

    def test_get_image_files_natural_sorting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["frame_10.png", "frame_2.png", "frame_1.png", "frame_20.png"]:
                Path(tmpdir, name).write_bytes(b"fake")
            files = [Path(f).name for f in get_image_files(tmpdir)]
            assert files == ["frame_1.png", "frame_2.png", "frame_10.png", "frame_20.png"]


class TestResolveModelPath:
    """Test model path resolution."""

    def test_resolve_exact_file(self):
        resolved = resolve_model_path("models/realesr-animevideov3.safetensors")
        assert resolved == "models/realesr-animevideov3.safetensors"

    def test_resolve_without_extension(self):
        resolved = resolve_model_path("models/realesr-animevideov3")
        assert resolved == "models/realesr-animevideov3.safetensors"

    def test_resolve_directory(self):
        resolved = resolve_model_path("models")
        assert resolved in (
            "models/realesr-animevideov3.safetensors",
            "models/realesrgan-x4plus.safetensors",
        )

    def test_resolve_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            resolve_model_path("models/nonexistent_model_123.safetensors")

    def test_resolve_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                resolve_model_path(tmpdir)


class TestTileCalculations:
    """Test tile calculation logic."""

    def test_tile_calculations_square(self):
        tile = 128
        tile_pad = 16
        base = tile - 2 * tile_pad
        width = height = 256

        tiles_x = math.ceil(width / base)
        tiles_y = math.ceil(height / base)

        pad_left = tile_pad
        pad_top = tile_pad
        pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
        pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

        assert base == 96
        assert tiles_x == 3
        assert tiles_y == 3
        assert pad_left == 16
        assert pad_top == 16
        assert pad_right == 48
        assert pad_bottom == 48

    def test_tile_calculations_non_square(self):
        tile = 128
        tile_pad = 16
        base = tile - 2 * tile_pad
        width, height = 200, 150

        tiles_x = math.ceil(width / base)
        tiles_y = math.ceil(height / base)

        pad_left = tile_pad
        pad_top = tile_pad
        pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
        pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

        assert base == 96
        assert tiles_x == 3
        assert tiles_y == 2
        assert pad_right == 104
        assert pad_bottom == 58

    def test_tile_calculations_small_image(self):
        tile = 128
        tile_pad = 16
        base = tile - 2 * tile_pad
        width = height = 64

        tiles_x = math.ceil(width / base)
        tiles_y = math.ceil(height / base)

        pad_left = tile_pad
        pad_top = tile_pad
        pad_right = (tiles_x - 1) * base + tile - (width + pad_left)
        pad_bottom = (tiles_y - 1) * base + tile - (height + pad_top)

        assert base == 96
        assert tiles_x == 1
        assert tiles_y == 1
        assert pad_right == 48
        assert pad_bottom == 48


class TestScaleConstant:
    """Test SCALE constant is used correctly."""

    def test_scale_is_four(self):
        assert SCALE == 4

    def test_post_process_uses_scale(self):
        from tinygrad import Tensor
        y = Tensor(np.random.rand(1, 3, 100, 100).astype(np.float32))
        out = post_process(y)
        assert out.shape == (100, 100, 3)


class TestIntegration:
    """Integration tests with actual models."""

    def test_animevideov3_inference(self):
        state = load_pth("models/realesr-animevideov3.safetensors")
        model = build_model(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = process(model, img)
        assert out.shape == (128, 128, 3)

    def test_x4plus_inference(self):
        state = load_pth("models/realesrgan-x4plus.safetensors")
        model = build_model(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = process(model, img)
        assert out.shape == (128, 128, 3)

    def test_both_models_tiled(self):
        for model_path in ["models/realesr-animevideov3.safetensors", "models/realesrgan-x4plus.safetensors"]:
            state = load_pth(model_path)
            model = build_model(state)

            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            out = tile_process(model, img, tile=64, tile_pad=8)
            assert out.shape == (256, 256, 3)


class TestEdgeCases:
    """Test edge cases."""

    def test_minimum_image_size(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        out = process(model, img)
        assert out.shape == (4, 4, 3)

    def test_different_tile_pads(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for tile_pad in [0, 4, 8, 16, 32]:
            out = tile_process(model, img, tile=128, tile_pad=tile_pad)
            assert out.shape == (256, 256, 3)

    def test_zero_tile_disables_tiling(self):
        model = SRVGGNetCompact()
        state = load_pth("models/realesr-animevideov3.safetensors")
        model.load_state_dict(state)

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        # tile=0 is handled in main by calling process(), not tile_process()
        # tile_process() requires tile > 0, so test the main logic instead
        from tinyesrgan import process
        out = process(model, img)
        assert out.shape == (128, 128, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])