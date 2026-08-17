import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from model import RRDBNet, SRVGGNetCompact
from weights import load_pth


class TorchRRDB(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = TorchRDB(num_feat, num_grow_ch)
        self.rdb2 = TorchRDB(num_feat, num_grow_ch)
        self.rdb3 = TorchRDB(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class TorchRRDBNet(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.ModuleList([TorchRRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.body[0](feat)
        for blk in self.body[1:]:
            body = blk(body)
        feat = feat + self.conv_body(body)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class TorchRDB(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class TorchSRVGG(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = torch.nn.ModuleList()
        self.body.append(torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(torch.nn.PReLU(num_parameters=num_feat))
        self.body.append(torch.nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = torch.nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def test_reflect_pad():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, (1, 3, 64, 64)).astype(np.float32)
    pre_pad = 10
    torch_ref = F.pad(torch.from_numpy(a), (0, pre_pad, 0, pre_pad), mode="reflect")
    from tinygrad import Tensor

    tiny = Tensor(a).pad((0, pre_pad, 0, pre_pad), mode="reflect")
    diff = np.abs(torch_ref.numpy() - tiny.numpy()).max()
    assert diff < 1e-6, f"reflect pad mismatch: {diff}"
    print(f"[ok] reflect pad max diff = {diff:.2e}")


def test_full_model(pth_path, st_path):
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    params = sd["params_ema"] if "params_ema" in sd else sd["params"]

    torch_model = TorchSRVGG()
    torch_model.load_state_dict(params)
    torch_model.eval()

    tiny_model = SRVGGNetCompact().load_state_dict(load_pth(st_path))

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, (1, 3, 32, 32)).astype(np.float32)

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    from tinygrad import Tensor

    y_tiny = tiny_model(Tensor(x)).numpy()

    diff = np.abs(y_torch - y_tiny)
    print(f"output shape: torch {y_torch.shape} vs tinygrad {y_tiny.shape}")
    print(f"max abs diff = {diff.max():.4e}")
    print(f"mean abs diff = {diff.mean():.4e}")
    assert y_torch.shape == y_tiny.shape
    assert diff.max() < 1e-3, "outputs differ too much from torch reference"


def test_rrdbnet(pth_path, st_path):
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    params = sd["params_ema"] if "params_ema" in sd else sd["params"]

    torch_model = TorchRRDBNet()
    torch_model.load_state_dict(params)
    torch_model.eval()

    tiny_model = RRDBNet().load_state_dict(load_pth(st_path))

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (1, 3, 32, 32)).astype(np.float32)

    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)).numpy()

    from tinygrad import Tensor

    y_tiny = tiny_model(Tensor(x)).numpy()

    diff = np.abs(y_torch - y_tiny)
    print(f"output shape: torch {y_torch.shape} vs tinygrad {y_tiny.shape}")
    print(f"max abs diff = {diff.max():.4e}")
    print(f"mean abs diff = {diff.mean():.4e}")
    assert y_torch.shape == y_tiny.shape
    assert diff.max() < 1e-3, "outputs differ too much from torch reference"


def test_tiled_golden_images(model_name: str, ckpt_path: str, golden_dir: str):
    """Compare tinygrad tiled inference output against torch golden images (tile=32, pre_pad=10).

    Golden images are generated with the original Real-ESRGAN impl (tile=32, tile_pad=10, pre_pad=10).
    """
    from pathlib import Path

    from realesrgan import build_model, tile_process

    model = build_model(load_pth(ckpt_path))
    for in_path in sorted(Path("golden_inputs").glob("*.png")):
        golden_path = Path(golden_dir) / (in_path.stem + "_out.png")
        assert golden_path.exists(), f"missing golden {golden_path}"

        img = np.asarray(Image.open(in_path).convert("RGB"))
        out = tile_process(model, img, scale=4, pre_pad=10, tile=32, tile_pad=10, fp16=False)
        golden = np.asarray(Image.open(golden_path).convert("RGB")).astype(np.float32)

        assert out.shape == golden.shape, f"shape mismatch {out.shape} vs {golden.shape}"
        diff = np.abs(out.astype(np.float32) - golden)
        print(f"[tiled {model_name}] {in_path.name}: max diff = {diff.max()}, mean diff = {diff.mean():.4f}")
        assert diff.max() <= 1, f"tiled output differs from torch golden: {in_path.name} max diff {diff.max()}"


if __name__ == "__main__":
    import sys

    pth_anime = sys.argv[1] if len(sys.argv) > 1 else "realesr-animevideov3.pth"
    pth_x4plus = sys.argv[2] if len(sys.argv) > 2 else "RealESRGAN_x4plus.pth"
    st_anime = pth_anime.replace(".pth", ".safetensors")
    st_x4plus = pth_x4plus.replace(".pth", ".safetensors")

    test_reflect_pad()
    test_full_model(pth_anime, st_anime)
    test_rrdbnet(pth_x4plus, st_x4plus)
    test_tiled_golden_images("animevideov3", st_anime, "golden_outputs_torch_animevideov3_t32")
    test_tiled_golden_images("x4plus", st_x4plus, "golden_outputs_torch_x4plus_t32")
    print("all tests passed")