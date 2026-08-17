import os

if os.environ.get("CPU") and not os.environ["CPU"].lstrip("-").isdigit():
    os.environ.pop("CPU")

from tinygrad import Tensor


class SRVGGNetCompact:
    """RealESRGAN SRVGGNetCompact (realesr-animevideov3) implemented in tinygrad.

    Mirrors the PyTorch reference arch (realesrgan/archs/srvgg_arch.py):

        body: conv(3->64) -> PReLU, then 16x (conv(64->64) -> PReLU),
              then conv(64 -> 3*upscale^2), PixelShuffle(upscale), + nearest upsample residual.

    Layer indexing matches the torch state dict: body.<2i>.weight/bias for conv i,
    body.<2i+1>.weight for the PReLU alpha of layer i.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        dtype=None,
    ):
        self.upscale = upscale
        self.convs: list[tuple[Tensor, Tensor]] = []
        self.alphas: list[Tensor] = []

        self.convs.append(
            (
                Tensor.zeros(num_feat, num_in_ch, 3, 3, dtype=dtype),
                Tensor.zeros(num_feat, dtype=dtype),
            )
        )
        self.alphas.append(Tensor.zeros(num_feat, dtype=dtype))

        for _ in range(num_conv):
            self.convs.append(
                (
                    Tensor.zeros(num_feat, num_feat, 3, 3, dtype=dtype),
                    Tensor.zeros(num_feat, dtype=dtype),
                )
            )
            self.alphas.append(Tensor.zeros(num_feat, dtype=dtype))

        self.convs.append(
            (
                Tensor.zeros(num_out_ch * upscale * upscale, num_feat, 3, 3, dtype=dtype),
                Tensor.zeros(num_out_ch * upscale * upscale, dtype=dtype),
            )
        )

    def load_state_dict(self, state: dict[str, Tensor]) -> "SRVGGNetCompact":
        for i in range(len(self.convs)):
            self.convs[i] = (state[f"body.{2 * i}.weight"], state[f"body.{2 * i}.bias"])
        for i in range(len(self.alphas)):
            self.alphas[i] = state[f"body.{2 * i + 1}.weight"]
        return self

    @staticmethod
    def _prelu(x: Tensor, alpha: Tensor) -> Tensor:
        a = alpha.reshape(1, alpha.numel(), 1, 1)
        return (x > 0).where(x, a * x)

    @staticmethod
    def _pixel_shuffle(x: Tensor, r: int) -> Tensor:
        n, c2, h, w = x.shape
        c = c2 // (r * r)
        return x.reshape(n, c, r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(n, c, h * r, w * r)

    def __call__(self, x: Tensor) -> Tensor:
        out = x
        for i in range(len(self.alphas)):
            w, b = self.convs[i]
            out = self._prelu(out.conv2d(w, b, padding=1), self.alphas[i])
        w, b = self.convs[-1]
        out = out.conv2d(w, b, padding=1)
        out = self._pixel_shuffle(out, self.upscale)
        base = x.repeat_interleave(self.upscale, dim=2).repeat_interleave(self.upscale, dim=3)
        return out + base


class ResidualDenseBlock:
    """RDB (5 convs, growing channels) used inside RRDB."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, dtype=None):
        self.convs: list[tuple[Tensor, Tensor]] = []
        in_ch = num_feat
        for i in range(5):
            out_ch = num_grow_ch if i < 4 else num_feat
            self.convs.append(
                (Tensor.zeros(out_ch, in_ch, 3, 3, dtype=dtype), Tensor.zeros(out_ch, dtype=dtype))
            )
            in_ch += num_grow_ch

    def __call__(self, x: Tensor) -> Tensor:
        feats = [x]
        for i in range(4):
            w, b = self.convs[i]
            y = feats[0].cat(*feats[1:], dim=1).conv2d(w, b, padding=1).leaky_relu(0.2)
            feats.append(y)
        w, b = self.convs[4]
        y = feats[0].cat(*feats[1:], dim=1).conv2d(w, b, padding=1)
        return y * 0.2 + x


class RRDB:
    """Residual in Residual Dense Block: 3 RDBs + 0.2 residual."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, dtype=None):
        self.rdbs = [ResidualDenseBlock(num_feat, num_grow_ch, dtype) for _ in range(3)]

    def __call__(self, x: Tensor) -> Tensor:
        out = self.rdbs[0](x)
        out = self.rdbs[1](out)
        out = self.rdbs[2](out)
        return out * 0.2 + x


class RRDBNet:
    """RealESRGAN_x4plus (RRDBNet, 23 RRDB blocks) implemented in tinygrad.

    Mirrors the torch reference (basicsr/archs/rrdbnet_arch.py):

        conv_first(3->64) -> 23x RRDB -> conv_body (residual) ->
        2x (nearest x2 + conv64->64 + LeakyReLU(0.2)) ->
        conv_hr(64->64) + LeakyReLU(0.2) -> conv_last(64->3)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        dtype=None,
    ):
        self.scale = scale
        in_ch = num_in_ch * {2: 4, 1: 16}.get(scale, 1)
        self.conv_first: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_feat, in_ch, 3, 3, dtype=dtype),
            Tensor.zeros(num_feat, dtype=dtype),
        )
        self.body = [RRDB(num_feat, num_grow_ch, dtype) for _ in range(num_block)]
        self.conv_body: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_feat, num_feat, 3, 3, dtype=dtype),
            Tensor.zeros(num_feat, dtype=dtype),
        )
        self.conv_up1: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_feat, num_feat, 3, 3, dtype=dtype),
            Tensor.zeros(num_feat, dtype=dtype),
        )
        self.conv_up2: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_feat, num_feat, 3, 3, dtype=dtype),
            Tensor.zeros(num_feat, dtype=dtype),
        )
        self.conv_hr: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_feat, num_feat, 3, 3, dtype=dtype),
            Tensor.zeros(num_feat, dtype=dtype),
        )
        self.conv_last: tuple[Tensor, Tensor] = (
            Tensor.zeros(num_out_ch, num_feat, 3, 3, dtype=dtype),
            Tensor.zeros(num_out_ch, dtype=dtype),
        )

    def load_state_dict(self, state: dict[str, Tensor]) -> "RRDBNet":
        for name in ("conv_first", "conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"):
            setattr(self, name, (state[f"{name}.weight"], state[f"{name}.bias"]))
        for i, block in enumerate(self.body):
            for j, rdb in enumerate(block.rdbs):
                for k in range(5):
                    prefix = f"body.{i}.rdb{j + 1}.conv{k + 1}"
                    rdb.convs[k] = (state[f"{prefix}.weight"], state[f"{prefix}.bias"])
        return self

    @staticmethod
    def _pixel_unshuffle(x: Tensor, r: int) -> Tensor:
        n, c, h, w = x.shape
        return x.reshape(n, c, h // r, r, w // r, r).permute(0, 3, 5, 1, 2, 4).reshape(n, c * r * r, h // r, w // r)

    def __call__(self, x: Tensor) -> Tensor:
        feat = x
        if self.scale == 2:
            feat = self._pixel_unshuffle(feat, 2)
        elif self.scale == 1:
            feat = self._pixel_unshuffle(feat, 4)

        w, b = self.conv_first
        feat = feat.conv2d(w, b, padding=1)
        out = feat
        for block in self.body:
            out = block(out)
        w, b = self.conv_body
        feat = feat + out.conv2d(w, b, padding=1)

        for up in (self.conv_up1, self.conv_up2):
            w, b = up
            feat = feat.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).conv2d(w, b, padding=1).leaky_relu(0.2)

        w, b = self.conv_hr
        feat = feat.conv2d(w, b, padding=1).leaky_relu(0.2)
        w, b = self.conv_last
        return feat.conv2d(w, b, padding=1)
