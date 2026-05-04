import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .base import Conditioner
from .conditioner import EnergyChannelConditioner

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], -1)
        return F.pad(emb, (0, self.dim % 2))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, t_dim=128, drop=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, cout))
        self.n2 = nn.GroupNorm(min(32, cout), cout)
        self.drop = nn.Dropout(drop)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, te):
        h = self.act(self.n1(x))
        h = self.c1(h) + self.t_mlp(te)[:, :, None, None]
        h = self.act(self.n2(h))
        h = self.c2(self.drop(h))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x):
        return self.c(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):

    def __init__(
        self,
        encoder_shape: Tuple[int, int, int],
        conditioner: Optional[Conditioner] = None,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_levels: Tuple[int, ...] = (1, 2, 3),
        time_embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        C, H, W = encoder_shape
        self.encoder_shape = encoder_shape
        self.conditioner = conditioner
        self.n_levels = len(channel_mults)
        self.n_res = num_res_blocks

        n_downs = self.n_levels - 1
        assert H % (2 ** n_downs) == 0 and W % (2 ** n_downs) == 0, (
            f"Encoder shape {encoder_shape} not divisible by 2^{n_downs}. "
            f"Reduce channel_mults length or change encoder image_size.")


        extra_in = conditioner.extra_in_channels if conditioner else 0
        in_ch  = C + extra_in
        out_ch = C                                # predict noise of same shape

        extra_emb = conditioner.extra_embed_dim if conditioner else 0
        self.t_emb = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        if extra_emb > 0:
            self.cond_proj = nn.Linear(extra_emb, time_embed_dim)
        else:
            self.cond_proj = None

        self.init_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        skips = [base_channels]
        ch = base_channels

        for lv in range(self.n_levels):
            oc = base_channels * channel_mults[lv]
            blk = nn.ModuleList()
            for _ in range(num_res_blocks):
                blk.append(ResBlock(ch, oc, time_embed_dim, dropout))
                ch = oc
                if lv in attn_levels:
                    blk.append(AttnBlock(ch))
                skips.append(ch)
            self.enc.append(blk)
            if lv < self.n_levels - 1:
                self.downs.append(Down(ch))
                skips.append(ch)
            else:
                self.downs.append(None)


        self.mid1 = ResBlock(ch, ch, time_embed_dim, dropout)
        self.mid_a = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_embed_dim, dropout)


        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for lv in reversed(range(self.n_levels)):
            oc = base_channels * channel_mults[lv]
            blk = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                sc = skips.pop()
                blk.append(ResBlock(ch + sc, oc, time_embed_dim, dropout))
                ch = oc
                if lv in attn_levels:
                    blk.append(AttnBlock(ch))
            self.dec.append(blk)
            self.ups.append(Up(ch) if lv > 0 else None)


        self.final = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )
        assert len(skips) == 0, f"Skip mismatch: {len(skips)} left"

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        B, C, H, W = x_t.shape
        cond = cond or {}


        if self.conditioner is not None:
            spatial_c = self.conditioner.spatial(cond, (H, W))
            if spatial_c is not None:
                x_t = torch.cat([x_t, spatial_c], dim=1)


        te = self.t_emb(t)
        if self.conditioner is not None and self.cond_proj is not None:
            v = self.conditioner.vector(cond)
            if v is not None:
                te = te + self.cond_proj(v)

        h = self.init_conv(x_t)
        sk = [h]

        for lv in range(self.n_levels):
            blk = self.enc[lv]; idx = 0
            for _ in range(self.n_res):
                h = blk[idx](h, te); idx += 1
                if idx < len(blk) and isinstance(blk[idx], AttnBlock):
                    h = blk[idx](h); idx += 1
                sk.append(h)
            if self.downs[lv] is not None:
                h = self.downs[lv](h); sk.append(h)

        h = self.mid1(h, te); h = self.mid_a(h); h = self.mid2(h, te)

        for di, lv in enumerate(reversed(range(self.n_levels))):
            blk = self.dec[di]; idx = 0
            for _ in range(self.n_res + 1):
                s = sk.pop()
                if h.shape[2:] != s.shape[2:]:
                    h = F.interpolate(h, s.shape[2:], mode='nearest')
                h = torch.cat([h, s], 1)
                h = blk[idx](h, te); idx += 1
                if idx < len(blk) and isinstance(blk[idx], AttnBlock):
                    h = blk[idx](h); idx += 1
            if self.ups[di] is not None:
                h = self.ups[di](h)

        return self.final(h)



def build_unet_for_encoder(encoder, conditioner=None, **kwargs):
    return UNet(encoder_shape=encoder.output_shape,
                conditioner=conditioner, **kwargs)


if __name__ == "__main__":
    from ..encoding.base_image import BaseImageEncoder
    from ..encoding.hilbert3ch_image import Hilbert3ChEncoder
    from ..encoding.hilbert4ch_image import Hilbert4ChEncoder

    import numpy as np

    centroid = np.zeros(3)
    encoders = {
        "Base":       BaseImageEncoder(image_size=24, centroid=centroid),
        "Hilbert4Ch": Hilbert4ChEncoder(image_size=32, centroid=centroid),
        "Hilbert3Ch": Hilbert3ChEncoder(image_size=32, centroid=centroid),
    }

    for name, enc in encoders.items():
        C, H, W = enc.output_shape
        depth = 4 if H % 16 == 0 else 3
        mults = (1, 2, 4, 8)[:depth]
        cond = EnergyChannelConditioner(n_channels=C)

        model = UNet(
            encoder_shape=enc.output_shape,
            conditioner=cond,
            base_channels=64,
            channel_mults=mults,
            num_res_blocks=2,
            attn_levels=(1, 2),
            time_embed_dim=128,
        )
        x = torch.randn(2, C, H, W)
        t = torch.randint(0, 1000, (2,))
        energy = torch.tensor([0.2, 0.5])
        out = model(x, t, cond={"energy": energy})
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s} in={enc.output_shape}  out={tuple(out.shape)}  "
              f"params={n_params:,}")