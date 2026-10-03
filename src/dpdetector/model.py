# SPDX-License-Identifier: Apache-2.0
"""Frozen FLUX VAE encoder, learned cross-level comparisons, transformer classifier."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FrozenFluxEncoder(nn.Module):
    def __init__(self, directory=None, vae_config=None, load_vae=True):
        super().__init__()
        from diffusers import AutoencoderKLFlux2
        # Dependency: Diffusers AutoencoderKLFlux2 (Apache-2.0); BFL FLUX.2 Klein VAE weights.
        vae = (AutoencoderKLFlux2.from_pretrained(str(directory), local_files_only=True) if load_vae
               else AutoencoderKLFlux2.from_config(vae_config))
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.requires_grad_(False)
        self.eval()
        self._features = {}
        for index in (0, 1, 2):
            self.encoder.down_blocks[index].resnets[-1].register_forward_hook(self._hook(index))

    def _hook(self, index):
        def capture(module, args, result):
            self._features[index] = result
        return capture

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        # Posterior-mean VAE features (Kingma and Welling, arXiv:1312.6114); TF32 follows runtime settings.
        with torch.autocast(device_type=images.device.type, enabled=False):
            moments = self.quant_conv(self.encoder(images.float() * 2 - 1))
            mean = moments.chunk(2, dim=1)[0]
            result = (self._features.pop(0), self._features.pop(1), self._features.pop(2), F.pixel_unshuffle(mean, 2))
        return result


class EarlyProjection(nn.Module):
    def __init__(self, channels, reductions, width):
        super().__init__()
        layers = []
        for _ in range(reductions):
            layers.extend([nn.Conv2d(channels, 128, 3, 2, 1), nn.GroupNorm(16, 128), nn.SiLU()])
            channels = 128
        self.spatial = nn.Sequential(*layers)
        self.projection = nn.Conv2d(128, width, 1)

    def forward(self, x):
        return self.projection(self.spatial(x)).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, width=1024, heads=16):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(width * 4, width))
        # Research basis: LayerScale, Touvron et al. (2021), arXiv:2103.17239; original implementation here.
        self.scale1 = nn.Parameter(torch.full((width,), 0.01))
        self.scale2 = nn.Parameter(torch.full((width,), 0.01))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Research basis: scaled dot-product attention, Vaswani et al. (2017), arXiv:1706.03762.
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.scale1 * self.proj(y)
        return x + self.scale2 * self.mlp(self.norm2(x))


class Detector(nn.Module):
    def __init__(self, vae_directory=None, width=1024, depth=20, checkpoint_blocks=True, heads=16, vae_config=None, load_vae=True):
        super().__init__()
        self.backbone = FrozenFluxEncoder(vae_directory, vae_config, load_vae)
        self.early1 = EarlyProjection(128, 4, width)
        self.early2 = EarlyProjection(256, 3, width)
        self.early3 = EarlyProjection(512, 2, width)
        self.latent = nn.Linear(128, width)
        self.level_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.fusion = nn.Sequential(nn.Linear(width * 7, width), nn.LayerNorm(width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.randn(1, 65, width) * 0.01)
        active = torch.zeros(8, 8, dtype=torch.long)
        active[1:7, 1:7] = 1
        self.register_buffer('active_tokens', active.flatten(), persistent=True)
        self.region = nn.Embedding(2, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 1)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, images):
        return self.forward_features(self.backbone(images))

    def forward_features(self, features):
        f1, f2, f3, z = features
        a = self.level_norms[0](self.early1(f1))
        b = self.level_norms[1](self.early2(f2))
        c = self.level_norms[2](self.early3(f3))
        latent = self.level_norms[3](self.latent(z.flatten(2).transpose(1, 2)))
        x = self.fusion(torch.cat((a, b, c, latent, a - latent, b - latent, c - latent), dim=-1))
        x = x + self.region(self.active_tokens)[None]
        x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), dim=1) + self.position
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpoint_blocks else block(x)
        pooled = (x[:, 0] + x[:, 1:][:, self.active_tokens.bool()].mean(dim=1)) * 0.5
        return self.classifier(self.norm(pooled)).squeeze(-1)

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad))


LargeDetector = Detector
