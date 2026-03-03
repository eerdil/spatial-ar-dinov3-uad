import torch
import torch.nn as nn
import math

class SinCos2DPositionalEmbedding(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding for HxW grids.
    """
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.embed_dim = embed_dim

    def forward(self, H, W, device):
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        dim_t = torch.arange(self.embed_dim // 4, device=device)
        dim_t = 1.0 / (10000 ** (2 * dim_t / (self.embed_dim // 2)))

        pos_x = xx[..., None] * dim_t
        pos_y = yy[..., None] * dim_t

        pos = torch.cat(
            [
                torch.sin(pos_x), torch.cos(pos_x),
                torch.sin(pos_y), torch.cos(pos_y),
            ],
            dim=-1,
        )

        return pos.view(H * W, self.embed_dim)
    
class CausalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + h

        h = self.norm2(x)
        x = x + self.mlp(h)
        return x
    
class AR2DTransformer(nn.Module):
    """
    Transformer-based AR / context model for 2D feature maps.

    Modes:
    - causal=True                     → strict autoregressive
    - causal=False + mask_radius=0    → center-masked only
    - causal=False + mask_radius=1    → center + 4-neighbors masked
    - causal=False + mask_radius=1.5  → center + 8-neighbors masked
    """

    def __init__(
        self,
        in_channels,
        embed_dim=256,
        depth=5,
        num_heads=8,
        causal=True,
        mask_radius=None,
        dropout=0.0,
    ):
        super().__init__()

        if causal and mask_radius is not None:
            raise ValueError("mask_radius is only valid when causal=False")

        self.causal = causal
        self.mask_radius = mask_radius

        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = SinCos2DPositionalEmbedding(embed_dim)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, in_channels)

    def _build_causal_mask(self, N, device):
        return torch.triu(
            torch.ones(N, N, device=device),
            diagonal=1
        ).bool()
    
    def _build_neighborhood_mask(self, H, W, radius, device):
        """
        Mask tokens within Chebyshev distance <= radius.
        """
        N = H * W
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).view(N, 2)

        diff = coords[:, None, :] - coords[None, :, :]
        dist = diff.abs().max(dim=-1).values

        mask = dist <= radius
        return mask
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W
        device = x.device

        # [B, C, H, W] → [B, N, C]
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)

        # Token embedding + position
        x = self.input_proj(x)
        x = x + self.pos_embed(H, W, device).unsqueeze(0)

        causal_mask = None
        neighborhood_mask = None

        if self.causal:
            causal_mask = self._build_causal_mask(N, device)
        elif self.mask_radius is not None:
            neighborhood_mask = self._build_neighborhood_mask(
                H, W, self.mask_radius, device
            )

        # Transformer blocks
        for i, blk in enumerate(self.blocks):
            if self.causal:
                attn_mask = causal_mask
            elif i == 0 and neighborhood_mask is not None:
                attn_mask = neighborhood_mask
            else:
                attn_mask = None

            x = blk(x, attn_mask)

        x = self.norm(x)
        x = self.output_proj(x)

        # [B, N, C] → [B, C, H, W]
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x