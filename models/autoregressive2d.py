import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Causal 2D convolution (PixelCNN-style).

    mask_type:
        'A': blocks the center pixel — used as the first layer so the model
             cannot trivially copy its input.
        'B': allows the center pixel but blocks all future pixels in raster-scan
             order — used for all subsequent layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, mask_type="A", **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in ["A", "B"]
        self.mask_type = mask_type
        self.register_buffer("mask", torch.ones_like(self.weight))
        self._build_mask()

    def _build_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2
        mask = torch.ones_like(self.weight)
        # Zero out all rows after the center row, and all columns to the right
        # of center in the center row (future pixels in raster-scan order).
        mask[:, :, yc + 1:, :] = 0
        mask[:, :, yc, xc + 1:] = 0
        if self.mask_type == "A":
            mask[:, :, yc, xc] = 0  # also block the center pixel itself
        self.mask = mask

    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class CenterMaskedConv2d(nn.Conv2d):
    """
    Center-masked convolution: sees all spatial neighbors but not the center pixel.

    Used as the first layer in bidirectional mode. This gives the model full
    360-degree context around each token without leaking the token it needs to
    predict, avoiding trivial identity solutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        self._build_mask()

    def _build_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2
        mask = torch.ones_like(self.weight)
        mask[:, :, yc, xc] = 0  # block only the center pixel
        self.mask = mask

    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AR2DModelDilated(nn.Module):
    """
    Dilated convolutional model operating on 2D DINO feature maps [B, C, H, W].

    Supports three operating modes, selected via `causal` and `center_masked_first`:

        causal=True,  center_masked_first=False  →  Causal mode (default)
            Autoregressive prediction in raster-scan order. The first layer uses
            a type-A MaskedConv2d (cannot see current pixel); subsequent layers
            use type-B MaskedConv2d (see past but not future pixels).

        causal=False, center_masked_first=True   →  Bidirectional mode
            The first layer uses CenterMaskedConv2d (sees all neighbors but not
            the center token). Remaining layers use standard Conv2d with full
            receptive field. Provides 360-degree spatial context without leaking
            the target token.

        causal=False, center_masked_first=False  →  Standard (non-causal) mode
            All layers are standard Conv2d with no masking. Equivalent to a
            fully-convolutional reconstruction model at the feature-map level.

    Args:
        in_channels: Number of input (and output) feature channels (= DINO dim C).
        hidden_channels: Number of channels in intermediate conv layers.
        n_layers: Total number of conv layers (must be >= 3).
        kernel_size: Convolution kernel size (must be odd for 'same' padding).
        causal: If True, use causal masking (autoregressive / causal mode).
        center_masked_first: If True, use center-masked first layer (bidirectional mode).
        dilation_schedule: List of dilation factors indexed by layer position.
                           The first entry applies to layer 0, subsequent entries
                           to middle layers by index, and the last entry is reused
                           for the final projection layer.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        n_layers=5,
        kernel_size=3,
        causal=True,
        center_masked_first=False,
        dilation_schedule=[4, 4, 4, 4],
    ):
        super().__init__()
        self.causal = causal
        self.center_masked_first = center_masked_first

        if causal and center_masked_first:
            raise ValueError("center_masked_first cannot be True when causal=True.")
        assert kernel_size % 2 == 1, "kernel_size must be odd for 'same' padding."
        assert n_layers >= 3

        layers = []

        # --- First layer ---
        d0 = dilation_schedule[0]
        p0 = (kernel_size // 2) * d0
        if causal:
            layers.append(MaskedConv2d(
                in_channels, hidden_channels,
                kernel_size=kernel_size, padding=p0, dilation=d0, mask_type="A",
            ))
        elif center_masked_first:
            layers.append(CenterMaskedConv2d(
                in_channels, hidden_channels,
                kernel_size=kernel_size, padding=p0, dilation=d0,
            ))
        else:
            layers.append(nn.Conv2d(
                in_channels, hidden_channels,
                kernel_size=kernel_size, padding=p0, dilation=d0,
            ))
        layers.append(nn.ReLU(inplace=True))

        # --- Middle layers ---
        for i in range(1, n_layers - 1):
            d = dilation_schedule[i]
            p = (kernel_size // 2) * d
            if causal:
                layers.append(MaskedConv2d(
                    hidden_channels, hidden_channels,
                    kernel_size=kernel_size, padding=p, dilation=d, mask_type="B",
                ))
            else:
                layers.append(nn.Conv2d(
                    hidden_channels, hidden_channels,
                    kernel_size=kernel_size, padding=p, dilation=d,
                ))
            layers.append(nn.ReLU(inplace=True))

        # --- Last layer: project back to input channel dimension ---
        dL = dilation_schedule[-1]
        pL = (kernel_size // 2) * dL
        if causal:
            layers.append(MaskedConv2d(
                hidden_channels, in_channels,
                kernel_size=kernel_size, padding=pL, dilation=dL, mask_type="B",
            ))
        else:
            layers.append(nn.Conv2d(
                hidden_channels, in_channels,
                kernel_size=kernel_size, padding=pL, dilation=dL,
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
