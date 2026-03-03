import torch.nn as nn
import torch
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Causal 2D conv à la PixelCNN.
    mask_type:
      'A' = cannot see current pixel (strict autoregressive for first layer)
      'B' = can see current pixel but not future ones (for later layers)
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

        # Zero out "future" positions
        mask[:, :, yc + 1 :, :] = 0
        mask[:, :, yc, xc + 1 :] = 0

        if self.mask_type == "A":
            # Also block the current position (yc, xc)
            mask[:, :, yc, xc] = 0

        self.mask = mask

    # def forward(self, x):
    #     # apply mask to weights
    #     self.weight.data *= self.mask
    #     return super().forward(x)
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class CenterMaskedConv2d(nn.Conv2d):
    """
    Center-masked conv: sees all neighbors except the center pixel.
    Used for the first layer in non-causal 'context' mode for anomaly detection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.register_buffer("mask", torch.ones_like(self.weight))
        self._build_mask()

    def _build_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2

        mask = torch.ones_like(self.weight)
        # Zero out only the center location
        mask[:, :, yc, xc] = 0
        self.mask = mask

    # def forward(self, x):
    #     self.weight.data *= self.mask
    #     return super().forward(x)
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class AR2DModel(nn.Module):
    """
    2D autoregressive / reconstruction model over DINO feature maps.

    If causal=True:
        uses MaskedConv2d (AR / PixelCNN-style)
    If causal=False:
        uses standard Conv2d (sees both past and future pixels)

    Input : [B, C, H, W]
    Output: [B, C, H, W]  (prediction of features at each location)
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        n_layers=5,
        kernel_size=3,
        causal=True,
        center_masked_first=False,
    ):
        super().__init__()
        self.causal = causal
        self.center_masked_first = center_masked_first

        if self.causal and self.center_masked_first:
            raise ValueError("center_masked_first cannot be True when causal=True.")

        padding = kernel_size // 2

        layers = []

        # First layer
        if causal:
            layers.append(
                MaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    mask_type="A",
                )
            )
        elif self.center_masked_first:
            layers.append(
                CenterMaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(n_layers - 2):
            if causal:
                layers.append(
                    MaskedConv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        mask_type="B",
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        if causal:
            layers.append(
                MaskedConv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    mask_type="B",
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Bottleneck version
# ------------------------------


class AR2DModelBottleneck(nn.Module):
    """
    PixelCNN-style masked conv stack with an optional *channel bottleneck* inserted mid-network.
    No spatial downsampling.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        n_layers: int = 5,
        kernel_size: int = 3,
        bottleneck_channels: int | None = None,  # e.g., 64; None disables
        bottleneck_layers: int = 2,  # number of layers spent in bottleneck width (must be >= 1 if enabled)
    ):
        super().__init__()
        assert n_layers >= 3, "Need at least 3 layers: first(A), middle(B...), last(B)."
        padding = kernel_size // 2

        layers = []

        # 1) First layer: mask A
        layers.append(
            MaskedConv2d(
                in_channels,
                hidden_channels,
                kernel_size,
                padding=padding,
                mask_type="A",
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # How many masked-B conv layers remain between first and last?
        # Total convs = n_layers. We've used 1 (first) and will use 1 (last), so middle_convs = n_layers - 2.
        middle_convs = n_layers - 2

        if bottleneck_channels is None:
            # Simple: all middle layers are hidden->hidden masked-B
            for _ in range(middle_convs - 1):
                layers.append(
                    MaskedConv2d(
                        hidden_channels, hidden_channels, 3, padding=1, mask_type="B"
                    )
                )
                layers.append(nn.ReLU(inplace=True))

            # Last layer: hidden -> in_channels (masked-B)
            layers.append(
                MaskedConv2d(hidden_channels, in_channels, 3, padding=1, mask_type="B")
            )
        else:
            assert bottleneck_layers >= 1
            # We will insert:
            #   hidden -> bottleneck (B)
            #   (bottleneck_layers - 1) times: bottleneck -> bottleneck (B)
            #   bottleneck -> hidden (B)
            #
            # That's (bottleneck_layers + 2) convs. These must fit within the middle_convs.
            needed = bottleneck_layers + 2
            assert needed <= middle_convs, (
                f"Not enough layers for bottleneck. Need at least {needed} middle convs, "
                f"but have {middle_convs}. Increase n_layers or reduce bottleneck_layers."
            )

            # Number of "plain" hidden->hidden middle convs before bottleneck
            pre = (middle_convs - needed) // 2
            post = (middle_convs - needed) - pre

            # pre hidden->hidden
            for _ in range(pre):
                layers.append(
                    MaskedConv2d(
                        hidden_channels, hidden_channels, 3, padding=1, mask_type="B"
                    )
                )
                layers.append(nn.ReLU(inplace=True))

            # down-proj into bottleneck
            layers.append(
                MaskedConv2d(
                    hidden_channels, bottleneck_channels, 3, padding=1, mask_type="B"
                )
            )
            layers.append(nn.ReLU(inplace=True))

            # bottleneck stack
            for _ in range(bottleneck_layers - 1):
                layers.append(
                    MaskedConv2d(
                        bottleneck_channels,
                        bottleneck_channels,
                        3,
                        padding=1,
                        mask_type="B",
                    )
                )
                layers.append(nn.ReLU(inplace=True))

            # up-proj back to hidden
            layers.append(
                MaskedConv2d(
                    bottleneck_channels, hidden_channels, 3, padding=1, mask_type="B"
                )
            )
            layers.append(nn.ReLU(inplace=True))

            # post hidden->hidden
            for _ in range(post):
                layers.append(
                    MaskedConv2d(
                        hidden_channels, hidden_channels, 3, padding=1, mask_type="B"
                    )
                )
                layers.append(nn.ReLU(inplace=True))

            # last layer: hidden -> in_channels
            layers.append(
                MaskedConv2d(hidden_channels, in_channels, 3, padding=1, mask_type="B")
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Dilated version
# ------------------------------
class AR2DModelDilated(nn.Module):
    """
    AR2DModel with optional dilation schedule for the masked conv layers.
    Uses same kernel_size everywhere. For kernel=3, uses padding=dilation to keep H,W same.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        n_layers=5,
        kernel_size=3,
        causal=True,
        center_masked_first=False,
        dilation_schedule=[2, 2, 4, 4, 4],
        # dilation_schedule=[2, 1, 1, 1],        # e.g., [1,2,4,8]
        use_dilation_in_first=True,  # keep first at dilation=1 by default
    ):
        super().__init__()
        self.causal = causal
        self.center_masked_first = center_masked_first

        if self.causal and self.center_masked_first:
            raise ValueError("center_masked_first cannot be True when causal=True.")
        assert (
            kernel_size % 2 == 1
        ), "Use odd kernel_size for 'same' padding with dilation."

        # Number of conv layers: first + (n_layers-2 middle) + last
        assert n_layers >= 3

        # Build dilation list for middle layers (and optionally first/last)
        # We will apply dilation to middle layers by default, and keep last also dilated.
        middle_len = (
            n_layers - 2
        )  # number of convs between first and last? (actually middle conv count)
        # In your original, middle layers are (n_layers-2) convs? No: you had first + (n_layers-2) middle? + last.
        # Your original loop is range(n_layers - 2) then last => total convs = 1 + (n_layers-2) + 1 = n_layers
        # So "middle conv count" = n_layers - 2
        # middle_convs = n_layers - 2

        # if dilation_schedule is None:
        #     dilation_schedule = [1]
        # assert len(dilation_schedule) >= 1

        # # Repeat/truncate schedule to match middle conv count
        # dilations_mid = (
        #     dilation_schedule
        #     * ((middle_convs + len(dilation_schedule) - 1) // len(dilation_schedule))
        # )[:middle_convs]

        layers = []

        # First layer
        # d0 = dilations_mid[0] if use_dilation_in_first else 1
        d0 = dilation_schedule[0]
        p0 = (kernel_size // 2) * d0

        if causal:
            layers.append(
                MaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=p0,
                    dilation=d0,
                    mask_type="A",
                )
            )
        elif center_masked_first:
            layers.append(
                CenterMaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=p0,
                    dilation=d0,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=p0,
                    dilation=d0,
                )
            )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers (masked-B or normal conv)
        for i in range(1, n_layers - 1):
            d = dilation_schedule[i]
            p = (kernel_size // 2) * d
            if causal:
                layers.append(
                    MaskedConv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=p,
                        dilation=d,
                        mask_type="B",
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=p,
                        dilation=d,
                    )
                )
            layers.append(nn.ReLU(inplace=True))

        # Last layer: use last dilation from schedule (or 1)
        # dL = dilations_mid[-1] if len(dilations_mid) > 0 else 1
        dL = dilation_schedule[-1]
        pL = (kernel_size // 2) * dL
        if causal:
            layers.append(
                MaskedConv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=pL,
                    dilation=dL,
                    mask_type="B",
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=pL,
                    dilation=dL,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
