import torch
import torch.nn as nn


class AE2DModel(nn.Module):
    """
    Autoencoder-style 2D model over feature maps (no spatial downsampling),
    designed to have the *same depth/width* style as your AR2DModel.

    Total number of Conv2d layers == n_layers (same as AR2DModel).
    Structure (for n_layers=5):
        in -> hidden -> hidden -> latent -> hidden -> in

    The "bottleneck" here is a *channel* bottleneck via latent_channels.
    If you set latent_channels == hidden_channels, you essentially remove the bottleneck
    but keep the same overall capacity profile.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        latent_channels: int = 64,   # set to hidden_channels for "no bottleneck"
        n_layers: int = 5,
        kernel_size: int = 3,
        activation=nn.ReLU,
    ):
        super().__init__()
        assert n_layers >= 3, "Need at least 3 conv layers: in->hidden, bottleneck, hidden->out."

        padding = kernel_size // 2

        # We will allocate:
        # - 1 conv: in -> hidden
        # - (pre_hidden_convs) convs: hidden -> hidden
        # - 1 conv: hidden -> latent   (down-proj)
        # - 1 conv: latent -> hidden   (up-proj)
        # - (post_hidden_convs) convs: hidden -> hidden
        # - 1 conv: hidden -> in
        #
        # Total convs = 1 + pre + 1 + 1 + post + 1 = pre + post + 4
        # So pre + post = n_layers - 4
        #
        # For n_layers=5 => pre+post=1 (we put it before bottleneck by default)
        remaining = n_layers - 4
        pre_hidden_convs = (remaining + 1) // 2   # bias extra to encoder side
        post_hidden_convs = remaining // 2

        layers = []

        # 1) in -> hidden
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding))
        layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 2) hidden -> hidden (pre)
        for _ in range(pre_hidden_convs):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 3) bottleneck down-projection: hidden -> latent
        layers.append(nn.Conv2d(hidden_channels, latent_channels, 3, padding=1))
        layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 4) bottleneck up-projection: latent -> hidden
        layers.append(nn.Conv2d(latent_channels, hidden_channels, 3, padding=1))
        layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 5) hidden -> hidden (post)
        for _ in range(post_hidden_convs):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 6) hidden -> out (in_channels)
        layers.append(nn.Conv2d(hidden_channels, in_channels, 3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    


class SpatialBottleneckAE2D(nn.Module):
    """
    Autoencoder with a *spatial bottleneck* (dimensionality reduction in H,W)
    while keeping a similar conv-stack style to AE2DModel.

    - Total conv-like layers is controlled roughly by n_layers.
    - Bottleneck is introduced by n_down strided convs (each /2 in H,W)
      and n_down transposed convs (each *2 in H,W).

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        bottleneck_channels: int = 256,
        n_layers: int = 7,
        kernel_size: int = 3,
        n_down: int = 1,              # 1 => H,W -> H/2,W/2 ; 2 => H/4,W/4
        activation=nn.ReLU,
    ):
        super().__init__()
        assert n_layers >= (2 * n_down + 3), (
            "n_layers too small for the requested down/up structure. "
            "Need at least 2*n_down + 3."
        )
        assert n_down >= 1

        Act = (lambda: activation(inplace=True)) if activation is nn.ReLU else activation

        pad = kernel_size // 2

        def conv(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)

        def deconv(in_ch, out_ch):
            # 4x4, stride 2, padding 1 is the common "exact x2" upsample for even sizes
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

        layers = []

        # Stem: in -> hidden
        layers += [conv(in_channels, hidden_channels, k=kernel_size, s=1, p=pad), Act()]

        # We will spend some layers in encoder pre-processing
        # and some layers in decoder post-processing, around the down/up ops.
        #
        # Mandatory ops:
        #   - n_down strided convs (downsample)
        #   - n_down transposed convs (upsample)
        #   - final conv hidden->in
        #
        # Convs used so far: 1 (stem) + 1 (final) + n_down (down) + n_down (up) = 2 + 2*n_down
        # Remaining convs: n_layers - (2 + 2*n_down)
        remaining_convs = n_layers - (2 + 2 * n_down)
        # We’ll split remaining convs into:
        #   pre-down convs at hidden resolution,
        #   bottleneck convs at low resolution,
        #   post-up convs at hidden resolution.
        # Simple split: 1/3, 1/3, 1/3 (with rounding)
        pre = remaining_convs // 3
        mid = remaining_convs // 3
        post = remaining_convs - pre - mid

        # Pre-down convs (hidden -> hidden)
        for _ in range(pre):
            layers += [conv(hidden_channels, hidden_channels, k=kernel_size, s=1, p=pad), Act()]

        # Downsampling path
        cur_ch = hidden_channels
        for d in range(n_down):
            out_ch = bottleneck_channels if d == n_down - 1 else cur_ch
            layers += [conv(cur_ch, out_ch, k=kernel_size, s=2, p=pad), Act()]
            cur_ch = out_ch

        # Bottleneck processing at low resolution
        for _ in range(mid):
            layers += [conv(cur_ch, cur_ch, k=kernel_size, s=1, p=pad), Act()]

        # Upsampling path
        for d in range(n_down):
            out_ch = hidden_channels if d == n_down - 1 else cur_ch
            layers += [deconv(cur_ch, out_ch), Act()]
            cur_ch = out_ch

        # Post-up convs (hidden -> hidden)
        for _ in range(post):
            layers += [conv(hidden_channels, hidden_channels, k=kernel_size, s=1, p=pad), Act()]

        # Final: hidden -> in
        layers += [conv(hidden_channels, in_channels, k=kernel_size, s=1, p=pad)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)