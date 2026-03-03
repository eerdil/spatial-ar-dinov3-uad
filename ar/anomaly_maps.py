import torch
import torch.nn.functional as F
from models.dinov3_utils import extract_dino_tokens_2d

# def compute_anomaly_maps_2d(dino_model, ar_model, imgs, device):
#     """
#     imgs: [B, 3, H, W]
#     Returns:
#         anomaly_maps: [B, H_tok, W_tok]  (per-location reconstruction error)
#     """
#     dino_model.eval()
#     ar_model.eval()

#     with torch.no_grad():
#         feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]
#         preds = ar_model(feats_2d)                                   # [B, C, H, W]

#         # per-location MSE across channels
#         mse = (preds - feats_2d).pow(2).mean(dim=1)                  # [B, H, W]

#     return mse  # anomaly_maps

def compute_anomaly_maps_2d_bidirectional(dino_model, ar_model_fwd, ar_model_bwd, imgs, device, reduce="mean"):
    """
    imgs: [B, 3, H, W]
    Returns:
        anomaly_maps: [B, H_tok, W_tok]
    Combines forward and backward reconstruction errors.

    reduce: 'mean' or 'min'
      - 'mean': average MSE from both directions
      - 'min' : per-location min(MSE_fwd, MSE_bwd)
    """
    dino_model.eval()
    ar_model_fwd.eval()
    ar_model_bwd.eval()

    with torch.no_grad():
        feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)   # [B, C, H, W]

        # ---------- forward direction ----------
        preds_fwd = ar_model_fwd(feats_2d)                            # [B, C, H, W]
        mse_fwd = (preds_fwd - feats_2d).pow(2).mean(dim=1)           # [B, H, W]

        # ---------- backward direction ----------
        feats_flip = torch.flip(feats_2d, dims=[2, 3])                # [B, C, H, W]
        preds_flip = ar_model_bwd(feats_flip)                         # [B, C, H, W]
        preds_bwd = torch.flip(preds_flip, dims=[2, 3])               # [B, C, H, W]
        mse_bwd = (preds_bwd - feats_2d).pow(2).mean(dim=1)           # [B, H, W]

        if reduce == "mean":
            mse = 0.5 * (mse_fwd + mse_bwd)
        elif reduce == "min":
            mse = torch.min(mse_fwd, mse_bwd)
        else:
            raise ValueError(f"Unknown reduce='{reduce}', use 'mean' or 'min'.")

    return mse

def upsample_anomaly_map(amap, img_size):
    # amap: [1, H_tok, W_tok] or [B, H_tok, W_tok]
    if amap.dim() == 3:
        amap = amap.unsqueeze(1)  # [B, 1, H_tok, W_tok]
    else:
        amap = amap.unsqueeze(0).unsqueeze(0)
    amap_up = F.interpolate(amap, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return amap_up.squeeze()