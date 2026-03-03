import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from models.dinov3_utils import extract_dino_tokens_2d
# from ar.anomaly_maps import compute_anomaly_maps_2d

def compute_anomaly_maps_2d(feats_2d, preds):
    """
    Returns:
        anomaly_maps: [B, H_tok, W_tok]  (per-location reconstruction error)
    """

    # per-location MSE across channels
    mse = (preds - feats_2d).pow(2).mean(dim=1)                  # [B, H, W]

    return mse  # anomaly_maps

def visualize_anomaly_grid(
    dino_model,
    ar_model,
    imgs_vis,        # [B_vis, 3, H, W] on CPU
    labels_vis,      # [B_vis, 1, H, W] on CPU, values {0,1}
    device,
    img_size,
    output_dir,
    epoch=None,
):
    """
    Create a 5×N grid:

      Row 0: normal (healthy) images
      Row 1: predicted anomaly maps for normals
      Row 2: anomalous images
      Row 3: predicted anomaly maps for anomalies
      Row 4: GT anomaly masks (labels) for anomalies

    Up to 10 normals and 10 anomalous are shown (if available).
    """

    dino_model.eval()
    ar_model.eval()

    # Move images to device to compute anomaly maps
    imgs_vis_dev = imgs_vis.to(device)

    with torch.no_grad():
        
        # 1) anomaly maps at token resolution
        feats_2d = extract_dino_tokens_2d(dino_model, imgs_vis_dev, device)  # [B, C, H, W]
        preds = ar_model(feats_2d)  # [B, C, H, W]

        anomaly_maps = compute_anomaly_maps_2d(feats_2d, preds)  # [B, H_tok, W_tok]

        # 2) upsample to image resolution
        B = imgs_vis_dev.size(0)
        anomaly_maps_up = F.interpolate(
            anomaly_maps.unsqueeze(1),           # [B,1,H_tok,W_tok]
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                             # [B, img_size, img_size]

    # Bring everything to CPU for plotting
    imgs_vis_cpu    = imgs_vis_dev.detach().cpu()
    maps_vis_cpu    = anomaly_maps_up.detach().cpu()
    labels_vis_cpu  = labels_vis.detach().cpu()  # [B,1,H,W] with 0/1

    maps_np = maps_vis_cpu.numpy()  # [B,H,W]

    global_min = maps_np.min()
    global_max = maps_np.max()
    den = (global_max - global_min) + 1e-8
    # ------------------------------------------------------------------
    # Split into normal vs anomalous based on label mask
    # (if any pixel is 1 → anomalous)
    # ------------------------------------------------------------------
    # [B]
    is_anom = (labels_vis_cpu.view(B, -1).sum(dim=1) > 0)

    normal_idxs = torch.where(~is_anom)[0]
    anom_idxs   = torch.where(is_anom)[0]

    n_norm = min(10, normal_idxs.numel())
    n_anom = min(10, anom_idxs.numel())

    if n_norm == 0:
        print("[Vis] Warning: no normal samples in visualization batch.")
    if n_anom == 0:
        print("[Vis] Warning: no anomalous samples in visualization batch.")

    normal_idxs = normal_idxs[:n_norm]
    anom_idxs   = anom_idxs[:n_anom]

    n_cols = max(n_norm, n_anom, 1)  # at least 1 to avoid matplotlib issues

    # ------------------------------------------------------------------
    # Build 5×N grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, n_cols, figsize=(2 * n_cols, 10))

    # Handle the case n_cols == 1 (axes is 1D per row)
    def get_ax(row, col):
        if n_cols == 1:
            return axes[row]
        else:
            return axes[row, col]

    # -------------------------
    # 1) Normal samples (rows 0–1)
    # -------------------------
    for j in range(n_cols):
        ax_img = get_ax(0, j)
        ax_map = get_ax(1, j)

        if j < n_norm:
            idx = normal_idxs[j].item()

            # --- original image (denormalized) ---
            img_denorm = denormalize_img(imgs_vis_cpu[idx])   # [3,H,W]
            img_np = img_denorm.permute(1, 2, 0).numpy()      # [H,W,3]
            ax_img.imshow(img_np)
            ax_img.set_title("Normal Img")
            ax_img.axis("off")

            # --- anomaly map ---
            amap = maps_vis_cpu[idx].numpy()                  # [H,W]
            
            amap_vis = (amap - global_min) / den
            amap_vis = np.clip(amap_vis, 0.0, 1.0)
            
            ax_map.imshow(amap_vis, cmap="hot", vmin=0.0, vmax=1.0)
            ax_map.set_title("Anomaly Map")
            ax_map.axis("off")
        else:
            ax_img.axis("off")
            ax_map.axis("off")

    # -------------------------
    # 2) Anomalous samples (rows 2–4)
    # -------------------------
    for j in range(n_cols):
        ax_img  = get_ax(2, j)
        ax_map  = get_ax(3, j)
        ax_mask = get_ax(4, j)

        if j < n_anom:
            idx = anom_idxs[j].item()

            # --- anomalous image ---
            img_denorm = denormalize_img(imgs_vis_cpu[idx])   # [3,H,W]
            img_np = img_denorm.permute(1, 2, 0).numpy()
            ax_img.imshow(img_np)
            ax_img.set_title("Anomaly Img")
            ax_img.axis("off")

            # --- anomaly map (prediction) ---
            amap = maps_vis_cpu[idx].numpy()                  # [H,W]
            amap_vis = (amap - global_min) / den
            amap_vis = np.clip(amap_vis, 0.0, 1.0)
            
            ax_map.imshow(amap_vis, cmap="hot", vmin=0.0, vmax=1.0)
            ax_map.set_title("Anomaly Map")
            ax_map.axis("off")

            # --- GT anomaly mask ---
            gt_mask = labels_vis_cpu[idx, 0].numpy()          # [H,W], 0/1
            ax_mask.imshow(gt_mask, cmap="gray", vmin=0.0, vmax=1.0)
            ax_mask.set_title("GT Mask")
            ax_mask.axis("off")
        else:
            ax_img.axis("off")
            ax_map.axis("off")
            ax_mask.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    if epoch is None:
        fname = "anomaly_grid.png"
    else:
        fname = f"anomaly_grid_epoch_{epoch:04}.png"

    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[Visualization] Saved anomaly grid to: {out_path}")

    return out_path

def denormalize_img(img_tensor):
    """
    img_tensor: [3, H, W] normalized with ImageNet stats.
    Returns: [3, H, W] in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1)

def select_visualization_subset_from_loader(
    dataloader_valid,
    n_anom: int = 10,
    n_norm: int = 10,
    max_batches: int | None = None,
):
    """
    Iterate over `dataloader_valid` until we collect:
      - `n_anom` samples with target == 1
      - `n_norm` samples with target == 0

    Stops early when enough are collected. If the dataloader ends first,
    returns as many as available.

    Returns:
        imgs_vis:   [B_vis, 3, H, W]
        labels_vis: [B_vis, 1, H, W]
    """
    anom_imgs, anom_labels = [], []
    norm_imgs, norm_labels = [], []

    found_anom = 0
    found_norm = 0

    for b_idx, batch in enumerate(dataloader_valid):
        if max_batches is not None and b_idx >= max_batches:
            break

        imgs, labels, targets, meta = batch  # CPU by default
        targets = targets.view(-1)

        # Take anomalies from this batch (up to remaining needed)
        if found_anom < n_anom:
            anom_mask = (targets == 1)
            if anom_mask.any():
                anom_idx = anom_mask.nonzero(as_tuple=True)[0]
                take = min(n_anom - found_anom, anom_idx.numel())
                sel = anom_idx[:take]
                anom_imgs.append(imgs[sel])
                anom_labels.append(labels[sel])
                found_anom += take

        # Take normals from this batch (up to remaining needed)
        if found_norm < n_norm:
            norm_mask = (targets == 0)
            if norm_mask.any():
                norm_idx = norm_mask.nonzero(as_tuple=True)[0]
                take = min(n_norm - found_norm, norm_idx.numel())
                sel = norm_idx[:take]
                norm_imgs.append(imgs[sel])
                norm_labels.append(labels[sel])
                found_norm += take

        # Stop if we have enough of both
        if found_anom >= n_anom and found_norm >= n_norm:
            break

    if found_anom == 0:
        print("[Init] Warning: no anomalous samples found in the dataloader.")
    if found_norm == 0:
        print("[Init] Warning: no normal samples found in the dataloader.")

    # Concatenate and enforce ordering: anomalies first, then normals
    imgs_parts = []
    labels_parts = []

    if len(anom_imgs) > 0:
        imgs_parts.append(torch.cat(anom_imgs, dim=0))
        labels_parts.append(torch.cat(anom_labels, dim=0))
    if len(norm_imgs) > 0:
        imgs_parts.append(torch.cat(norm_imgs, dim=0))
        labels_parts.append(torch.cat(norm_labels, dim=0))

    if len(imgs_parts) == 0:
        # Total fallback: return empty tensors with correct dims if possible
        print("[Init] No samples collected at all.")
        return torch.empty(0), torch.empty(0)

    imgs_vis = torch.cat(imgs_parts, dim=0)
    labels_vis = torch.cat(labels_parts, dim=0)

    print(
        f"[Init] Visualization set: {found_anom}/{n_anom} anomalous + "
        f"{found_norm}/{n_norm} normal = {imgs_vis.size(0)} total "
        f"(scanned {b_idx + 1} batches)."
    )

    return imgs_vis, labels_vis