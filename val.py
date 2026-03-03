import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data import AVAILABLE_DATASETS, get_anomaly_loader, get_train_loader

# from data.brats import get_anomaly_loader
from models.autoregressive2d import AR2DModel, AR2DModelDilated
from models.autoencoder import AE2DModel, SpatialBottleneckAE2D
from train import load_dinov3_models, extract_dino_tokens_2d

from config.paths import PROJECT_PATH, DATASET_PATH, checkpoint_paths
from models.dinov3_utils import load_dinov3_models
from ar.visualization import (
    select_visualization_subset_from_loader,
    visualize_anomaly_grid,
)
from ar.train_loops import evaluate_ar2d_model

# ---------------------------------------------------------------
# Evaluation logic (pixel-level, normal vs anomalous stats)
# ---------------------------------------------------------------
# def evaluate_ar2d_model_pixelwise(
#     dino_model,
#     ar_model,
#     dataloader_test,
#     device,
#     img_size=448,
#     output_dir=None,
#     max_batches=None,
# ):
#     dino_model.eval()
#     ar_model.eval()
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)

#     # Streaming metrics (no giant concatenation)
#     auroc_metric = BinaryAUROC().to(device)
#     aupr_metric = BinaryAveragePrecision().to(device)

#     with torch.no_grad():
#         for i, (imgs, labels, targets, meta) in enumerate(tqdm(dataloader_test, desc="Evaluating")):
#             # imgs: [B,3,H,W], labels: [B,1,H,W], targets: [B] (0=normal,1=anomalous)
#             imgs = imgs.to(device)
#             labels = labels.to(device)          # pixel GT mask (0/1)
#             targets = targets.to(device)        # image-level 0/1 (not needed for pixel metrics)

#             # 1) anomaly maps at token level
#             anomaly_maps = compute_anomaly_maps_2d(dino_model, ar_model, imgs, device)  # [B, H_tok, W_tok]

#             # 2) upsample to image resolution and add channel dim: [B,1,H,W]
#             anomaly_maps_up = F.interpolate(
#                 anomaly_maps.unsqueeze(1),      # [B,1,H_tok,W_tok]
#                 # size=(img_size, img_size),
#                 size=(240, 240),
#                 mode="bilinear",
#                 align_corners=False,
#             )                                   # [B,1,H,W]

#             # ensure label map matches upsampled prediction size
#             if labels.shape[-2:] != anomaly_maps_up.shape[-2:]:
#                 labels = F.interpolate(labels, size=anomaly_maps_up.shape[-2:], mode="nearest")

#             # 3) flatten per-batch for TorchMetrics
#             preds_batch = anomaly_maps_up.view(-1)        # [B*H*W]
#             labels_batch = labels.view(-1).int()         # [B*H*W]

#             # 4) update streaming metrics
#             auroc_metric.update(preds_batch, labels_batch)
#             aupr_metric.update(preds_batch, labels_batch)

#             if max_batches is not None and (i + 1) >= max_batches:
#                 break

#     # -----------------------------------------------------------
#     # Compute final metrics
#     # -----------------------------------------------------------
#     auroc = auroc_metric.compute().item()
#     aupr = aupr_metric.compute().item()

#     print(f"Pixel-wise Test AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")

#     # Save metrics
#     output_metrics_path = os.path.join(output_dir, "metrics/")
#     os.makedirs(output_metrics_path, exist_ok=True)
#     with open(os.path.join(output_metrics_path, "test_metrics_pixelwise.txt"), "w") as f:
#         f.write(f"AUROC_pixel: {auroc:.4f}\n")
#         f.write(f"AUPR_pixel:  {aupr:.4f}\n")

#     return {"AUROC_pixel": auroc, "AUPR_pixel": aupr}


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def _save_grayscale_png(path, arr2d, vmin=0.0, vmax=1.0):
    """
    Saves a single-channel 2D array as a grayscale PNG.
    """
    plt.imsave(path, arr2d, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------
# Main script
# ---------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate AR2D anomaly model on test set."
    )

    # ARGUMENT PARSING
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vits16",
        choices=checkpoint_paths.keys(),
        help="Type of DINOv3 model to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=AVAILABLE_DATASETS[0],
        choices=AVAILABLE_DATASETS,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--img_size", type=int, default=448, help="Size to resize images to."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--non_causal",
        action="store_true",
        help="Use non-causal convs (see past and future pixels).",
    )
    parser.add_argument(
        "--center_masked_first",
        action="store_true",
        help="Use center-masked convs (see all neighbors except center pixel).",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for AR model convolution.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (must match a folder in results/ directory)",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="model_best_val_loss.pth",
        help="Name of the checkpoint file to load (model_best_val_loss.pth, model_best_val_aupr.pth, model_best_test_aupr.pth)",
    )
    parser.add_argument(
        "--dilation_schedule",
        type=int,
        nargs="+",
        default=[2, 4, 4, 8],
        help="Layer sizes, e.g. --layers 2 4 4 8",
    )

    args = parser.parse_args()

    model_type = args.model
    dataset_name = args.dataset_name
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    causal = not args.non_causal
    kernel_size = args.kernel_size
    center_masked_first = args.center_masked_first
    experiment_name = args.experiment_name
    ckpt_name = args.ckpt_name
    dilation_schedule = args.dilation_schedule

    output_path = os.path.join(PROJECT_PATH, f"results/{experiment_name}")
    checkpoint_path = os.path.join(output_path, f"ckpt/{ckpt_name}")

    # ----------------------------
    # Load DINO and AR2D model
    # ----------------------------
    print(f"Loading DINOv3 model: {args.model}")
    dino = load_dinov3_models(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    print(f"Loading AR2D checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    img_size = checkpoint.get("img_size", args.img_size)

    # ----------------------------
    # Load test dataset (mixed good + bad)
    # ----------------------------
    dataloader_test = get_anomaly_loader(
        dataset_name,
        dataset_root=DATASET_PATH[dataset_name],
        batch_size=batch_size,
        split_subdir="valid",
        shuffle=False,
        img_size=img_size,
        num_workers=num_workers,
    )

    # Determine feature dim from the data loader
    dummy_imgs, _, _, _ = next(iter(dataloader_test))
    dummy_feats = extract_dino_tokens_2d(dino, dummy_imgs.to(device), device)
    C = dummy_feats.shape[1]

    if causal is False and center_masked_first is False:
        print("Using Autoencoder-style 2D model (non-causal, standard convs).")
        # ar_model = AE2DModel(
        #     in_channels=C,
        #     hidden_channels=256,
        #     latent_channels=64,
        #     n_layers=5,
        #     kernel_size=kernel_size,
        # ).to(device)
        ar_model = SpatialBottleneckAE2D(
            in_channels=C,
            hidden_channels=256,
            bottleneck_channels=128,
            n_layers=5,
            kernel_size=kernel_size,
        ).to(device)
    else:
        print("Using Convolutional AR2D model.")
        # ar_model = AR2DModel(
        #     in_channels=C,
        #     hidden_channels=256,
        #     n_layers=5,
        #     kernel_size=kernel_size,
        #     causal=causal,
        #     center_masked_first=center_masked_first,
        # ).to(device)
        ar_model = AR2DModelDilated(
            in_channels=C,
            hidden_channels=256,
            n_layers=5,
            kernel_size=kernel_size,
            causal=causal,
            center_masked_first=center_masked_first,
            dilation_schedule=dilation_schedule,
        ).to(device)

    ar_model.load_state_dict(checkpoint["model_state_dict"])

    # ----------------------------
    # Define criterion
    # ----------------------------
    criterion = nn.MSELoss()

    # ----------------------------
    # Evaluate (pixel-wise)
    # ----------------------------
    # _, metrics, _ = evaluate_ar2d_model(
    #     dino_model=dino,
    #     ar_model=ar_model,
    #     dataloader=dataloader_test,
    #     imgs_vis=None,
    #     labels_vis=None,
    #     criterion=criterion,
    #     device=device,
    #     output_path=os.path.join(output_path, "test/"),
    # )
    rotate_by = 0
    if dataset_name in ["brats"]:
        rotate_by = 3

    global_min = float("inf")
    global_max = float("-inf")
    n_seen = 0
    with torch.no_grad():
        for imgs, labels, targets, meta in dataloader_test:
            imgs = imgs.to(device)
            labels = labels.to(device)  # pixel GT mask (0/1)

            targets = targets.to(
                device
            )  # image-level 0/1 (not needed for pixel metrics)

            feats_2d = extract_dino_tokens_2d(dino, imgs, device)  # [B, C, H, W]
            preds = ar_model(feats_2d)  # [B, C, H, W]

            anomaly_maps = (preds - feats_2d).pow(2).mean(dim=1)

            # 2) upsample to image resolution and add channel dim: [B,1,H,W]
            anomaly_maps_up = F.interpolate(
                anomaly_maps.unsqueeze(1),  # [B,1,H_tok,W_tok]
                size=(240, 240),
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                1
            )  # [B,H,W]

            batch_min = anomaly_maps_up.amin().item()
            batch_max = anomaly_maps_up.amax().item()
            global_min = min(global_min, batch_min)
            global_max = max(global_max, batch_max)

            n_seen += imgs.size(0)

    eps = 1e-12
    denom = max(global_max - global_min, eps)

    print(
        f"[VAL] Seen {n_seen} images. Global min={global_min:.6g}, max={global_max:.6g}"
    )

    imgs_path = os.path.join(output_path, "val/images")
    os.makedirs(imgs_path, exist_ok=True)
    # -------------------------
    # PASS 2: save per-sample images + GT + normalized prediction maps
    # -------------------------
    i_img = 0
    for b, (imgs, labels, targets, meta) in enumerate(dataloader_test):

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats_2d = extract_dino_tokens_2d(dino, imgs, device)  # [B,C,H,W]
        preds = ar_model(feats_2d)  # [B,C,H,W]

        anomaly_maps = (preds - feats_2d).pow(2).mean(dim=1)  # [B,Htok,Wtok]
        # anomaly_maps_up = F.interpolate(
        #     anomaly_maps.unsqueeze(1),
        #     size=(240, 240),
        #     mode="bilinear",
        #     align_corners=False,
        # )  # [B,1,H,W]

        # imgs = F.interpolate(
        #     imgs,
        #     size=(240, 240),
        #     mode="bilinear",
        #     align_corners=False,
        # )  # [B,1,H,W]

        # labels = F.interpolate(
        #     labels,
        #     size=(240, 240),
        #     mode="nearest",
        # )  # [B,1,H,W]

        # Normalize with global min/max (like your colleague)
        anomaly_norm = (anomaly_maps - global_min) / denom
        anomaly_norm = anomaly_norm.clamp(0.0, 1.0)  # [B,1,H,W]

        imgs_np = _to_numpy(imgs)  # typically [B,3,H,W] or [B,1,H,W]
        labels_np = _to_numpy(labels)  # [B,1,H,W] or [B,H,W]
        pred_raw_np = _to_numpy(anomaly_maps)  # [B,1,H,W]
        pred_norm_np = _to_numpy(anomaly_norm)  # [B,1,H,W]

        B = imgs_np.shape[0]
        for k in range(B):
            # --- pick a name ---
            # Prefer something stable from meta if you have it; fallback to running index.
            # Common patterns: meta["name"], meta["img_id"], meta["path"], etc.
            if isinstance(meta, dict):
                name = None
                for key in ["name", "id", "img_id", "filename", "path"]:
                    if key in meta:
                        v = meta[key]
                        # v might be list of strings length B
                        if isinstance(v, (list, tuple)) and len(v) == B:
                            name = os.path.splitext(os.path.basename(str(v[k])))[0]
                        else:
                            name = f"{key}_{i_img:04d}"
                        break
                if name is None:
                    name = f"{i_img:04d}"
            else:
                name = f"{i_img:04d}"

            # --- image to save (grayscale) ---
            # If RGB: use first 3 channels; if 1-ch: squeeze
            im = imgs_np[k]
            if im.ndim == 3 and im.shape[0] in (1, 3):
                # CHW -> HWC
                if im.shape[0] == 3:
                    im_hwc = np.transpose(im, (1, 2, 0))
                    # optional: if your imgs are normalized, you may want to denorm here.
                    # We'll just min-max for visualization.
                    im_min, im_max = im_hwc.min(), im_hwc.max()
                    if im_max > im_min:
                        im_hwc = (im_hwc - im_min) / (im_max - im_min)
                    img_to_save = im_hwc
                else:
                    im2d = im[0]
                    im_min, im_max = im2d.min(), im2d.max()
                    if im_max > im_min:
                        im2d = (im2d - im_min) / (im_max - im_min)
                    img_to_save = im2d
            else:
                # fallback
                img_to_save = im.squeeze()

            # --- GT mask ---
            gt = labels_np[k]
            if gt.ndim == 3 and gt.shape[0] == 1:
                gt2d = gt[0]
            else:
                gt2d = gt.squeeze()

            # --- Pred maps rotate ---
            pred_norm_2d = pred_norm_np[k]  # [H,W]
            pred_raw_2d = pred_raw_np[k]  # [H,W]

            file_name = meta["img_path"][k].split("/")[-1]
            subfolder_name = meta["class_name"][k]
            if rotate_by != 0:
                pred_norm_2d = np.rot90(pred_norm_2d, rotate_by)
                pred_raw_2d = np.rot90(pred_raw_2d, rotate_by)
                # If your GT/images also need rotation to match, rotate them too:
                gt2d = np.rot90(gt2d, rotate_by)
                if img_to_save.ndim == 2:
                    img_to_save = np.rot90(img_to_save, rotate_by)
                elif img_to_save.ndim == 3:
                    img_to_save = np.rot90(img_to_save, rotate_by)

            # Save image
            os.makedirs(os.path.join(imgs_path, subfolder_name, "img"), exist_ok=True)
            img_path = os.path.join(imgs_path, subfolder_name, "img", f"{file_name}")
            if isinstance(img_to_save, np.ndarray) and img_to_save.ndim == 3:
                plt.imsave(img_path, img_to_save)  # RGB
            else:
                _save_grayscale_png(img_path, img_to_save, vmin=0.0, vmax=1.0)

            # Save GT (binary) as 0..1
            os.makedirs(os.path.join(imgs_path, subfolder_name, "gt"), exist_ok=True)
            gt_path = os.path.join(imgs_path, subfolder_name, "gt", f"{file_name}")
            gt_bin = (gt2d > 0.5).astype(np.float32)
            _save_grayscale_png(gt_path, gt_bin, vmin=0.0, vmax=1.0)

            # Save normalized prediction map (your colleague style)
            os.makedirs(os.path.join(imgs_path, subfolder_name, "pred"), exist_ok=True)
            pred_path = os.path.join(imgs_path, subfolder_name, "pred", f"{file_name}")
            _save_grayscale_png(
                pred_path, pred_norm_2d.astype(np.float32), vmin=0.0, vmax=1.0
            )

            # Optionally save raw map (float) for metrics
            # if save_raw_npy:
            #     np.save(
            #         os.path.join(raw_dir, f"anomaly_raw_{name}.npy"),
            #         pred_raw_2d.astype(np.float32),
            #     )

            i_img += 1

    print(f"[VAL] Saved {i_img} samples to: {imgs_path}")
