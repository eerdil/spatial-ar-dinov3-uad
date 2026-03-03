import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import tracemalloc
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


# ---------------------------------------------------------------
# Main script
# ---------------------------------------------------------------
if __name__ == "__main__":
    tracemalloc.start()

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
        split_subdir="test",
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
    _, metrics, _ = evaluate_ar2d_model(
        dino_model=dino,
        ar_model=ar_model,
        dataloader=dataloader_test,
        imgs_vis=None,
        labels_vis=None,
        criterion=criterion,
        device=device,
        output_path=os.path.join(output_path, "test/"),
    )

    log_dict = {"test/AUROC": metrics["AUROC"], "test/AUPR": metrics["AUPR"]}

    log_msg = f"".join([f"{k}: {v:.4f}" for k, v in log_dict.items() if k != "epoch"])
    print("=== Final Test Set Metrics ===")
    print("AUROC AUPR Time Loss")
    print(
        f"{metrics['AUROC']:.4f} {metrics['AUPR']:.4f} {metrics['Time']:.4f} {metrics['Loss']:.4f}"
    )

    peak_cpu_memory = tracemalloc.get_traced_memory()[1]
    peak_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"Peak CPU Memory usage: {peak_cpu_memory / 1024 / 1024 / 1024} GB")
    print(f"Peak GPU Memory usage: {peak_gpu_memory / 1024 / 1024 / 1024} GB")
    print(
        f"Final Test Metrics: {peak_cpu_memory / 1024 / 1024 / 1024 + peak_gpu_memory / 1024 / 1024 / 1024}"
    )
