import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data import AVAILABLE_DATASETS, get_anomaly_loader, get_train_loader

from models.autoregressive2d import AR2DModelDilated
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
        "--mode",
        type=str,
        default="causal",
        choices=["causal", "bidirectional", "standard"],
        help=(
            "AR model operating mode. Must match the mode used during training. "
            "'causal': autoregressive prediction in raster-scan order (default). "
            "'bidirectional': center-masked first layer with full 360-degree context. "
            "'standard': standard convolutions with no masking."
        ),
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
        default=[4, 4, 4, 4],
        help="Layer sizes, e.g. --layers 2 4 4 8",
    )

    args = parser.parse_args()

    model_type = args.model
    dataset_name = args.dataset_name
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    mode = args.mode
    causal = (mode == "causal")
    center_masked_first = (mode == "bidirectional")
    kernel_size = args.kernel_size
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

