import argparse
import os
import torch
import random
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from datetime import datetime

from data import AVAILABLE_DATASETS, get_anomaly_loader, get_train_loader
from models.autoregressive2d import AR2DModelDilated
from models.dinov3_utils import load_dinov3_models, extract_dino_tokens_2d
from config.paths import PROJECT_PATH, DATASET_PATH, checkpoint_paths
from ar.train_loops import train_ar2d_model
from ar.visualization import select_visualization_subset_from_loader


def set_seed(seed: int = 42):
    print(f"[Seed] Setting seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # === ARGUMENT PARSING ===
    parser = argparse.ArgumentParser(description="Visualize DINOv3 features.")
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
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs for autoregressive model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for autoregressive model training.",
    )
    parser.add_argument(
        "--val_interval", type=int, default=1, help="Validate every N epochs."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="causal",
        choices=["causal", "bidirectional", "standard"],
        help=(
            "AR model operating mode. "
            "'causal': autoregressive prediction in raster-scan order (default). "
            "'bidirectional': center-masked first layer gives full 360-degree spatial context. "
            "'standard': standard convolutions with no masking (non-causal autoencoder-style)."
        ),
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for AR model convolution.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--random_seed",
        action="store_true",
        help="Use random seed - overwrites --seed.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable logging to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="AR_DinoV3_Anomaly_Detection",
        help="WandB project name (used only if --use_wandb is set).",
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
    epochs = args.epochs
    lr = args.lr
    val_interval = args.val_interval
    mode = args.mode
    causal = (mode == "causal")
    center_masked_first = (mode == "bidirectional")
    kernel_size = args.kernel_size
    seed = args.seed
    random_seed = args.random_seed
    use_wandb = args.use_wandb
    wandb_project_name = args.wandb_project_name
    dilation_schedule = args.dilation_schedule

    if random_seed:
        seed = random.randint(0, 2**32 - 1)

    # === SET SEED ===
    set_seed(seed)

    # === SET EXPERIMENT NAME AND CREATE PATHS TO LOG RESULTS ===
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dilation_schedule_str = "x".join(map(str, dilation_schedule))
    exp_name = f"{model_type}_{dataset_name}_{mode}_ks{kernel_size}_dil{dilation_schedule_str}_seed{seed}_ep{epochs}_img{img_size}_{timestamp}"
    output_path = os.path.join(PROJECT_PATH, f"results/{exp_name}")

    # === INITIALIZE WANDB ===
    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        run = wandb.init(project=wandb_project_name, name=exp_name)
        print("[W&B] Run initialized")
        print("[W&B] Run dir:", run.dir)
        print("[W&B] Run URL:", run.url)

    # === GET TRAIN/TEST/VAL DATA LOADERS ===
    dataloader_train = get_train_loader(
        dataset_name,
        dataset_root=DATASET_PATH[dataset_name],
        batch_size=batch_size,
        shuffle=True,
        img_size=img_size,
        num_workers=num_workers,
    )

    dataloader_valid = get_anomaly_loader(
        dataset_name,
        dataset_root=DATASET_PATH[dataset_name],
        batch_size=batch_size,
        split_subdir="valid",
        shuffle=False,
        img_size=img_size,
        num_workers=num_workers,
    )

    dataloader_test = get_anomaly_loader(
        dataset_name,
        dataset_root=DATASET_PATH[dataset_name],
        batch_size=batch_size,
        split_subdir="test",
        shuffle=False,
        img_size=img_size,
        num_workers=num_workers,
    )

    # === PICK A FIXED SET OF VALIDATION AND TEST IMAGES FOR VISUALIZATION ===
    imgs_vis, labels_vis = select_visualization_subset_from_loader(
        dataloader_valid,
        n_anom=10,
        n_norm=10,
    )

    # === LOAD DINOV3 ===
    dino = load_dinov3_models(model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    # === GET FEATURE DIMENSION C FROM DINOV3 ===
    imgs_batch, _ = next(iter(dataloader_train))
    feats_2d = extract_dino_tokens_2d(dino, imgs_batch.to(device), device)
    C = feats_2d.shape[1]
    print(f"DINO feature channels: {C}")

    # === CREATE MODEL ===
    print(f"Using AR2DModelDilated in '{mode}' mode.")
    ar2d = AR2DModelDilated(
        in_channels=C,
        hidden_channels=256,
        n_layers=5,
        kernel_size=kernel_size,
        causal=causal,
        center_masked_first=center_masked_first,
        dilation_schedule=dilation_schedule,
    ).to(device)

    ar2d = train_ar2d_model(
        dino_model=dino,
        ar_model=ar2d,
        train_loader=dataloader_train,
        val_loader=dataloader_valid,
        test_loader=dataloader_test,
        device=device,
        epochs=epochs,
        lr=lr,
        val_interval=val_interval,
        output_dir=output_path,
        img_size=img_size,
        imgs_vis=imgs_vis,
        labels_vis=labels_vis,
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    main()
