import os
import torch
import torch.nn as nn
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from models.dinov3_utils import extract_dino_tokens_2d
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from ar.visualization import visualize_anomaly_grid
import torch.nn.functional as F
import time


def evaluate_ar2d_model(
    dino_model,
    ar_model,
    dataloader,
    criterion,
    device,
    imgs_vis=None,
    labels_vis=None,
    epoch=None,
    img_size=240,
    output_path=None,
):
    """
    Evaluate the AR model on an anomaly detection dataloader (no-DINO variant).

    Operates directly on raw images instead of DINO feature maps.  Computes MSE
    reconstruction loss, pixel-level AUROC, and AUPR, and saves metrics to disk.

    Returns:
        avg_loss: Mean MSE over all batches.
        metrics: Dict with keys AUROC, AUPR, Time, Loss.
        output_path_visualizations: Always None (visualization not implemented here).
    """
    ar_model.eval()

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    # Streaming metrics (no giant concatenation)
    auroc_metric = BinaryAUROC().to(device)
    aupr_metric = BinaryAveragePrecision().to(device)

    running_loss = 0.0
    n_batches = 0
    total_time = 0.0
    with torch.no_grad():
        for imgs, labels, targets, meta in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)  # pixel GT mask (0/1)
            targets = targets.to(
                device
            )  # image-level 0/1 (not needed for pixel metrics)

            start_time = time.time()
            # feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]
            preds = ar_model(imgs)  # [B, C, H, W]

            end_time = time.time()

            total_time += end_time - start_time

            loss = criterion(preds, imgs)
            running_loss += loss.item()
            n_batches += 1

            # 1) anomaly maps at token level
            anomaly_maps = (preds - imgs).pow(2).mean(dim=1)

            # 2) upsample to image resolution and add channel dim: [B,1,H,W]
            anomaly_maps_up = F.interpolate(
                anomaly_maps.unsqueeze(1),  # [B,1,H_tok,W_tok]
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )

            # ensure label map matches upsampled prediction size
            if labels.shape[-2:] != anomaly_maps_up.shape[-2:]:
                labels = F.interpolate(
                    labels, size=anomaly_maps_up.shape[-2:], mode="nearest"
                )

            # 3) flatten per-batch for TorchMetrics
            preds_batch = anomaly_maps_up.view(-1)  # [B*H*W]
            labels_batch = labels.view(-1).int()  # [B*H*W]

            # 4) update streaming metrics
            auroc_metric.update(preds_batch, labels_batch)
            aupr_metric.update(preds_batch, labels_batch)

    avg_loss = running_loss / max(1, n_batches)
    avg_time_per_batch = total_time / max(1, n_batches)

    print(
        f"Evaluation results - Avg Loss: {avg_loss:.6f}, Avg Time/Batch: {avg_time_per_batch:.4f} sec"
    )
    # Compute final metrics
    auroc = auroc_metric.compute().item()
    aupr = aupr_metric.compute().item()

    output_path_metrics = os.path.join(output_path, "metrics/")
    os.makedirs(output_path_metrics, exist_ok=True)
    with open(os.path.join(output_path_metrics, "metrics.txt"), "w") as f:
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"AUPR:  {aupr:.4f}\n")

    metrics = {
        "AUROC": auroc,
        "AUPR": aupr,
        "Time": avg_time_per_batch,
        "Loss": avg_loss,
    }

    # Visualization on fixed images
    output_path_visualizations = None
    # if imgs_vis is not None and labels_vis is not None:
    #     output_path_visualizations = os.path.join(output_path, "visualizations/")
    #     output_path_visualizations = visualize_anomaly_grid(
    #         dino_model=dino_model,
    #         ar_model=ar_model,
    #         imgs_vis=imgs_vis,  # fixed across epochs
    #         labels_vis=labels_vis,
    #         device=device,
    #         img_size=img_size,
    #         output_dir=output_path_visualizations,
    #         epoch=epoch,
    #     )

    return avg_loss, metrics, output_path_visualizations


def train_ar2d_model(
    dino_model,
    ar_model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=1e-3,
    output_dir=None,
    val_interval=1,
    img_size=None,
    imgs_vis=None,
    labels_vis=None,
    use_wandb=False,
):
    """
    Train the AR model to reconstruct raw images directly (no-DINO variant).

    Identical training loop to the DINO version but operates on pixel space
    rather than DINO feature maps.  Useful for ablation / baseline comparisons.

    Returns:
        ar_model: The trained model (best checkpoints saved to disk).
    """
    ar_model.train()

    criterion = nn.MSELoss()
    # optimizer = optim.Adam(ar_model.parameters(), lr=lr)
    optimizer = optim.AdamW(ar_model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_test_aupr = -float("inf")
    best_val_aupr = -float("inf")

    best_epoch = -1

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        ar_model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            # 1) DINO features as 2D maps
            # feats_2d = extract_dino_tokens_2d(dino_model, imgs, device)  # [B, C, H, W]

            aug_prob = 0.2  # apply augmentation to 80% of batches
            noise_coef = 0.03  # scale of noise relative to feature s
            patch_size = 2  # size of square patch to mask
            mask_p = 0.1  # probability of masking each patch

            feats_2d_aug = imgs.clone()
            if torch.rand((), device=imgs.device) < aug_prob:
                # (a) noise scaled by per-sample, per-channel std
                std = (
                    imgs.detach()
                    .flatten(2)
                    .std(dim=2, keepdim=True)
                    .unsqueeze(-1)
                    .clamp_min(1e-6)
                )
                feats_2d_aug = feats_2d_aug + noise_coef * std * torch.randn_like(
                    feats_2d_aug
                )

            #     # (b) patch masking
            #     feats_2d_aug = random_patch_mask(feats_2d_aug, patch_size=patch_size, p=mask_p)

            # 2) AR forward
            preds = ar_model(feats_2d_aug)  # [B, C, H, W]

            # 3) reconstruction loss (per-location feature prediction)
            loss = criterion(preds, imgs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        # === VALIDATION STEP ===
        # val_loss, val_metrics, val_visualizations_path = evaluate_ar2d_model(dino_model, ar_model, val_loader, device)
        val_loss, val_metrics, val_visualizations_path = evaluate_ar2d_model(
            dino_model=dino_model,
            ar_model=ar_model,
            dataloader=val_loader,
            imgs_vis=imgs_vis,
            labels_vis=labels_vis,
            criterion=criterion,
            device=device,
            epoch=epoch,
            output_path=os.path.join(output_dir, "val"),
        )

        # === TEST STEP ===
        # test_loss, test_metrics, test_visualizations_path = evaluate_ar2d_model(dino_model, ar_model, test_loader, device)
        test_loss, test_metrics, test_visualizations_path = evaluate_ar2d_model(
            dino_model=dino_model,
            ar_model=ar_model,
            dataloader=test_loader,
            imgs_vis=imgs_vis,
            labels_vis=labels_vis,
            criterion=criterion,
            device=device,
            epoch=epoch,
            output_path=os.path.join(output_dir, "test"),
        )

        # === SAVE BEST MODELS ===
        ckpt_dir = os.path.join(output_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model with best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            ckpt_path_val = os.path.join(ckpt_dir, "model_best_val_loss.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ar_model.state_dict(),
                    "val_loss": val_loss,
                    "img_size": img_size,
                },
                ckpt_path_val,
            )

        # Save model with best validation AUPR
        if val_metrics["AUPR"] > best_val_aupr:
            best_val_aupr = val_metrics["AUPR"]

            ckpt_path_val = os.path.join(ckpt_dir, "model_best_val_aupr.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ar_model.state_dict(),
                    "val_loss": val_loss,
                    "img_size": img_size,
                },
                ckpt_path_val,
            )

        # Save model with best test AUPR
        if test_metrics["AUPR"] > best_test_aupr:
            best_test_aupr = test_metrics["AUPR"]

            ckpt_path_test = os.path.join(ckpt_dir, "model_best_test_aupr.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ar_model.state_dict(),
                    "val_loss": val_loss,
                    "img_size": img_size,
                },
                ckpt_path_test,
            )

        # === LOGGING ===
        log_dict = {
            "epoch": epoch,
            "train/train_loss": train_loss,
            "val/val_loss": val_loss,
            "test/test_loss": test_loss,
            "val/AUROC": val_metrics["AUROC"],
            "val/AUPR": val_metrics["AUPR"],
            "test/AUROC": test_metrics["AUROC"],
            "test/AUPR": test_metrics["AUPR"],
        }

        # Print epoch progress to console
        log_msg = f"[Epoch {epoch}/{epochs}] | " + " | ".join(
            [f"{k}: {v:.4f}" for k, v in log_dict.items() if k != "epoch"]
        )
        print(log_msg)

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(log_dict, step=epoch)

    return ar_model
