import os

# Project root: defaults to the directory containing this config file's parent.
PROJECT_PATH = os.getenv(
    "PROJECT_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Path to DINOv3 pretrained checkpoints.
# Download checkpoints and place them in pretrained_models/ (default),
# or set the DINO_CHECKPOINTS_PATH environment variable.
DINO_PATH = os.getenv(
    "DINO_CHECKPOINTS_PATH",
    os.path.join(PROJECT_PATH, "pretrained_models")
)

# Dataset paths — default to data/<dataset_name>/ inside the project.
# Override with environment variables if your data lives elsewhere.
DATASET_PATH = {
    'brats': os.getenv("DATASET_BRATS_PATH", os.path.join(PROJECT_PATH, "data", "brats")),
    'bmad':  os.getenv("DATASET_BMAD_PATH",  os.path.join(PROJECT_PATH, "data", "bmad")),
    'resc':  os.getenv("DATASET_RESC_PATH",  os.path.join(PROJECT_PATH, "data", "resc")),
}


checkpoint_paths = {
    'dinov3_vits16':         os.path.join(DINO_PATH, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    'dinov3_vits16plus':     os.path.join(DINO_PATH, "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    'dinov3_vitb16':         os.path.join(DINO_PATH, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    'dinov3_vitl16':         os.path.join(DINO_PATH, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    'dinov3_vith16plus':     os.path.join(DINO_PATH, "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    'dinov3_vit7b16':        os.path.join(DINO_PATH, "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
    'dinov3_convnext_tiny':  os.path.join(DINO_PATH, "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
    'dinov3_convnext_small': os.path.join(DINO_PATH, "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
    'dinov3_convnext_base':  os.path.join(DINO_PATH, "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
    'dinov3_convnext_large': os.path.join(DINO_PATH, "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"),
    'dinov3_vitl16_sat':     os.path.join(DINO_PATH, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
    'dinov3_vit7b16_sat':    os.path.join(DINO_PATH, "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"),
}
