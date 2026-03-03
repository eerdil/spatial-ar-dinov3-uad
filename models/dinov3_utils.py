from sklearn.decomposition import PCA
import torch
import os
from config.paths import PROJECT_PATH, checkpoint_paths

def load_dinov3_models(model_type):
    # DINOv3 ViT models pretrained on web images
    model = torch.hub.load(repo_or_dir = PROJECT_PATH, 
                           model = model_type, 
                           source='local', 
                           weights=checkpoint_paths[model_type])
    return model

def create_pca_image(features, n_components=3, H=None, W=None):
    N, S, D = features.shape
    if S < n_components:
        raise ValueError(f"Number of tokens {S} is less than the number of PCA components {n_components}.")
    
    features = features.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features.reshape(N*S, D))
    if H is None or W is None:
        H = W = int(S**0.5)
    pca_result = pca_result.reshape(N, H, W, n_components)
    return pca_result

def extract_dino_tokens_2d(dino_model, imgs, device):
    """
    imgs: [B, 3, H, W]
    Returns:
        feats_2d: [B, C, H_tok, W_tok]
    """
    with torch.no_grad():
        feats = dino_model.get_intermediate_layers(imgs)[0]  # [B, S, D]

    B, S, D = feats.shape
    side = int(S ** 0.5)
    assert side * side == S, f"S={S} is not a perfect square"

    feats_2d = feats.view(B, side, side, D).permute(0, 3, 1, 2)  # [B, D, H, W]
    return feats_2d