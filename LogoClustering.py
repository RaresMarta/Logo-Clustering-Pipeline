import os
import sys
import time
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pandas as pd
import numpy as np
import requests
import re
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import models, transforms

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------
# Global Constants & Configuration
# ---------------------------------------
DATASET_FILE = "logos.snappy.parquet"
FAVICON_CSV = "favicons_fetched.csv"
TARGET_SIZE = (64, 64)
MAX_WORKERS_FETCH = 15
MAX_WORKERS_DOWNLOAD = 10

# Fixed DBSCAN parameters
DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 5

# =======================================
# 1. Load dataset
# =======================================
def load_and_clean_data() -> pd.DataFrame:
    """Load dataset, remove duplicates, and filter domains by regex."""
    print("\nLoading dataset...")

    df = pd.read_parquet(DATASET_FILE, engine="pyarrow")
    print(f"Initial dataset size: {len(df)}")

    # Remove duplicates
    dup_count = df['domain'].duplicated().sum()
    df = df.drop_duplicates(subset=["domain"])

    print(f"Removed {dup_count} duplicate domains.")
    return df

# =======================================
# 2. Favicon Fetching
# =======================================
def is_valid_image(response) -> bool:
    """Returns True if the response content is a valid image."""
    try:
        Image.open(BytesIO(response.content))
        return True
    except UnidentifiedImageError:
        return False

def get_favicon(domain: str) -> str | None:
    """
    Try Google's favicon API first, then direct /favicon.ico.
    Returns a URL if found, otherwise None.
    """
    google_favicon = f"https://www.google.com/s2/favicons?sz=128&domain={domain}"
    direct_favicon = f"https://{domain}/favicon.ico"
    for url in [google_favicon, direct_favicon]:
        try:
            response = requests.get(url, timeout=4, stream=True)
            time.sleep(0.1)
            if response.status_code == 200 and is_valid_image(response):
                return url
        except requests.RequestException:
            continue
    return None

def fetch_favicon_urls(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve favicon URLs, optionally cached in a CSV."""
    if os.path.exists(FAVICON_CSV):
        print("\nLoading previously fetched favicon URLs...")
        fav_df = pd.read_csv(FAVICON_CSV)
    else:
        print("\nFetching favicon URLs in parallel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_FETCH) as executor:
            valid_df["logo_url"] = list(executor.map(get_favicon, valid_df["domain"]))
        fav_df = valid_df.dropna(subset=["logo_url"]).copy()
        print(f"Favicons retrieved for {len(fav_df)} domains out of {len(valid_df)} validated domains.")
        fav_df.to_csv(FAVICON_CSV, index=False)
    return fav_df

# =======================================
# 3. Download Images (in memory only)
# =======================================
def download_favicon_image(logo_url: str) -> tuple[str, Image.Image | None]:
    """
    Download an image from the provided favicon URL and store it in RAM.
    """
    try:
        response = requests.get(logo_url, timeout=5, stream=True)
        response.raise_for_status()
        if not is_valid_image(response):
            return logo_url, None

        image = Image.open(BytesIO(response.content))
        # Composite transparency onto white if needed.
        if image.mode in ("P", "RGBA"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

        image = image.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)
        return logo_url, image
    except requests.RequestException:
        return logo_url, None

def download_favicon_images(fav_df: pd.DataFrame) -> dict:
    """Download favicon images in parallel (all in memory)."""
    print("Downloading favicon images...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as executor:
        results = list(executor.map(download_favicon_image, fav_df["logo_url"]))

    images_dict = {url: img for (url, img) in results if img is not None}
    print(f"Successfully downloaded {len(images_dict)} favicon images.")
    return images_dict

# =======================================
# 4. Remove Duplicates
# =======================================
def compute_md5(image: Image.Image) -> str:
    """Compute an MD5 hash for the in-memory image (PNG format)."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

def remove_duplicate_icons(images_dict: dict) -> dict:
    """
    Remove duplicates in memory based on MD5 hash.
    No icons are saved to disk.
    """
    unique = {}
    for url, img in images_dict.items():
        md5_hash = compute_md5(img)
        if md5_hash not in unique:
            unique[md5_hash] = (url, img)

    duplicates_removed = len(images_dict) - len(unique)
    print(f"Removed {duplicates_removed} duplicate icons in memory.")
    return {url: img for (url, img) in unique.values()}

# =======================================
# 5. Model Loading & Feature Extraction
# =======================================
def load_model(model_name: str):
    """Load the specified model architecture, removing classification layer."""
    if model_name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Invalid model: {model_name}")

    net = torch.nn.Sequential(*list(net.children())[:-1])  # remove final layer
    net.eval()
    return net

def extract_features_from_images(images_dict: dict, model_name: str) -> tuple[list, np.ndarray]:
    """Extract deep features from each image using the chosen model."""
    print(f"\nExtracting features using {model_name}...")

    model = load_model(model_name)
    transform_pipeline = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def extract_feats(image: Image.Image) -> np.ndarray | None:
        try:
            x = transform_pipeline(image).unsqueeze(0)
            with torch.no_grad():
                feats = model(x).flatten().numpy()
            return feats
        except Exception:
            return None

    feat_map = {}
    for url, img in images_dict.items():
        feats = extract_feats(img)
        if feats is not None:
            feat_map[url] = feats

    if not feat_map:
        print("No features extracted. Exiting...")
        sys.exit()

    feature_urls = list(feat_map.keys())
    features_matrix = np.array(list(feat_map.values()))
    print(f"Extracted features from {len(feature_urls)} images.")
    return feature_urls, features_matrix

# =======================================
# 6. DBSCAN Clustering
# =======================================
def cluster_features(features_matrix: np.ndarray) -> np.ndarray:
    """
    Cluster features using DBSCAN only (eps=0.2, min_samples=5).
    """
    print(f"\nClustering with DBSCAN...")
    dist_matrix = cosine_distances(features_matrix)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters (label -1 indicates noise).")
    return labels

# ---------------------------------------
# 7. Visualization & Sample Display (modified)
# ---------------------------------------
def visualize_clusters(features_matrix: np.ndarray, labels: np.ndarray, model_name: str, save_folder: str):
    """Reduce features to 2D using t-SNE, plot the clusters, and save the figure."""
    print("Reducing dimensions with t-SNE...")

    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200.0)
    X_embedded = tsne.fit_transform(features_matrix)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='plasma', edgecolors='k')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("Favicon Clustering (DBSCAN)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Build the filename: e.g. "64_resnet50_tsne.png"
    filename = os.path.join(save_folder, f"{TARGET_SIZE[0]}_{model_name}_tsne.png")
    plt.savefig(filename)
    print(f"t-SNE plot saved as {filename}")
    plt.close()

def print_clusters_and_samples(labels: np.ndarray, feature_urls: list, images_dict: dict, model_name: str, save_folder: str, n_samples: int = 10):
    """Save each cluster's sample favicons in a horizontal layout to disk."""
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # remove noise points

    for cluster_id in unique_clusters:
        cluster_urls = [u for u, lab in zip(feature_urls, labels) if lab == cluster_id]
        print(f"Cluster {cluster_id} has {len(cluster_urls)} icons.")

        if not cluster_urls:
            continue

        sample_urls = np.random.choice(cluster_urls, size=min(n_samples, len(cluster_urls)), replace=False)
        fig, axes = plt.subplots(1, len(sample_urls), figsize=(15, 5))
        for i, url in enumerate(sample_urls):
            axes[i].imshow(images_dict[url])
            axes[i].axis('off')
        plt.suptitle(f"Cluster {cluster_id} (showing {len(sample_urls)} of {len(cluster_urls)})", fontsize=14)
        
        # Save the cluster sample plot with a filename e.g. "64_resnet50_cluster_3.png"
        filename = os.path.join(save_folder, f"{TARGET_SIZE[0]}_{model_name}_cluster_{cluster_id}.png")
        plt.savefig(filename)
        print(f"Cluster {cluster_id} sample plot saved as {filename}")
        plt.close()

# ---------------------------------------
# 8. Argument Parsing & Main (modified part)
# ---------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Favicon Clustering with DBSCAN (No Screen Plotting)")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2"], default="mobilenet_v2",
                        help="Model architecture (default: mobilenet_v2)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Create a "plots" folder if it doesn't exist.
    base_plots_folder = "plots"
    if not os.path.exists(base_plots_folder):
        os.makedirs(base_plots_folder)
    
    # Create a unique subfolder for this run.
    # Using the structure: TARGET_SIZE_model (e.g. "64_resnet50")
    base_run_folder = f"{TARGET_SIZE[0]}_{args.model}"
    run_folder = os.path.join(base_plots_folder, base_run_folder)
    counter = 1
    while os.path.exists(run_folder):
        run_folder = os.path.join(base_plots_folder, f"{TARGET_SIZE[0]}_{args.model}_{counter}")
        counter += 1
    os.makedirs(run_folder)
    print(f"Plots will be saved to: {run_folder}")

    # 1. Load & Clean Data
    df = load_and_clean_data()

    # 2. Fetch Favicon URLs
    fav_df = fetch_favicon_urls(df)

    # 3. Download images in memory
    images_dict = download_favicon_images(fav_df)

    # 4. Remove duplicates in memory
    images_dict = remove_duplicate_icons(images_dict)

    # 5. Extract features (user-chosen model)
    feature_urls, features_matrix = extract_features_from_images(images_dict, args.model)

    # 6. DBSCAN Clustering (using default cosine distance)
    labels = cluster_features(features_matrix)

    # 7. Visualization: t-SNE plot saved to disk
    visualize_clusters(features_matrix, labels, args.model, run_folder)

    # 8. Save cluster sample plots
    print_clusters_and_samples(labels, feature_urls, images_dict, args.model, run_folder, n_samples=10)

    print("Pipeline complete. All plots have been saved to the folder.")

if __name__ == "__main__":
    main()
