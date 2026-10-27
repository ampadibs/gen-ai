"""Core analysis for GAN sample quality and diversity.

Pipeline:
  images -> feature vectors -> pairwise similarity matrix
         -> exact duplicates (MD5) and near-duplicate clusters
         -> diversity metrics, 2-D projection, and a verdict report
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

FEATURE_SIZE = 32          # images are downsampled to 32x32 before comparison
DUPLICATE_SIZE = 64        # size used for exact-duplicate hashing


def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Feature extraction and similarity
# ---------------------------------------------------------------------------
def extract_features(images: list[Image.Image], size: int = FEATURE_SIZE) -> np.ndarray:
    """Flatten each image into a mean-centred, unit-norm vector.

    With these vectors, the dot product of two rows equals the Pearson
    correlation of their pixel values: 1.0 means identical structure,
    values near 0 mean unrelated images.
    """
    rows = []
    for img in images:
        arr = np.asarray(
            to_rgb(img).resize((size, size), Image.BILINEAR), dtype=np.float32
        ).ravel() / 255.0
        arr = arr - arr.mean()
        norm = np.linalg.norm(arr)
        rows.append(arr / norm if norm > 1e-8 else arr)
    if not rows:
        return np.empty((0, size * size * 3), dtype=np.float32)
    return np.stack(rows)


def similarity_matrix(feats: np.ndarray) -> np.ndarray:
    sim = feats @ feats.T
    np.fill_diagonal(sim, 1.0)
    return sim


def exact_duplicate_groups(images: list[Image.Image]) -> list[list[int]]:
    """Groups of indices whose pixels are byte-identical after resizing."""
    groups: dict[str, list[int]] = {}
    for i, img in enumerate(images):
        arr = np.asarray(
            to_rgb(img).resize((DUPLICATE_SIZE, DUPLICATE_SIZE)), dtype=np.uint8
        )
        digest = hashlib.md5(arr.tobytes()).hexdigest()
        groups.setdefault(digest, []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def cluster_unique(sim: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    """Greedy clustering: an image joins the first cluster whose representative
    it resembles at or above `threshold`; otherwise it starts a new cluster.
    Returns per-image labels and the number of clusters (effective unique count).
    """
    reps: list[int] = []
    labels = np.full(sim.shape[0], -1, dtype=int)
    for i in range(sim.shape[0]):
        for c, r in enumerate(reps):
            if sim[i, r] >= threshold:
                labels[i] = c
                break
        else:
            reps.append(i)
            labels[i] = len(reps) - 1
    return labels, len(reps)


def pca_2d(feats: np.ndarray) -> np.ndarray:
    """Project feature vectors onto their first two principal components."""
    n = feats.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    centered = feats - feats.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[: min(2, vt.shape[0])].T
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 1))])
    return coords


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class DiversityReport:
    n: int
    exact_duplicates: int
    unique_clusters: int
    unique_ratio: float
    mean_similarity: float
    max_similarity: float
    verdict: str
    explanation: str


def build_report(
    images: list[Image.Image], sim: np.ndarray, sim_threshold: float = 0.95
) -> DiversityReport:
    n = len(images)
    if n < 2:
        return DiversityReport(n, 0, n, 1.0, 0.0, 0.0,
                               "Not enough samples",
                               "Upload or generate at least two images to measure diversity.")

    off_diag = sim[~np.eye(n, dtype=bool)]
    mean_sim = float(off_diag.mean())
    max_sim = float(off_diag.max())

    exact = sum(len(g) - 1 for g in exact_duplicate_groups(images))
    _, k = cluster_unique(sim, sim_threshold)
    ratio = k / n

    if ratio >= 0.9:
        verdict = "Diverse"
        text = (f"{k} of {n} samples are distinct at a similarity threshold of "
                f"{sim_threshold:.2f}. The sample set covers varied content.")
    elif ratio >= 0.6:
        verdict = "Moderately diverse"
        text = (f"Only {k} of {n} samples are distinct. Some outputs repeat or "
                f"closely resemble each other, which may point to partial mode collapse.")
    else:
        verdict = "Repetitive"
        text = (f"Only {k} of {n} samples are distinct. The generator is producing "
                f"largely the same outputs, a sign of mode collapse.")

    return DiversityReport(n, exact, k, ratio, mean_sim, max_sim, verdict, text)


# ---------------------------------------------------------------------------
# Sample generation from a pretrained GAN
# ---------------------------------------------------------------------------
def generate_from_torchscript(
    model_path: str, n: int, latent_dim: int, seed: int = 0
) -> list[Image.Image]:
    """Sample n images from a generator exported with torch.jit.

    The generator must take a (batch, latent_dim) tensor and return
    (batch, C, H, W) values in [-1, 1] (e.g. a tanh output). Values are
    rescaled to [0, 255] for display and comparison.
    """
    import torch

    model = torch.jit.load(model_path, map_location="cpu").eval()
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, latent_dim, generator=gen)
    with torch.no_grad():
        out = model(z)

    out = ((out.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().byte()
    arr = out.permute(0, 2, 3, 1).cpu().numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
        return [Image.fromarray(a, mode="L") for a in arr]
    return [Image.fromarray(a) for a in arr]
