"""Streamlit frontend: GAN Image Quality and Diversity Analyzer.

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from analyzer import (
    build_report,
    extract_features,
    generate_from_torchscript,
    pca_2d,
    similarity_matrix,
)

GALLERY_LIMIT = 64
HEATMAP_LIMIT = 100

st.set_page_config(page_title="GAN Diversity Analyzer", layout="wide")
st.title("GAN Image Quality and Diversity Analyzer")
st.caption("Evaluate whether a set of generated images is diverse or repetitive.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Input")
source = st.sidebar.radio("Sample source", ["Upload images", "Generate from GAN"])

images: list[Image.Image] = []

if source == "Upload images":
    files = st.sidebar.file_uploader(
        "Generated images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
    )
    if files:
        images = [Image.open(f) for f in files]
else:
    model_path = st.sidebar.text_input("TorchScript generator path", value="generator.pt")
    n_samples = st.sidebar.slider("Number of samples", 4, 256, 64)
    latent_dim = st.sidebar.number_input("Latent dimension", min_value=1, value=100)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42)
    if st.sidebar.button("Generate samples"):
        try:
            with st.spinner("Sampling generator..."):
                st.session_state["generated"] = generate_from_torchscript(
                    model_path, int(n_samples), int(latent_dim), int(seed)
                )
        except Exception as exc:  # surface load/shape errors in the UI
            st.sidebar.error(f"Generation failed: {exc}")
    images = st.session_state.get("generated", [])

st.sidebar.header("Analysis")
sim_threshold = st.sidebar.slider(
    "Near-duplicate threshold", 0.80, 0.99, 0.95, 0.01,
    help="Two samples with correlation at or above this value count as the same image.",
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if len(images) < 2:
    st.info("Provide at least two images: upload them or generate them from a GAN.")
    st.stop()

feats = extract_features(images)
sim = similarity_matrix(feats)
report = build_report(images, sim, sim_threshold)

# Gallery
st.subheader("Generated samples")
shown = images[:GALLERY_LIMIT]
cols = st.columns(8)
for i, img in enumerate(shown):
    cols[i % 8].image(img, caption=f"#{i}", use_container_width=True)
if len(images) > GALLERY_LIMIT:
    st.caption(f"Showing first {GALLERY_LIMIT} of {len(images)} samples. "
               "All samples are used in the analysis.")

# Metrics
st.subheader("Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", report.n)
m2.metric("Unique (clusters)", report.unique_clusters, f"{report.unique_ratio:.0%} unique",
          delta_color="off")
m3.metric("Exact duplicates", report.exact_duplicates)
m4.metric("Mean pairwise similarity", f"{report.mean_similarity:.3f}")

# Report
st.subheader("Diversity report")
if report.verdict == "Diverse":
    st.success(f"**{report.verdict}.** {report.explanation}")
elif report.verdict == "Moderately diverse":
    st.warning(f"**{report.verdict}.** {report.explanation}")
else:
    st.error(f"**{report.verdict}.** {report.explanation}")
st.caption(f"Highest pairwise similarity between two different samples: {report.max_similarity:.3f}")

# Visualizations
st.subheader("Visualizations")
left, right = st.columns(2)

with left:
    st.markdown("**Pairwise similarity heatmap**")
    if report.n <= HEATMAP_LIMIT:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")
        fig.colorbar(im, ax=ax, label="Correlation")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption(f"Heatmap hidden above {HEATMAP_LIMIT} samples to keep rendering fast.")

with right:
    st.markdown("**Diversity map (PCA of pixel features)**")
    coords = pca_2d(feats)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="plasma", s=40)
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.set_title("Spread of samples: tight cluster = repetitive")
    st.pyplot(fig2)
    plt.close(fig2)

# Per-sample similarity to its nearest neighbour
st.subheader("Nearest-neighbour similarity")
if report.n > 1:
    nn = np.max(np.where(np.eye(report.n, dtype=bool), -np.inf, sim), axis=1)
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.hist(nn, bins=30, range=(-1, 1), color="#4C72B0")
    ax3.axvline(sim_threshold, color="red", linestyle="--", label="duplicate threshold")
    ax3.set_xlabel("Similarity to closest other sample")
    ax3.set_ylabel("Count")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)
