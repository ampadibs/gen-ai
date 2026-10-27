# GAN Image Quality and Diversity Analyzer

A Streamlit app that tells you whether a set of GAN-generated images is diverse or repetitive. It runs locally.

## Features

- **Two input sources**
  - Upload a folder's worth of generated images (PNG, JPG, WEBP).
  - Sample directly from a pretrained GAN exported as TorchScript.
- **Gallery** of the generated samples.
- **Metrics**: number of unique images (clusters), exact duplicates, mean pairwise similarity.
- **Visualizations**: similarity heatmap, PCA diversity map, nearest-neighbour similarity histogram.
- **Report**: a verdict of *Diverse*, *Moderately diverse*, or *Repetitive*, with an explanation.

## How it works

1. Each image is resized to 32×32 RGB, flattened, and mean-centred. It is then scaled to unit length, so the dot product between two images equals their Pearson correlation (1.0 = same structure, ~0 = unrelated).
2. **Exact duplicates** are found by MD5-hashing the pixels of each image (at 64×64).
3. **Unique count** uses greedy clustering: an image joins an existing cluster if its correlation with that cluster's representative is at or above the threshold (default 0.95). Each cluster counts as one unique image.
4. **Verdict** uses the unique ratio (unique ÷ total):
   - ≥ 0.90 → Diverse
   - 0.60 – 0.90 → Moderately diverse
   - < 0.60 → Repetitive (possible mode collapse)

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

### Option A: upload images

1. In the sidebar, choose **Upload images**.
2. Upload at least two images, for example outputs saved from a GAN or from a folder of generated samples.
3. Adjust the near-duplicate threshold if needed.

### Option B: generate from a GAN

The app loads a generator saved with `torch.jit`, so you do not need the model's class definition. Export one from your own code:

```python
import torch
G = ...                      # your trained generator, set to eval()
example_z = torch.randn(1, 100)
traced = torch.jit.trace(G.eval(), example_z)
traced.save("generator.pt")
```

Requirements for the generator:
- Input: a `(batch, latent_dim)` tensor.
- Output: `(batch, C, H, W)` with values in [-1, 1] (for example, a `tanh` output).

Then in the app, choose **Generate from GAN**, set the model path, latent dimension (must match the model), and seed, and click **Generate samples**.

For a pretrained model without your own code, you can export a DCGAN or similar generator the same way once you have its weights.

## Project structure

```
gan-diversity-analyzer/
├── app.py            # Streamlit UI
├── analyzer.py       # features, similarity, duplicates, clustering, report, GAN sampling
├── requirements.txt
└── README.md
```

## Notes and limitations

- Pixel correlation is a simple proxy. It catches exact copies, shifted copies, and near-identical outputs, but it does not measure semantic diversity. For stronger evaluation, add a feature-based metric such as Inception-based FID or precision/recall.
- The verdict thresholds are heuristics. Tune `sim_threshold` for your data and state the chosen values in your report.
- The app reads samples into memory, so very large sets (thousands of high-resolution images) may be slow.
