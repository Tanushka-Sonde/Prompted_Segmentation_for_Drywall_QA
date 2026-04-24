# 🧱 Prompted Segmentation for Drywall QA

#### link to google colab : https://colab.research.google.com/drive/1h3PegpoC0AxBzOUQK-kh3jhk7-4Bp6vX?usp=sharing 


**Text-conditioned binary segmentation** — given an image and a natural-language prompt, produce a pixel-perfect mask for drywall defects.

```
Input:  [drywall image]  +  "segment crack"
Output: [binary mask — 0/255 PNG, same spatial size]
```

---

## 📌 Quick Summary

| Item | Detail |
|---|---|
| **Task** | Prompted semantic segmentation (2 classes) |
| **Prompts** | `"segment crack"` · `"segment taping area"` |
| **Method A** | Train from scratch — CLIP text encoder + U-Net + FiLM |
| **Method B** | Fine-tune — CLIPSeg (`CIDAS/clipseg-rd64-refined`) |
| **Loss** | BCE (0.5) + Dice (0.5) |
| **Global Seed** | `42` (Python · NumPy · PyTorch · CUDA) |
| **Image size** | 256 × 256 |
| **Output masks** | Single-channel PNG · values `{0, 255}` · original spatial size |

---

## 📊 Results at a Glance

### Per-Prompt Metrics (Test Set)

| Model | Prompt | N | mIoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|
| **Scratch (CLIP-UNet)** | segment taping area | 154 | **0.6162** | **0.7434** | 0.7422 | **0.8176** |
| **Scratch (CLIP-UNet)** | segment crack | 806 | **0.4811** | **0.6264** | 0.6478 | 0.7452 |
| **Scratch (CLIP-UNet)** | **Overall** | **960** | **0.5027** | **0.6451** | **0.6629** | **0.7568** |
| Fine-tuned (CLIPSeg) | segment taping area | 154 | 0.4871 | 0.6420 | **0.6940** | 0.6663 |
| Fine-tuned (CLIPSeg) | segment crack | 806 | 0.4480 | 0.5958 | 0.5911 | **0.7516** |
| Fine-tuned (CLIPSeg) | **Overall** | **960** | 0.4543 | 0.6032 | 0.6076 | 0.7379 |

### Runtime & Footprint

| Method | Architecture | Params | Train Time | Avg Inference | Model Size |
|---|---|---|---|---|---|
| **Method A — Scratch** | CLIP-UNet + FiLM | 183.3 M | 181.2 min | **22.5 ms/img** | 481.7 MB |
| **Method B — Fine-tune** | CLIPSeg decoder | 150.7 M | **71.0 min** | 37.8 ms/img | 603.0 MB |

> **Key finding:** Method A (from scratch) outperforms Method B on all metrics. Method B trains 2.5× faster but plateaus lower. Method A has faster inference (22.5 vs 37.8 ms/img).

---

## 📂 Repository Structure

```
drywall-segmentation/
│
├── drywall_segmentation_full.ipynb    ← Main notebook (run top-to-bottom)
├── README.md                          ← This file
├── drywall_QA_report.docx             ← Full technical report
│
├── data/
│   ├── taping/                        ← Dataset 1: Drywall-Join-Detect
│   └── crack/                         ← Dataset 2: Cracks-3ii36
│
├── predictions/
│   ├── scratch/                       ← Method A masks
│   │   └── {image_id}__{prompt_slug}.png
│   └── finetune/                      ← Method B masks
│       └── {image_id}__{prompt_slug}.png
│
├── checkpoints/
│   ├── scratch_best.pth               ← Best Method A checkpoint (481.7 MB)
│   └── clipseg_ft_best/               ← Best Method B checkpoint (HuggingFace fmt)
│
└── report/
    ├── eval_results.csv
    ├── comparison_table.csv
    ├── training_curves.png
    ├── metric_comparison_bar.png
    ├── visual_examples.png
    ├── iou_distribution.png
    ├── failures_scratch_clip_unet.png
    └── failures_fine_tuned_clipseg.png
```

---

## 🗃️ Datasets

| Dataset | Roboflow Project | Prompt Pool |
|---|---|---|
| **Taping area** | `objectdetect-pu6rn / drywall-join-detect` | `"segment taping area"`, `"segment joint tape"`, `"segment drywall seam"`, `"highlight tape region"` |
| **Cracks** | `fyp-ny1jt / cracks-3ii36` | `"segment crack"`, `"segment wall crack"`, `"detect crack"`, `"highlight crack region"` |

### Data Splits (seed = 42)

| Dataset | Total | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|
| Taping | ~220 | ~154 | ~33 | ~33 |
| Crack | ~927 | ~649 | ~139 | ~139 |
| **Combined** | **~1147** | **~803** | **~172** | **~172** |

> Confirmed test-set counts: 154 taping + 806 crack = **960 total**.

---

## 🔧 Setup

```bash
pip install torch torchvision transformers timm
pip install opencv-python-headless Pillow matplotlib scikit-learn
pip install torchmetrics tqdm albumentations pycocotools
pip install git+https://github.com/openai/CLIP.git
```

Replace `CFG['api_key']` in the notebook with your Roboflow API key, then **run all cells top-to-bottom**.

---

## 🏗️ Method A — Training from Scratch (CLIP-UNet + FiLM)

```
"segment crack" ──► CLIP ViT-B/32 ──► text_emb (512-d) ──► γ, β projections
                                                                    │
Image (3×256×256) ──► U-Net Encoder ──► skip connections ──► Decoder + FiLM ──► mask logits
                         enc1: 64ch                          up4: FiLM(512d)
                         enc2: 128ch                         up3: FiLM(256d)
                         enc3: 256ch                         up2: FiLM(128d)
                         enc4: 512ch                         up1: FiLM(64d)
                         bottleneck: 1024ch                  head: 1×1 conv
```

**FiLM (Feature-wise Linear Modulation):**
```
output = γ(text_emb) · feature_map + β(text_emb)
```

**Training strategy:**
| Phase | Epochs | CLIP encoder | LR |
|---|---|---|---|
| Warm-up | 1–5 | Frozen | 1e-3 |
| Full fine-tune | 6–30 | Unfrozen at 1% LR | 1e-3 / 1e-5 |

---

## 🔬 Method B — Fine-tuning CLIPSeg

- Base: `CIDAS/clipseg-rd64-refined` (Lüddecke & Ecker, CVPR 2022)
- **Frozen**: CLIP vision encoder + CLIP text encoder
- **Trainable**: Decoder only (~2M params)
- Epochs: 15 · LR: 5e-5 · AdamW · CosineAnnealingLR

---

## ⚙️ Data Augmentation (Training — Method A)

```python
A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.2),
A.Rotate(limit=15, p=0.3),
A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
```

---

## 🔍 Known Failure Modes

**Both methods:**
- Thin hairline cracks with low contrast are missed
- Very dark / saturated images reduce quality
- Partial defects near image edges cause erratic masks

**Method A specific:**
- Boundary leakage on taping areas near corners/edges
- Slow convergence in first 5 epochs (randomly initialised encoder)

**Method B specific:**
- Over-segmentation on noisy/textured walls
- Lower recall on taping (0.6663 vs 0.8176 for Method A)
- Higher inference latency due to HuggingFace processor overhead

---

## 📤 Output Mask Specification

| Property | Specification |
|---|---|
| Format | PNG |
| Channels | 1 (single-channel grayscale) |
| Pixel values | `0` = background · `255` = defect |
| Spatial size | Same as original source image |
| Filename | `{image_id}__{prompt_slug}.png` |

---

## 🌱 Reproducibility

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ['PYTHONHASHSEED']       = str(SEED)
```

---

## 📄 References

- **CLIPSeg**: Lüddecke & Ecker, "Image Segmentation Using Text and Image Prompts", CVPR 2022
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- [Dataset 1 — Drywall-Join-Detect @ Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- [Dataset 2 — Cracks-3ii36 @ Roboflow](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)

---

*Seed: 42 · Framework: PyTorch · Python 3.10*