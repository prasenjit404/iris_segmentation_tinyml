# 👁️ Iris Segmentation Using TinyML

### Edge-Optimized Deep Learning Pipeline for Robust Iris Segmentation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-Inference-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 📋 Overview

This project presents a **lightweight, edge-optimized iris segmentation pipeline** built around a custom **TinyUNet** architecture containing only ~28,000 trainable parameters. Trained on the UBIRIS visible-light eye dataset using PyTorch, the model accurately segments the iris annulus while deliberately excluding the pupil, sclera, eyelashes, and specular glare — challenges that are notoriously difficult under visible-light conditions.

The key innovation lies in the **deployment philosophy**: while PyTorch is used for offline training, all inference-time computation is executed through **pure NumPy array operations**, eliminating the need for heavyweight deep-learning frameworks (PyTorch, TensorFlow, ONNX Runtime) at runtime. This makes the pipeline viable for deployment on resource-constrained edge devices and TinyML-class hardware (ARM Cortex-M, Raspberry Pi, etc.) where GPU support and large runtime libraries are unavailable. A multi-stage **geometric refinement** post-processor transforms the raw binary mask into a smooth, circle-fitted annular overlay on the original RGB image, producing presentation-ready visualisations.

---

## ✨ Key Features

- **🧠 TinyUNet Architecture** — A minimal 3-stage encoder–decoder with skip connections (~28K parameters), purpose-built for binary iris segmentation at 128×128 resolution.
- **⚡ Pure-NumPy Inference** — Framework-free forward pass using only NumPy array operations — no GPU, no PyTorch, no TensorFlow required at deployment.
- **🔵 Geometric Refinement** — Least-squares circle fitting (Kåsa + Gauss–Newton) with 4× supersampled antialiased mask rendering transforms pixelated raw masks into smooth annular overlays.
- **📊 Batch Processing CLI** — Process entire multi-subject datasets offline, producing both raw binary masks and refined RGB overlays for every image.
- **🌐 Interactive Streamlit UI** — Upload an eye image, run segmentation in real-time, and visualise the Original → Raw Mask → Refined Overlay side-by-side with circle-fitting diagnostics.
- **📥 One-Click Downloads** — Export both the raw mask and refined overlay as PNG files directly from the web interface.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Model Training** | PyTorch 2.x | Offline training with Adam optimiser, class-weighted cross-entropy, augmentation |
| **Inference** | NumPy 1.24+ | Framework-free forward pass (convolution, batch norm, ReLU, pooling) |
| **Image Processing** | OpenCV 4.x, Pillow 10.x | Morphological ops, contour detection, circle drawing, format I/O |
| **Circle Fitting** | SciPy 1.11+ | Least-squares algebraic + geometric circle fitting |
| **Web Interface** | Streamlit 1.30+ | Interactive single-page demonstration app |
| **Language** | Python 3.10+ | End-to-end implementation |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/iris-segmentation-tinyml.git
cd iris-segmentation-tinyml

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Interactive Streamlit UI

Launch the web application for real-time, single-image segmentation:

```bash
streamlit run streamlit_app.py
```

Upload an eye image → click **🚀 Run Segmentation** → view the three-column output (Original | Raw Mask | Refined Overlay) with circle-fitting diagnostics and download buttons.

### Batch Processing

Process an entire multi-subject dataset from the command line:

```bash
# Default: processes dataset/ → segmented_output/
python batch_segmentation.py

# Custom paths and person count
python batch_segmentation.py --dataset dataset --output segmented_output --n-persons 10
```

For each input image, two files are saved:
- `{image_name}_mask.png` — Raw binary mask (black/white)
- `{image_name}_overlay.png` — Geometrically refined RGB overlay

### Train TinyUNet (Optional)

Re-train the model on the UBIRIS dataset:

```bash
python scripts/train_tinyunet.py
```

The best checkpoint is saved to `outputs/models/tinyunet.pth`.

---

## 📁 Project Structure

```
iris-segmentation-tinyml/
├── src/eye_feature_pipeline/
│   ├── tinyunet.py               # TinyUNet model, training loop, inference
│   ├── geometric_refinement.py   # Circle fitting, overlay composition
│   └── __init__.py
├── scripts/
│   ├── train_tinyunet.py         # Training script
│   ├── setup_venv.ps1            # Environment setup (Windows)
│   ├── setup_venv.sh             # Environment setup (Linux/macOS)
│   └── run_streamlit.ps1         # Streamlit launcher
├── configs/                      # YAML configuration files
├── dataset/                      # UBIRIS eye images (not tracked in git)
├── outputs/models/               # Trained model checkpoints
├── batch_segmentation.py         # Batch processing CLI
├── streamlit_app.py              # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 👥 Authors & Credits

| | |
|---|---|
| **Authors** | **Prasenjit Saha** & **Mayank Sharma** |
| **Supervisor** | **Dr. Binod Kumar Singh** |
| **Institution** | Department of Computer Science and Engineering, NIT Jamshedpur |
| **Programme** | Master of Computer Applications (MCA), 6th Semester |
| **Date** | April 2026 |

---

## 📄 License

This project was developed as an academic Major Project submission at NIT Jamshedpur.
