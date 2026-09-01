# 🗺️ Monocular Depth Estimation Microservice

A production-ready REST API microservice for single-image Monocular Depth Estimation. Built with a custom U-Net-like architecture (ResNet Encoder + Custom Decoder with Skip-Connections), exported to ONNX for high-speed CPU inference, and packaged into a lightweight Docker container powered by FastAPI.

---

## 🌟 Key Features

- **Custom Architecture:** Features high-level extraction via `timm` (ResNet) and a hand-crafted decoder with pyramidal skip-connections written in PyTorch.
- **Edge-Aware Loss:** Trained using a combined loss function ($L1_{log1p} + \alpha \cdot L1_{grad}$) engineered to maximize boundary sharpness around object edges.
- **High-Performance CPU Inference:** Serialized into **ONNX Runtime** format, enabling ultra-fast quantized CPU execution without requiring CUDA runtime overhead.
- **Asynchronous Lifecycle Management:** Uses FastAPI `lifespan` state managers to preload and retain ONNX model artifacts in RAM upon startup.
- **Zero Disk Memory Leaks:** End-to-end image processing executed entirely in RAM using `io.BytesIO` and `Pillow`, preventing temporary file leaks and disk bloat.
- **Lightweight Production Container:** Optimized Docker container based on `python:3.11-slim` stripped of heavy dev-dependencies (PyTorch, Torchvision), keeping image size minimal.

---

## 🏗️ System Architecture

[ Input Image (RGB) ]
         │
         ▼
 ┌───────────────┐
 │ FastAPI / App │
 └───────┬───────┘
         │ (Bytes stream in memory)
         ▼
 ┌───────────────┐
 │ Preprocessing │ ──> Resizing (384x384) & ImageNet Normalization
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ ONNX Runtime  │ ──> Optimized Engine Execution
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Postprocessing│ ──> Min-Max Normalization to 8-bit Gray PNG (Pillow)
 └───────┬───────┘
         │
         ▼
[ Binary Image Output ]

---

## 🛠️ Project Structure

```text
.
├── data/
│   ├── ny2_test/                # Flat Directory with Test Image Pairs (colors & depth)
│   │   ├── 00000_colors.png     # Sample RGB Input Image (8bit)
│   │   ├── 00000_depth.png      # Sample Ground Truth Depth Map (16bit)
│   │   └── ...
│   ├── ny2_train/               # Scene-based Training Sequences
│   │   ├── basement_0001a_out/  # Sequence Folder for Video Frames
│   │   │   ├── 1.jpg            # Video Frame RGB Image
│   │   │   ├── 1.png            # Ground Truth Depth for Frame
│   │   │   └── ...
│   │   └── ...
│   ├── ny2_test.csv             # Test set file paths
│   └── ny2_train.csv            # Train set file paths
│
├── models/                      # Model folder
│   └── unet_resnet34_depth_384.onnx
│
├── notebooks/                   # Model Development
│   └── 01_model_preparation.ipynb
│
├── src/                         # Source Code
│   ├── api/
│   │   ├── __init__.py      
│   │   ├── main.py             # FastAPI application & endpoints
│   │   └── pipeline.py         # FastAPI Inference Pipeline (load ONNX model & process images)
│   ├── config/
│   │   ├── __init__.py      
│   │   ├── config.py           # Environment & path configurations
│   │   └── logger.py           # Centralized logging setup
│   ├── data/
│   │   ├── __init__.py      
│   │   ├── dataloader.py       # PyTorch DataLoader instantiation and setup
│   │   └── dataset.py          # Custom Dataset class for loading images & depth maps
│   ├── metrics/
│   │   ├── __init__.py      
│   │   ├── loss.py             # Edge-aware DepthLoss implementation (L1_log1p + alpha * L1_grad)
│   │   └── sample_plots.py     # Visualization utils for depth predictions vs ground truth
│   ├── models/
│   │   ├── __init__.py      
│   │   └── models.py           # MonocularDepthModel architecture (ResNet + Decoder)
│   ├── pipeline/
│   │   ├── __init__.py      
│   │   └── train.py            # Training loop, validation logic & checkpoint handling
│   └── __init__.py
│
├── .dockerignore               # Excluded build contexts
├── .gitignore                  # Excluded cache
├── Dockerfile                  # Docker container recipe
├── LICENSE                     # MIT License
├── pyproject.toml              # Project metadata & tool settings
├── README.md                   # Project documentation
├── requirements.txt            # Production dependencies
└── requirements-dev.txt        # Development & testing tools
```

## 🚀 Quickstart with Docker

### 1. Build Image
```bash
docker build -t depth-estimator-api .
```

### 2. Run Container
```bash
docker run -d -p 8000:8000 --name depth_service depth-estimator-api
```

Once running, the service will be accessible at **`http://localhost:8000`**

---

## 📡 API Usage

### Interactive API Docs (Swagger UI)
Swagger documentation is available at **`http://localhost:8000/docs`**

### Endpoints

#### GET /health
Performs a readiness check verifying whether the ONNX model artifact is correctly loaded into memory.

cURL Request Example:
```bash
curl -X 'GET' 'http://localhost:8000/health'
```

Response Example:
```json
{"status": "ok", "service": "depth-estimator"}
```

#### POST /predict
Accepts an image (image/jpeg, image/png) and returns the depth map bytes in PNG format.

cURL Request Example:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: image/png' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.jpg' \
  --output depth_map.png
```

Python Request Example:
```py
import requests

url = "http://localhost:8000/predict"
files = {"file": open("test_image.jpg", "rb")}

response = requests.post(url, files=files)

with open("depth_result.png", "wb") as f:
    f.write(response.content)
```

---

## 🧪 Loss Function & R&D

To eliminate blurred boundaries (a common artifact in monocular depth estimation), the model utilizes an additive edge-aware loss:

$$\mathcal{L}_{total} = \text{L1}\left(\ln(1 + \max(D_{pred}, \epsilon)), \, \ln(1 + \max(D_{target}, \epsilon))\right) + \alpha \cdot \left( \text{L1}(\nabla_x D_{pred}, \nabla_x D_{target}) + \text{L1}(\nabla_y D_{pred}, \nabla_y D_{target}) \right)$$

- **L1_log1p** — Guarantees global scene geometry and correct scale order.
- **L1_grad** — Penalizes blurry transitions along spatial derivatives.
- **alpha Hyperparameter:** Tuning alpha (0.5 -> 0.8-1.0) forces sharp object separation along boundaries.

---

## 🛠️ Tech Stack

- **Deep Learning / Frameworks:** PyTorch, ONNX Runtime, timm, Albumentations
- **Web / Backend:** FastAPI, Uvicorn, Pillow, NumPy
- **DevOps / Containerization:** Docker, WSL2 (Ubuntu)

---

## 📜 License
Distributed under the MIT License.