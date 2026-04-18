# DeepLearning
XAI506 Deeplearning Mideterm

2025010653 조민규

## Requirements

python ≥ 3.10
PyTorch ≥ 2.1 (CUDA build)
CUDA  ≥ 11.8
GPU VRAM 12 GB (per model)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install PyTorch with CUDA support

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your CUDA version, or use the command below for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install common dependencies

```bash
pip install transformers accelerate huggingface_hub
pip install Pillow matplotlib numpy
pip install jupyter ipykernel
```

### 5. Install model-specific dependencies

**For SAM2:**

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**For Grounding DINO:**

```bash
pip install transformers[torch]   # already covered above
```

### 6. Prepare input images

Place your images inside the `img/` directory. The notebooks look for `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files in that folder.

```
img/
├── your_image1.jpg
├── your_image2.png
└── ...
```

### 7. Launch Jupyter

```bash
jupyter notebook
```

---

## Project Structure

```
.
├── img/                        # Input images
│   ├── desk.jpeg
│   └── xai506_example_image.jpg
├── blip.ipynb                  # Image captioning (BLIP)
├── qwen3.ipynb                 # Vision-language chat (Qwen3.5)
├── sam2.ipynb                  # Image segmentation (SAM2)
├── grounding_dino.ipynb        # Object detection (Grounding DINO)
└── README.md


