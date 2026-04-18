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
git clone https://github.com/Oscar-cho/DeepLearning.git
cd DeepLearning
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
├── qwen3.ipynb                 # Vision-language chat (Qwen3.5)
├── sam2.ipynb                  # Image segmentation (SAM2)
├── grounding_dino.ipynb        # Object detection (Grounding DINO)
└── README.md
```

---

### 1. Qwen3.5 — Vision-Language Chat

**Model:** [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen)

Qwen3.5 is a multimodal large language model that accepts both image and text as input and produces detailed natural-language answers. This notebook demonstrates single-turn Q&A, batch inference over multiple images, and an interactive chat loop.

#### What it does

| Mode | Description |
|---|---|
| **Single image Q&A** | Ask one question about a selected image |
| **Batch Q&A** | Run the same question across all images in `img/` |
| **Interactive chat** | Fix one image and ask multiple questions in a loop |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | Input image (auto-resized to ≤ 1024 px) |
| `question` | `str` | Natural-language question about the image |
| `max_new_tokens` | `int` | Maximum response length (default: 256) |

#### Output

A natural-language answer string.

#### Example

```
Input image : xai506_example_image.jpg
Question    : "Describe this image in detail."

Answer: "The image shows a conference room with several people seated around
         a large table. There are laptops and documents on the table. A projector
         screen is visible in the background displaying a presentation slide."
```

---

### 2. SAM2 — Image Segmentation

**Model:** [`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large)

**Paper:** [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

SAM 2 (Segment Anything Model 2) is a general-purpose segmentation model from Meta AI. It accepts point or bounding-box prompts to isolate any object in an image, and can also run fully automatically without any prompts.

#### What it does

| Mode | Description |
|---|---|
| **Point prompt** | Click a point (foreground or background) to segment the surrounding object |
| **Box prompt** | Draw a bounding box `[x0, y0, x1, y1]` to isolate the enclosed region |
| **Automatic mask generation** | Segment all objects in the image without any human prompt |
| **Interactive loop** | Select an image and alternate between point/box prompts in the terminal |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | Input image |
| `point_coords` | `list[list[int]]` | `[[x, y], ...]` — pixel coordinates of prompt points |
| `point_labels` | `list[int]` | `[1, 0, ...]` — `1` = foreground, `0` = background |
| `box` | `list[int]` | `[x0, y0, x1, y1]` — bounding box in pixel coordinates |
| `multimask_output` | `bool` | If `True`, returns 3 mask candidates ranked by score |

#### Output

| Output | Shape | Description |
|---|---|---|
| `masks` | `(N, H, W)` | Binary segmentation masks |
| `scores` | `(N,)` | Confidence score per mask |

Results are visualized as a color overlay on the original image.

#### Example

```
Input image : desk.jpeg
Point prompt: [[300, 200]], label = [1]  (foreground)

Output:
  Mask 1: score=0.943  ← best candidate (highlighted in color overlay)
  Mask 2: score=0.871
  Mask 3: score=0.802
```

```
Input image : desk.jpeg
Box prompt  : [100, 80, 500, 400]

Output:
  Mask 1: score=0.981
```

---

### 3. Grounding DINO — Open-Vocabulary Object Detection

**Model:** [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base)

**Paper:** [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

Grounding DINO is an open-vocabulary object detector. Unlike traditional detectors that are limited to a fixed set of categories, Grounding DINO accepts an arbitrary natural-language text query and detects all matching objects in the image.

#### What it does

| Mode | Description |
|---|---|
| **Single image detection** | Detect specified categories in one image |
| **Batch detection** | Run detection with the same query across all images in `img/` |
| **Interactive loop** | Fix one image and repeatedly change the text query in the terminal |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | Input image |
| `text_query` | `str` | Categories to detect, separated by `". "` (e.g. `"person. chair. desk"`) |
| `box_threshold` | `float` | Minimum box confidence score (default: `0.25`) |
| `text_threshold` | `float` | Minimum text-matching score (default: `0.25`) |

#### Output

| Key | Type | Description |
|---|---|---|
| `boxes` | `Tensor (N, 4)` | Detected bounding boxes in `[x0, y0, x1, y1]` pixel coordinates |
| `scores` | `Tensor (N,)` | Confidence score per detection |
| `text_labels` | `list[str]` | Matched category label per detection |

Results are visualized as labeled bounding boxes drawn over the original image.

#### Example

```
Input image : xai506_example_image.jpg
Text query  : "person. chair. table. desk. coffee"

Output:
  [person]  score=0.72  box=[120, 45, 310, 480]
  [chair]   score=0.65  box=[400, 200, 590, 460]
  [table]   score=0.58  box=[80, 300, 650, 480]
  [coffee]  score=0.41  box=[210, 310, 270, 370]
```

---

## Notes

- All notebooks are configured to run on a specific GPU (`cuda:2` or `cuda:3`). Change the `DEVICE` variable at the top of each notebook to match your hardware setup.
- Images are automatically resized to a maximum of **1024 px** on their longest edge before inference.
- Model weights are downloaded automatically from Hugging Face Hub on the first run. Ensure you have a stable internet connection and sufficient disk space (~5–20 GB depending on the model).
