# Image-to-Music Demo (SAM2 + Grounding DINO + Depth Anything + QWEN3)


제 학번과 이름은 2025010653 조민규입니다.
`sam2_depth.ipynb` 하나로 실행되는 최종 데모 문서입니다.  
이미지 속 사람을 분할하고, 깊이값을 음높이에 매핑한 뒤, 텍스트 프롬프트를 반영해 10초짜리 음악/영상 결과를 생성합니다.

## Project Structure

```text
.
├── img/                             # 입력 이미지
│   └── solvay.png
├── sam2_depth.ipynb                 # 전체 파이프라인 노트북
├── requirements.txt                 # 실행 의존성
├── outputs/                         # 생성 결과
│   ├── solvay_depth_segments.mid
│   ├── qwen_v2_<prompt>.mid
│   ├── qwen_v2_<prompt>.wav
│   └── qwen_v2_<prompt>_segments.mp4
└── README.md
```

---

### 1. Grounding DINO — Open-Vocabulary Person Detection

**Model:** [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base)

Grounding DINO는 텍스트 질의 기반 객체 검출 모델입니다. 본 프로젝트에서는 `person` 중심 질의로 단체 사진 속 사람 후보 박스를 찾습니다.

#### What it does

| Mode | Description |
|---|---|
| **Person detection** | 이미지에서 사람 후보 박스를 다중 검출 |
| **Threshold/NMS filtering** | confidence/IoU 기준으로 중복 박스 정리 |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | 입력 이미지 |
| `text_prompt` | `str` | 예: `"person . man . woman . human ."` |
| `threshold` | `float` | 박스 confidence 임계값 |

#### Output

| Key | Type | Description |
|---|---|---|
| `boxes_nms` | `list[list[int]]` | 사람 박스 리스트 (`[x0,y0,x1,y1]`) |
| `scores_nms` | `list[float]` | 박스 confidence |

---

### 2. SAM2 — Prompted Instance Segmentation

**Model:** [`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large)

SAM2는 프롬프트 기반 범용 세그멘테이션 모델입니다. 여기서는 Grounding DINO 박스를 프롬프트로 사용해 사람별 마스크를 생성합니다.

#### What it does

| Mode | Description |
|---|---|
| **Box-guided segmentation** | 각 사람 박스를 픽셀 단위 마스크로 변환 |
| **Person-wise masks** | `person_segments` 형태로 후속 단계에 전달 |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | 입력 이미지 |
| `box` | `list[int]` | Grounding DINO가 생성한 사람 박스 |

#### Output

| Output | Shape/Type | Description |
|---|---|---|
| `mask` | `(H, W)` bool | 사람 인스턴스 마스크 |
| `person_segments` | `list[dict]` | 박스/점수/마스크 묶음 |

---

### 3. Depth Anything V2 — Depth-to-Pitch Mapping

**Model:** [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

Depth Anything V2는 단일 이미지 depth 추정 모델입니다. 사람 마스크 내부 평균 깊이를 계산해 음계 매핑의 기준값으로 사용합니다.

#### What it does

| Mode | Description |
|---|---|
| **Depth estimation** | 전체 이미지 depth map 생성 |
| **Segment depth pooling** | 마스크 내부 평균 depth 계산 후 정렬 |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `image` | `PIL.Image` | 입력 이미지 |
| `person_segments` | `list[dict]` | SAM2 사람이미지 마스크 |

#### Output

| Output | Type | Description |
|---|---|---|
| `depth_map` | `ndarray (H, W)` | 추정 depth map |
| `mean_depth` | `float` | 세그먼트 대표 깊이값 |
| `midi_note` | `int` | depth 정렬 기반 할당된 음높이 |

---

### 4. QWEN3 — Prompt-Aligned Music Sequencing

**Model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen)  
**Fallback:** `Qwen/Qwen2.5-0.5B-Instruct` + heuristic

QWEN3는 사용자 프롬프트를 음악 파라미터(JSON)로 변환합니다.  
이 값을 이용해 MIDI/WAV/세그먼트 하이라이트 MP4를 10초 길이로 생성합니다.
QWEN3 model이 download되지 않으면, QWEN2.5 0.5B Instruct model로 대체하여 진행됩니다.

#### What it does

| Mode | Description |
|---|---|
| **Prompt parsing** | 프롬프트 -> `tempo`, `duration`, `recommended_order` 등 생성 |
| **Sequence generation** | `P1..Pn` 토큰 순서를 음악 이벤트로 변환 |
| **Media rendering** | MIDI/WAV/MP4 결과 저장 |

#### Input

| Parameter | Type | Description |
|---|---|---|
| `user_sequence` | `str` | 예: `P1-P2-P6` (없으면 모델 추천) |
| `music_prompt` | `str` | 예: `"긴장감 있는 분위기"`, `"비행기"` |
| `segment_note_map` | `dict` | `P토큰 -> MIDI note` |

#### Output

| File | Description |
|---|---|
| `qwen_v2_<prompt>.mid` | 프롬프트 반영 MIDI |
| `qwen_v2_<prompt>.wav` | 10초 WAV |
| `qwen_v2_<prompt>_segments.mp4` | 10초 세그먼트 하이라이트 비디오 |

---

## Interactive Mode (Click-to-Play)

노트북의 `Interactive` 셀에서는 이미지를 클릭하면 해당 세그먼트 음을 즉시 재생할 수 있습니다.

| 동작 | 설명 |
|---|---|
| Click | 클릭 좌표가 포함된 마스크 탐색 |
| Audio | 선택 세그먼트의 `midi_note`를 즉시 재생 |

---

## Requirements

```bash
pip install -r requirements.txt
```

권장 시스템 패키지:

| Package | Purpose |
|---|---|
| `ffmpeg` | 미디어 확인/변환 |
| `fluidsynth` + `.sf2` | 고품질 MIDI->WAV 렌더링 |

> `fluidsynth`가 없으면 노트북은 sine fallback 방식으로 WAV를 생성합니다.

