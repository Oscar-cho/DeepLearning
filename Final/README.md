# Image-to-Music Demo

`sam2_depth.ipynb` 실행용 안내 문서입니다.  
입력 이미지에서 사람을 분할하고, depth를 음높이에 매핑한 뒤, 프롬프트 기반으로 음악(MIDI/WAV)과 세그먼트 하이라이트 비디오(MP4)를 생성합니다.

## Project Installation

| 항목 | 내용 |
|---|---|
| 목표 | 이미지 이해 결과를 소리로 변환하는 인터랙티브 데모 |
| 핵심 입력 | `img/solvay.png` 등 단체/객체 이미지 |
| 핵심 출력 | `outputs/*.mid`, `outputs/*.wav`, `outputs/*_segments.mp4` |
| 데모 포인트 | 클릭 시 세그먼트 음 재생 + 10초 자동 생성 결과 |

## 모델별 역할

| 모델 | 왜 사용하는가 | 어떻게 동작하는가 |
|---|---|---|
| Grounding DINO | 사람 후보를 빠르게 찾기 위해 사용 | `person` 텍스트 조건으로 후보 박스 검출 |
| SAM2 | 사람 단위의 정밀 마스크가 필요해서 사용 | DINO 박스를 프롬프트로 받아 픽셀 단위 분할 |
| Depth Anything V2 | 거리 정보를 음높이 매핑 기준으로 사용 | depth map 추정 후 세그먼트 내부 평균 depth 계산 |
| QWEN3 | 사용자 프롬프트에 맞는 음악 순서를 만들기 위해 사용 | 텍스트를 `tempo/duration/recommended_order` JSON으로 변환 |

## Order

| 순서 | 노트북 단계 | 설명 |
|---|---|---|
| 1 | GPU/디바이스 설정 | 단일 GPU 고정 및 CUDA 확인 |
| 2 | 패키지 설치 | 필수 라이브러리 설치 확인 |
| 3 | 이미지 로드 | `img/` 폴더 이미지 로딩 |
| 4 | Grounding DINO | 사람 후보 박스 검출 |
| 5 | SAM2 | 사람별 마스크 생성 |
| 6 | Depth + 음계 매핑 | 세그먼트별 평균 depth -> note 매핑 |
| 7 | Interactive | 이미지 클릭 시 해당 세그먼트 음 재생 |
| 8 | QWEN Prompt-Aligned v2 | 10초 MIDI/WAV/MP4 자동 생성 |

## output

| 파일 패턴 | 설명 |
|---|---|
| `outputs/solvay_depth_segments.mid` | depth 기반 전체 세그먼트 MIDI |
| `outputs/qwen_v2_<prompt>.mid` | 프롬프트 반영 MIDI |
| `outputs/qwen_v2_<prompt>.wav` | 10초 WAV |
| `outputs/qwen_v2_<prompt>_segments.mp4` | 10초 세그먼트 하이라이트 영상 |

## installation

```bash
pip install -r requirements.txt
```

### System package (recommend)

| 항목 | 용도 |
|---|---|
| `ffmpeg` | 미디어 디버깅/변환 |
| `fluidsynth` + SoundFont(`.sf2`) | `pretty_midi.fluidsynth` 고품질 렌더링 |

> `fluidsynth`가 없으면 노트북은 sine fallback으로 WAV를 생성합니다.

## caution

| 항목 | 내용 |
|---|---|
| VRAM 부족 | QWEN3-8B 로딩 실패 시 fallback 모델/휴리스틱 사용 |
| 디스크 용량 | HF 캐시 다운로드 공간 필요(특히 8B 모델) |
| 인터랙티브 클릭 | 클릭 이벤트가 안 되면 `ipympl` 설치 후 커널 재시작 |

