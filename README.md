# VLMEvalKit Fork - Multimodal Workload Evaluation for Thesis Research

## Overview

This repository is a fork of the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) toolkit, customized for thesis research on **system-level evaluation of multimodal workloads**. It was utilized as part of the ***[MSc-Thesis: System-Level Modality Reduction for Efficient Multimodal Large Language Model Serving](https://github.com/DalianisIoannis/MSc-Thesis-System-Level-Modality-Reduction-for-Efficient-Multimodal-Large-Language-Model-Serving/tree/main)*** project. The focus of this work is evaluating how Vision-Language Models (VLMs) perform under different input modality alterations, specifically:

- **Image compression** at various quality levels
- **Video frame sampling techniques** (uniform, scene change detection, motion-based, sharpness-based)

The system-level performance evaluation was conducted in a separate project, while this repository focuses on the **accuracy and quality metrics** of VLM responses to altered input modalities.

## Research Objectives

The primary goal is to understand:
1. How image compression affects VLM performance on multiple-choice and captioning tasks
2. Which video frame sampling strategies are most effective for different video understanding benchmarks
3. Whether intelligent frame sampling can maintain accuracy while reducing computational load
4. The feasibility of running large multimodal models with memory optimizations (vLLM)

## Key Modifications & Extensions

### 1. **Image Compression Experiments**

Created compressed dataset variants at 10 different quality levels (0-90% compression) for:
- **MMBench_DEV_EN**: Multiple-choice visual reasoning benchmark
- **COCO_VAL**: Image captioning benchmark
- **LLaVABench**: Visual question answering benchmark

Dataset naming convention: `{DATASET}_bdp_lan_rgb_{LEVEL}` where LEVEL ranges from 00 to 09.

**Modified files:**
- [vlmeval/dataset/image_base.py](vlmeval/dataset/image_base.py) - Added compressed dataset variants
- [vlmeval/dataset/image_mcq.py](vlmeval/dataset/image_mcq.py) - Multiple-choice dataset URLs
- [vlmeval/dataset/image_vqa.py](vlmeval/dataset/image_vqa.py) - VQA dataset support

### 2. **Video Frame Sampling Techniques**

Implemented four intelligent sampling strategies for video understanding tasks:

| Technique | Description | Key Parameter |
|-----------|-------------|---------------|
| **Uniform** | Evenly spaced frames (baseline) | Number of frames |
| **Scene Change** | Frames at scene boundaries | Threshold (27 default) |
| **Motion-based** | Frames with high motion | Motion score threshold (1 default) |
| **Sharpness** | Sharpest frames from uniform grid | Blur threshold (100 default) |

**Modified files:**
- [vlmeval/vlm/pixtral.py](vlmeval/vlm/pixtral.py) - Pixtral model with clever sampling support
- [vlmeval/vlm/qwen2_vl/model.py](vlmeval/vlm/qwen2_vl/model.py) - Qwen2-VL clever sampling
- [vlmeval/vlm/llava/llava.py](vlmeval/vlm/llava/llava.py) - LLaVA model sampling support
- [vlmeval/inference_video.py](vlmeval/inference_video.py) - Video inference pipeline
- [giannis_stuff/giannis_utils.py](giannis_stuff/giannis_utils.py) - Sampling utility functions

### 3. **vLLM Integration for Pixtral**

Enabled vLLM inference for the Pixtral-12B model to reduce memory footprint and enable execution on resource-constrained hardware.

**Key changes:**
- Modified scheduler to handle large outputs without premature termination
- Added batch processing support
- Implemented smart image resizing to fit within model constraints

**Modified files:**
- [vlmeval/vlm/pixtral.py](vlmeval/vlm/pixtral.py)
- External vLLM library modifications documented in [trails.ipynb](giannis_stuff/trails.ipynb)

### 4. **Local Judge Model Integration**

Replaced paid OpenAI API-based evaluation with a local Llama-2-7B judge model to:
- Reduce evaluation costs
- Enable offline evaluation
- Support vLLM acceleration for faster judging

**Modified files:**
- [vlmeval/api/hf_chat_model.py](vlmeval/api/hf_chat_model.py) - HuggingFace chat model wrapper with vLLM support
- [vlmeval/dataset/utils/judge_util.py](vlmeval/dataset/utils/judge_util.py) - Judge model selection logic
- [vlmeval/dataset/utils/llavabench.py](vlmeval/dataset/utils/llavabench.py) - Custom prompts and parsing for Llama judge

### 5. **Custom Datasets**

Added support for custom video multiple-choice dataset:
- **LLaVA-Video-Multiple-Choice**: Custom evaluation dataset

**Files:**
- [vlmeval/dataset/llava_video_dataset.py](vlmeval/dataset/llava_video_dataset.py)
- [vlmeval/dataset/video_dataset_config.py](vlmeval/dataset/video_dataset_config.py)

## Models Evaluated

- **Qwen2-VL-7B-Instruct** & **Qwen2-VL-2B-Instruct**
- **LLaVA-OneVision Qwen2 7B** & **0.5B**
- **Pixtral-12B** (with and without vLLM)

## Datasets Used

### Image Benchmarks
- **MMBench_DEV_EN**: Multiple-choice visual reasoning (350 samples)
- **COCO_VAL**: Image captioning (350 samples)
- **LLaVABench**: Visual question answering

### Video Benchmarks
- **Video-MME**: Video multiple-choice QA (350 samples)
- **MMBench-Video**: Video-based reasoning
- **TempCompass**: Temporal captioning

## Repository Structure

```
VLMEvalKit/
├── giannis_stuff/              # Thesis-specific code and analysis
│   ├── giannis_utils.py        # Utility functions (sampling, metrics, plotting)
│   ├── trails.ipynb            # Experimental notebook with detailed notes
│   ├── run_video_configs.sh   # Batch evaluation script
│   └── plots/                  # Generated visualizations
├── vlmeval/                    # Modified VLMEvalKit core
│   ├── vlm/                    # Model implementations (Pixtral, Qwen, LLaVA)
│   ├── dataset/                # Dataset loaders and processors
│   ├── api/                    # API wrappers (including local judge)
│   └── inference_video.py      # Video inference pipeline
├── requirements/               # Environment-specific dependencies
│   ├── image_multiple_choice.txt
│   ├── llava_video_kind.txt
│   └── image_pixtral.txt
├── outputs/                    # Evaluation results (not in repo)
└── config.json                 # Evaluation configuration
```

## Installation & Setup

### Environment Setup

Three separate virtual environments were used for different model types:

```bash
# 1. Image Multiple Choice (Qwen models)
python3 -m venv .env_image_mc
source .env_image_mc/bin/activate
pip install -e .
pip install -r requirements/image_multiple_choice.txt

# 2. LLaVA Video models
python3 -m venv .env_llava_video
source .env_llava_video/bin/activate
pip install -e .
pip install -r requirements/llava_video_kind.txt

# 3. Pixtral with vLLM
python3 -m venv .env_pixtral_vllm
source .env_pixtral_vllm/bin/activate
pip install -e .
pip install -r requirements/image_pixtral.txt
pip install vllm
```

### HuggingFace Authentication

Some models require HuggingFace authentication:

```bash
huggingface-cli login
```

Store token in: `/path/to/data/.cache/huggingface/token`

## Usage

### Single Evaluation Run

```bash
python run.py \
    --data MMBench_DEV_EN \
    --model Qwen2-VL-7B-Instruct \
    --verbose \
    --work-dir /path/to/outputs
```

### Image Compression Evaluation

```bash
python run.py \
    --data MMBench_DEV_EN_bdp_lan_rgb_05 \
    --model Pixtral-12B \
    --verbose \
    --work-dir /path/to/outputs
```

### Video with Clever Sampling

```bash
python run.py \
    --data Video-MME_64frame \
    --model llava_onevision_qwen2_7b_ov \
    --clever_sampling scene_change \
    --max_frames 64 \
    --sampling_extra_param 27 \
    --verbose \
    --work-dir /path/to/outputs
```

### Batch Video Evaluation

```bash
bash giannis_stuff/run_video_configs.sh
```

This script systematically evaluates:
- Multiple models
- All sampling techniques
- Different frame counts
- All video datasets

### Using Configuration Files

```bash
python run.py \
    --config config.json \
    --verbose \
    --work-dir /path/to/outputs
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | Dataset name | Required |
| `--model` | Model name | Required |
| `--work-dir` | Output directory | `./` |
| `--clever_sampling` | Sampling technique: `scene_change`, `motion_based`, `sharpness` | None (uniform) |
| `--max_frames` | Maximum video frames | Dataset default |
| `--sampling_extra_param` | Technique-specific threshold | Varies |
| `--reuse` | Reuse cached results | False |
| `--verbose` | Verbose logging | False |

## Important Implementation Notes

### 1. Flash Attention Compatibility

Some models require disabling flash attention. Modified:
```python
# .env/lib/python3.11/site-packages/llava/model/builder.py
model = LlavaQwenForCausalLM.from_pretrained(
    model_path, 
    low_cpu_mem_usage=True,
    # attn_implementation=attn_implementation,  # Commented out
    **kwargs, 
    trust_remote_code=False
)
```

### 2. vLLM Scheduler Fix

Modified vLLM scheduler to prevent early termination on large outputs:
```python
# vllm/v1/core/sched/scheduler.py line 819
if request.num_computed_tokens > max_cache_tkns - 32:
    stopped = True
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
```

### 3. Dataset MD5 Checks

Commented out MD5 validation for custom compressed datasets:
- [vlmeval/dataset/image_base.py](vlmeval/dataset/image_base.py) lines 116-126
- [vlmeval/dataset/tempcompass.py](vlmeval/dataset/tempcompass.py) line 288
- [vlmeval/dataset/videomme.py](vlmeval/dataset/videomme.py) - various locations

### 4. Pixtral Frame Resizing

Pixtral requires frame dimension resizing to handle 64 frames:
```python
# Smart resize based on max_pixels constraint
# Implemented in vlmeval/vlm/pixtral.py
```

### 5. Cached Frame Storage

Sampled video frames are cached to avoid redundant processing:
- Uniform sampling: `/path/to/data/LMUData/images/{DATASET}/{VIDEO_ID}/frame-{N}-of-{TOTAL}.jpg`
- Clever sampling: `/path/to/data/.cache/huggingface/hub/datasets--{DATASET}/video_frames/{VIDEO_ID}/frame-{N}-{TECHNIQUE}-{PARAM}.jpg`

## Results & Analysis

Results are organized in timestamped directories:
```
outputs/
└── {MODEL_NAME}/
    └── {TIMESTAMP}/
        ├── {DATASET}.xlsx        # Detailed predictions
        └── {DATASET}_eval.csv    # Evaluation metrics
```

Analysis notebooks and plotting utilities are in [giannis_stuff/trails.ipynb](giannis_stuff/trails.ipynb).

## Utilities & Helper Functions

The [giannis_utils.py](giannis_stuff/giannis_utils.py) module provides:

- **Frame sampling**: `apply_clever_sampling()`, scene change detection, motion analysis, sharpness scoring
- **Data processing**: TSV/JSONL converters, dataset subsetting
- **Evaluation**: Accuracy calculators, metric scorers (ROUGE, BLEU, CIDEr)
- **Visualization**: CDF plots, frame size analysis, result comparisons
- **Path management**: Centralized path configuration

## Known Issues & Limitations

1. **Pixtral + LLaVABench**: When using vLLM, run inference first with vLLM, then evaluate without vLLM to avoid CUDA OOM
2. **Chinese responses**: Qwen2-VL-2B occasionally responds in Chinese for open-ended questions
3. **Frame caching**: Be careful mixing models that resize frames (Pixtral) with those that don't (Qwen, LLaVA)
4. **Judge model**: Local Llama judge may not match GPT-4 quality; results require manual review for edge cases
5. **Environment isolation**: Different models require different transformer versions - use separate virtual environments

## Environment Variables

```bash
# Data paths
export HF_HOME=/path/to/huggingface/cache
export CACHE_MODEL_DIR=/path/to/model/cache

# CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Judge model
export LOCAL_LLM=/path/to/Llama-2-7b-chat-hf
```

## Citation

If you use this work, please cite the original VLMEvalKit:

```bibtex
@misc{duan2024vlmevalkit,
      title={VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models}, 
      author={Haodong Duan and Junming Yang and Yuxuan Qiao and Xinyu Fang and Lin Chen and Yuan Liu and Xiaoyi Dong and Yuhang Zang and Pan Zhang and Jiaqi Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.11691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11691}
}
```

## Acknowledgments

- Original VLMEvalKit team for the excellent evaluation framework
- vLLM team for memory-efficient inference
- Model providers: Mistral AI (Pixtral), Alibaba (Qwen), LLaVA team
- Dataset creators: MMBench, COCO, Video-MME, TempCompass teams

## Contact & Support

For questions about the thesis-specific modifications, refer to the detailed notes in [trails.ipynb](giannis_stuff/trails.ipynb).

For general VLMEvalKit issues, consult the [original repository](https://github.com/open-compass/VLMEvalKit).

---

**Note**: This is research code. Paths are configured for specific server environments and may require adjustment for your setup. See [trails.ipynb](giannis_stuff/trails.ipynb) for environment-specific configuration details.
