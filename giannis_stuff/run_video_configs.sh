#!/bin/bash

# Batch evaluation script for video datasets with multiple sampling techniques
# This script runs systematic evaluations across models, datasets, and sampling strategies

# Models to evaluate
# models=("llava_onevision_qwen2_7b_ov" "llava_onevision_qwen2_0.5b_ov")
# models=("Qwen2-VL-7B-Instruct" "Qwen2-VL-2B-Instruct")
models=("Pixtral-12B")

# Frame counts to test
# max_frames_list=(4 8 16 32 64)
max_frames_list=(64)

# Sampling techniques and their optimal parameters
# Parameters were tuned based on empirical results
declare -A techniques
techniques["motion_based"]="1"     # Motion detection threshold
techniques["scene_change"]="27"    # Scene change sensitivity
techniques["sharpness"]="100"      # Blur detection threshold

# Video datasets to evaluate
datasets=("MMBench_Video_64frame_nopack" "Video-MME_64frame" "TempCompass_Captioning_64frame")

# Configure paths for your environment
VENV_PATH="$(pwd)/.env_pixtral_vllm"          # Virtual environment
PROJECT_PATH="$(pwd)"                          # Project root
OUTPUT_PATH="/srv/muse-lab/datasets/VLMEvalKitdata/outputs"  # Results directory

# Loop through all combinations
for model in "${models[@]}"; do
  for max_frames in "${max_frames_list[@]}"; do
    for technique in "${!techniques[@]}"; do
      for sampling_param in ${techniques[$technique]}; do
        for dataset in "${datasets[@]}"; do

          echo "========================================="
          echo "Model: $model"
          echo "Dataset: $dataset"
          echo "Frames: $max_frames"
          echo "Technique: $technique"
          echo "Parameter: $sampling_param"
          echo "========================================="

          "${VENV_PATH}/bin/python3" \
          "${PROJECT_PATH}/run.py" \
            --data "$dataset" \
            --model "$model" \
            --verbose \
            --work-dir "$OUTPUT_PATH" \
            --clever_sampling "$technique" \
            --max_frames "$max_frames" \
            --sampling_extra_param "$sampling_param" \
            --reuse
        done
      done
    done
  done
done

# Alternative virtual environments for different model types:
# .env_image_mc - For image multiple-choice (Qwen models)
# .env_llava_video - For LLaVA video models
# .env_pixtral_vllm - For Pixtral with vLLM support
# nohup ./giannis_stuff/run_video_configs.sh > ./giannis_stuff/batch_runner.log 2>&1 < /dev/null &