#!/usr/bin/env bash
set -euo pipefail

# Download ComfyUI Z-Image Turbo model files on a Vast.ai host.
# Usage:
#   ./download_zimage_turbo_models.sh
#   ./download_zimage_turbo_models.sh /path/to/ComfyUI
#
# Defaults to /workspace/ComfyUI, which is the common Vast.ai layout.

COMFYUI_DIR="${1:-${COMFYUI_DIR:-/workspace/ComfyUI}}"
MODELS_DIR="$COMFYUI_DIR/models"

TEXT_ENCODER_DOWNLOAD="text_encoders|qwen_3_4b.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
VAE_DOWNLOAD="vae|ae.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
OFFICIAL_DIFFUSION_DOWNLOAD="diffusion_models|z_image_turbo_bf16.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
NSFW_2602_DIFFUSION_DOWNLOAD="diffusion_models|2602_NSFW_ZIT_BSY_bf16.safetensors|https://huggingface.co/wiikoo/checkpoint/resolve/main/tongyi/2602_NSFW_ZIT_BSY_bf16.safetensors"
ZPENIS_V9_LORA_DOWNLOAD="loras|zpenis_v9_erect_limited_000033300.safetensors|https://huggingface.co/wiikoo/checkpoint/resolve/main/tongyi/zpenis_v9_erect_limited_000033300.safetensors"
SELECTED_DIFFUSION_NAME=""
SELECTED_DIFFUSION_DOWNLOAD=""
SELECTED_LORA_NAME=""
SELECTED_LORA_DOWNLOAD=""
HF_BIN=""
HF_TASK_DIR=""

cleanup_hf_cache() {
  if [[ -n "$HF_TASK_DIR" && -d "$HF_TASK_DIR" && "$HF_TASK_DIR" == "$MODELS_DIR"/.vastai-hf-* ]]; then
    rm -rf -- "$HF_TASK_DIR"
    echo "已清理本次 Hugging Face 下载缓存：$HF_TASK_DIR"
  fi
  HF_TASK_DIR=""
}

trap cleanup_hf_cache EXIT INT TERM

ensure_hf_cli() {
  if [[ -x /venv/main/bin/hf ]]; then
    HF_BIN=/venv/main/bin/hf
    return 0
  fi
  if command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
    return 0
  fi
  if [[ ! -x /venv/main/bin/python ]] || ! command -v uv >/dev/null 2>&1; then
    echo "错误：缺少 Hugging Face 官方 hf CLI，且无法通过 /venv/main + uv 自动安装。" >&2
    return 1
  fi
  echo "正在通过 /venv/main 安装 Hugging Face 官方 CLI..."
  uv pip --python /venv/main/bin/python --no-cache-dir install huggingface_hub
  HF_BIN=/venv/main/bin/hf
  [[ -x "$HF_BIN" ]]
}

normalize_choice() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-'
}

select_diffusion_model() {
  local choice="${ZIMAGE_TURBO_DIFFUSION_VARIANT:-}"

  if [[ -z "$choice" && -t 0 ]]; then
    echo "请选择 diffusion_models 模型来源："
    echo "  1. 官方 Z-Image-Turbo BF16"
    echo "  2. 2602 NSFW ZIT BSY BF16"
    read -r -p "请选择 [1-2，默认 1]: " choice
  fi

  choice="$(normalize_choice "$choice")"

  case "$choice" in
    ""|1|official|default|turbo)
      SELECTED_DIFFUSION_NAME="官方 Z-Image-Turbo BF16"
      SELECTED_DIFFUSION_DOWNLOAD="$OFFICIAL_DIFFUSION_DOWNLOAD"
      ;;
    2|2602|2602-nsfw|nsfw|tongyi|bsy)
      SELECTED_DIFFUSION_NAME="2602 NSFW ZIT BSY BF16"
      SELECTED_DIFFUSION_DOWNLOAD="$NSFW_2602_DIFFUSION_DOWNLOAD"
      ;;
    *)
      echo "无效选择：$choice，默认使用官方 Z-Image-Turbo BF16。"
      SELECTED_DIFFUSION_NAME="官方 Z-Image-Turbo BF16"
      SELECTED_DIFFUSION_DOWNLOAD="$OFFICIAL_DIFFUSION_DOWNLOAD"
      ;;
  esac
}

select_lora_model() {
  local choice="${ZIMAGE_TURBO_LORA_VARIANT:-}"

  if [[ -z "$choice" && -t 0 ]]; then
    echo "请选择是否下载 Z-Image-Turbo LoRA："
    echo "  1. 不下载 LoRA"
    echo "  2. zpenis v9 erect limited"
    read -r -p "请选择 [1-2，默认 1]: " choice
  fi

  choice="$(normalize_choice "$choice")"

  case "$choice" in
    ""|0|1|none|no|n|skip)
      SELECTED_LORA_NAME="不下载 LoRA"
      SELECTED_LORA_DOWNLOAD=""
      ;;
    2|yes|y|zpenis|zpenis-v9|lora)
      SELECTED_LORA_NAME="zpenis v9 erect limited"
      SELECTED_LORA_DOWNLOAD="$ZPENIS_V9_LORA_DOWNLOAD"
      ;;
    *)
      echo "无效选择：$choice，默认不下载 LoRA。"
      SELECTED_LORA_NAME="不下载 LoRA"
      SELECTED_LORA_DOWNLOAD=""
      ;;
  esac
}

download_file() {
  local url="$1"
  local target="$2"
  local rest repo remainder revision repo_file stage_dir staged_file

  if [[ -f "$target" ]]; then
    echo "跳过：已存在 $target"
    return 0
  fi

  rest="${url#https://huggingface.co/}"
  if [[ "$rest" == "$url" || "$rest" != */resolve/* ]]; then
    echo "错误：不支持的 Hugging Face 下载地址：$url" >&2
    return 1
  fi
  repo="${rest%%/resolve/*}"
  remainder="${rest#*/resolve/}"
  revision="${remainder%%/*}"
  repo_file="${remainder#*/}"
  stage_dir="$(mktemp -d "$HF_TASK_DIR/download-XXXXXX")"

  echo "使用 Hugging Face 官方 hf CLI 下载：$repo/$repo_file"
  HF_HOME="$HF_TASK_DIR/hf-home" \
  HF_HUB_CACHE="$HF_TASK_DIR/hf-home/hub" \
  HF_XET_CACHE="$HF_TASK_DIR/hf-home/xet" \
  HF_ASSETS_CACHE="$HF_TASK_DIR/hf-home/assets" \
    "$HF_BIN" download "$repo" "$repo_file" --revision "$revision" --local-dir "$stage_dir"

  staged_file="$stage_dir/$repo_file"
  if [[ ! -s "$staged_file" ]]; then
    echo "错误：hf download 未生成有效文件：$repo_file" >&2
    return 1
  fi
  mv "$staged_file" "$target"
  echo "完成：$target"
}

main() {
  echo "ComfyUI 目录：$COMFYUI_DIR"
  echo
  mkdir -p "$MODELS_DIR"
  HF_TASK_DIR="$(mktemp -d "$MODELS_DIR/.vastai-hf-XXXXXX")"
  ensure_hf_cli
  select_diffusion_model
  select_lora_model
  echo "已选择 diffusion 模型：$SELECTED_DIFFUSION_NAME"
  echo "已选择 LoRA：$SELECTED_LORA_NAME"
  echo

  DOWNLOADS=(
    "$TEXT_ENCODER_DOWNLOAD"
    "$SELECTED_DIFFUSION_DOWNLOAD"
    "$VAE_DOWNLOAD"
  )
  if [[ -n "$SELECTED_LORA_DOWNLOAD" ]]; then
    DOWNLOADS+=("$SELECTED_LORA_DOWNLOAD")
  fi

  for item in "${DOWNLOADS[@]}"; do
    IFS="|" read -r subdir filename url <<< "$item"
    target_dir="$MODELS_DIR/$subdir"
    target_file="$target_dir/$filename"

    mkdir -p "$target_dir"
    download_file "$url" "$target_file"
    echo
  done

  echo "Z-Image Turbo 模型文件检查完成。"
  cleanup_hf_cache
  trap - EXIT INT TERM
}

main "$@"
