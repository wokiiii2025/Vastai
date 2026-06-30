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

DOWNLOADS=(
  "text_encoders|qwen_3_4b.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
  "diffusion_models|z_image_turbo_bf16.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
  "diffusion_models|z-anime-distill-8step-bf16.safetensors|https://huggingface.co/SeeSee21/Z-Anime/resolve/main/diffusion_models/z-anime-distill-8step-bf16.safetensors"
  "vae|ae.safetensors|https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
)

download_file() {
  local url="$1"
  local target="$2"
  local partial="${target}.part"

  if [[ -f "$target" ]]; then
    echo "跳过：已存在 $target"
    return 0
  fi

  rm -f "$partial"
  echo "下载：$target"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server=8 \
      --split=8 \
      --min-split-size=64M \
      --dir "$(dirname "$target")" \
      --out "$(basename "$partial")" \
      "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 -o "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=3 -O "$partial" "$url"
  else
    echo "错误：未找到 aria2c、curl 或 wget，无法下载。" >&2
    return 1
  fi

  mv "$partial" "$target"
  echo "完成：$target"
}

main() {
  echo "ComfyUI 目录：$COMFYUI_DIR"
  echo

  for item in "${DOWNLOADS[@]}"; do
    IFS="|" read -r subdir filename url <<< "$item"
    target_dir="$MODELS_DIR/$subdir"
    target_file="$target_dir/$filename"

    mkdir -p "$target_dir"
    download_file "$url" "$target_file"
    echo
  done

  echo "Z-Image Turbo 模型文件检查完成。"
}

main "$@"
