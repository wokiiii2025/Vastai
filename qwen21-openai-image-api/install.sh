#!/usr/bin/env bash
# Install the six Qwen-Image-2.1 model files for an existing Vast.ai ComfyUI.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--plan] [--comfyui-dir DIR]

Download the Qwen-Image-2.1 checkpoint, text encoder, VAE, and three LoRAs (~17.5 GiB).
The default destination is /workspace/ComfyUI/models. Override it with
COMFYUI_DIR or --comfyui-dir. Set HF_TOKEN if Hugging Face requires authentication.
Use --plan to list the download sources and destination paths.
HELP
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

comfyui_dir="${COMFYUI_DIR:-/workspace/ComfyUI}"
plan=false
while (( $# )); do
    case "$1" in
        --comfyui-dir)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || fail '--comfyui-dir requires a directory'
            comfyui_dir="$2"
            shift 2
            ;;
        --plan) plan=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done

models_dir="$comfyui_dir/models"
if ! "$plan"; then
    [[ -d "$comfyui_dir" ]] || fail "ComfyUI directory not found: $comfyui_dir (use --comfyui-dir)"
    for program in curl flock sha256sum stat; do
        command -v "$program" >/dev/null || fail "Required command not found: $program"
    done
    mkdir -p -- "$models_dir"
    exec 9>>"$models_dir/.qwen21-download.lock"
    flock -n 9 || fail "Another Qwen-Image-2.1 installation is using $models_dir"
fi

matches() {
    local file="$1" bytes="$2" expected="$3" actual
    [[ -f "$file" && "$(stat -Lc '%s' -- "$file")" == "$bytes" ]] || return 1
    read -r actual _ < <(sha256sum < "$file")
    [[ "$actual" == "$expected" ]]
}

download() {
    local url="$1" partial="$2"
    local -a options=(
        --fail --location --connect-timeout 20 --speed-limit 1024 --speed-time 60
        --continue-at - --output "$partial" --progress-bar --show-error
    )
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl -q "${options[@]}" --header @- -- "$url" <<< "Authorization: Bearer $HF_TOKEN"
    else
        curl -q "${options[@]}" -- "$url"
    fi
}

install_model() {
    local target="$1" bytes="$2" digest="$3" url="$4"
    local partial="${target%/*}/.${target##*/}.qwen21.part" attempt

    if [[ -e "$target" || -L "$target" ]]; then
        matches "$target" "$bytes" "$digest" || fail "Existing model failed verification: $target"
        printf 'Already verified: %s\n' "$target"
        return
    fi

    mkdir -p -- "${target%/*}"
    [[ ! -L "$partial" && ( ! -e "$partial" || -f "$partial" ) ]] || fail "Invalid download path: $partial"
    printf 'Downloading: %s\n' "$target"
    for attempt in 1 2 3 4; do
        if [[ -f "$partial" ]] && (( $(stat -c '%s' -- "$partial") >= bytes )); then
            break
        fi
        if download "$url" "$partial"; then
            break
        fi
        if (( attempt < 4 )); then
            printf 'Retrying download (%s/4): %s\n' "$((attempt + 1))" "$target" >&2
            sleep "$((attempt * 2))"
        fi
    done

    [[ -f "$partial" ]] && (( $(stat -c '%s' -- "$partial") >= bytes )) ||
        fail "Download incomplete: $target. Run again to resume."
    if ! matches "$partial" "$bytes" "$digest"; then
        rm -- "$partial"
        fail "Download failed size or SHA-256 verification: $target. Run again to download a fresh copy."
    fi
    mv -nT -- "$partial" "$target"
    [[ ! -e "$partial" ]] || fail "Destination appeared during download: $target"
    printf 'Installed: %s\n' "$target"
}

while IFS='|' read -r relative_path bytes digest url; do
    target="$models_dir/$relative_path"
    if "$plan"; then
        printf '%s (%s bytes)\n  %s\n' "$target" "$bytes" "$url"
    else
        install_model "$target" "$bytes" "$digest" "$url"
    fi
done <<'MODELS'
diffusion_models/NoctQ_V4_int8_convrot.safetensors|7256784368|4d92d5538253ab36e6a73b056cce5f697950f4303cb9e6686a3198d0ea13f9e8|https://huggingface.co/Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1/resolve/a81b9af51120a78e285e57906f2250a2a02080e9/NoctQ_V4_int8_convrot.safetensors
text_encoders/qwen3vl_8b_int8_convrot_heretic.safetensors|9350828392|f15ce4275428e04f42cdb59e3a253cb290cae5de99259527f01d3d2e51153653|https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot/resolve/033b4bf23f8816b22c065309bc9b349953193978/qwen3vl_8b_int8_convrot_heretic.safetensors
vae/qwen_image_2.1_vae_bf16.safetensors|675509688|bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9|https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/aacf31a6d48e707f352ecba714cee0680da3479b/vae/qwen_image_2.1_vae_bf16.safetensors
loras/qwen_image_2.1_lightning_4step.safetensors|67144280|53c69320d4163da51032f90275074028cb098556039c72b45f8dce9f0f9e34a3|https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/resolve/6b0a1176327c01522a44276c5da67331a025b6fc/pytorch_lora_weights.safetensors
loras/turbo8_lora_step2500.safetensors|1359148744|a0b7a7634ff2f82ccbe209e16b0ba19b25e2d9e18624d89edc490503b6a75d39|https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1/resolve/611a3f86d0cc53d21a1a09d07ea7dc82df3d68a1/turbo8_lora_step2500.safetensors
loras/Qwen-Image-2.1 NSFW Image Edit.safetensors|83943552|9c143cbfe7b016efdce7499506c02d0bfa1842ae6524bb64e05888f2e325e67a|https://huggingface.co/mpasila/Qwen-Image-2.1-LoRAs/resolve/401c3423daed3d678eb7bb6477eb695ba0a01f4d/Qwen-Image-2.1%20NSFW%20Image%20Edit.safetensors
MODELS

if ! "$plan"; then
    printf 'All six Qwen-Image-2.1 model files are ready in %s\n' "$models_dir"
fi
