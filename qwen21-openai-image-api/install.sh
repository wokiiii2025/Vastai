#!/usr/bin/env bash
# Install the Qwen-Image-2.1 model files for an existing Vast.ai ComfyUI.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash install.sh [--plan] [--comfyui-dir DIR] [--optional NAME]... [--optional-only]

Install the Qwen-Image-2.1 checkpoint, two text encoders, VAE, and three LoRAs (~28.3 GiB).
The default destination is /workspace/ComfyUI/models. Override it with
COMFYUI_DIR or --comfyui-dir. Set HF_TOKEN if Hugging Face requires authentication.
Set CIVITAI_API_TOKEN for Civitai downloads. An existing verified copy in
models/loras or QWEN21_MERGE_SOURCE (file path or HTTPS URL) can also supply
the 8-step LoRA.
Use --plan to list the download sources and destination paths.

Repeat --optional NAME to add any of these files:
  vagina-lora      Qwen 2.1 Vagina v2.0
  fineview-fp8     FineView STORY3 REDQW21 v2 TI2I, FP8
  fineview-int8c   FineView STORY3 REDQW21 v2 TI2I, INT8C
  coachbate-lora   CoachBate Qwen Image 2.1 v1
Use --optional-only with --optional to install just the selected extras.
HELP
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

comfyui_dir="${COMFYUI_DIR:-/workspace/ComfyUI}"
plan=false
optional_only=false
declare -A selected=()
while (( $# )); do
    case "$1" in
        --comfyui-dir)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || fail '--comfyui-dir requires a directory'
            comfyui_dir="$2"
            shift 2
            ;;
        --plan) plan=true; shift ;;
        --optional)
            case "${2:-}" in
                vagina-lora|fineview-fp8|fineview-int8c|coachbate-lora) selected["$2"]=yes ;;
                *) fail '--optional requires vagina-lora, fineview-fp8, fineview-int8c, or coachbate-lora' ;;
            esac
            shift 2
            ;;
        --optional-only) optional_only=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done
if "$optional_only" && [[ -z "${selected[*]-}" ]]; then
    fail '--optional-only requires at least one --optional NAME'
fi

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
    local url="$1" partial="$2" token=''
    local -a options=(
        --fail --location --proto '=https' --proto-redir '=https'
        --connect-timeout 20 --speed-limit 1024 --speed-time 60
        --continue-at - --output "$partial" --progress-bar --show-error
    )
    case "$url" in
        https://huggingface.co/*) token="${HF_TOKEN:-}" ;;
        https://civitai.com/api/download/*|https://civitai.red/api/download/*)
            token="${CIVITAI_API_TOKEN:-}"
            [[ -n "$token" ]] || fail 'Set CIVITAI_API_TOKEN to download model files from Civitai.'
            ;;
    esac
    if [[ -n "$token" ]]; then
        curl -q "${options[@]}" --header @- -- "$url" <<< "Authorization: Bearer $token"
    else
        curl -q "${options[@]}" -- "$url"
    fi
}

install_model() {
    local target="$1" bytes="$2" digest="$3" source="$4"
    local partial="${target%/*}/.${target##*/}.qwen21.part" attempt

    if [[ -e "$target" || -L "$target" ]]; then
        matches "$target" "$bytes" "$digest" || fail "Existing model failed verification: $target"
        printf 'Already verified: %s\n' "$target"
        return
    fi

    mkdir -p -- "${target%/*}"
    [[ ! -L "$partial" && ( ! -e "$partial" || -f "$partial" ) ]] || fail "Invalid download path: $partial"
    if [[ "$source" != https://* ]]; then
        [[ -f "$source" ]] ||
            fail "Provide QWEN21_MERGE_SOURCE as a local file or HTTPS download URL for ${target##*/}."
        matches "$source" "$bytes" "$digest" || fail "Source model failed verification: $source"
        printf 'Copying: %s\n' "$target"
        if [[ ! "$source" -ef "$partial" ]]; then
            cp -- "$source" "$partial"
        fi
    fi

    for attempt in 1 2 3 4; do
        if [[ -f "$partial" ]] && (( $(stat -c '%s' -- "$partial") >= bytes )); then
            break
        fi
        printf 'Downloading: %s (%s/4)\n' "$target" "$attempt"
        if download "$source" "$partial"; then
            break
        fi
        if (( attempt < 4 )); then
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

process_model() {
    local relative_path="$1" bytes="$2" digest="$3" url="$4"
    local target="$models_dir/$relative_path" source
    if "$plan"; then
        printf '%s (%s bytes)\n  %s\n' "$target" "$bytes" "$url"
    else
        source="$url"
        if [[ "$relative_path" == loras/QwenImage_8Step_merge.safetensors ]]; then
            source="${QWEN21_MERGE_SOURCE:-$url}"
        fi
        install_model "$target" "$bytes" "$digest" "$source"
    fi
}

if ! "$optional_only"; then
    while IFS='|' read -r relative_path bytes digest url; do
        process_model "$relative_path" "$bytes" "$digest" "$url"
    done <<'MODELS'
loras/QwenImage_8Step_merge.safetensors|637586048|3a18527010d95101746845d9ccf1a4bb7e5df090b144aefdaef1db6999f25f68|https://civitai.com/api/download/models/3362980?fileId=3250799
diffusion_models/NoctQ_V4_int8_convrot.safetensors|7256784368|4d92d5538253ab36e6a73b056cce5f697950f4303cb9e6686a3198d0ea13f9e8|https://huggingface.co/Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1/resolve/a81b9af51120a78e285e57906f2250a2a02080e9/NoctQ_V4_int8_convrot.safetensors
text_encoders/qwen3vl_8b_int8_convrot_heretic.safetensors|9350828392|f15ce4275428e04f42cdb59e3a253cb290cae5de99259527f01d3d2e51153653|https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot/resolve/033b4bf23f8816b22c065309bc9b349953193978/qwen3vl_8b_int8_convrot_heretic.safetensors
text_encoders/qwen3.5_9b_uncensored_int8_convrot.safetensors|12341908719|273a01295787728011797cfec0a62b43a877e30ff237d87ee238758381d66a2a|https://civitai.com/api/download/models/3416420?fileId=3309482
vae/qwen_image_2.1_vae_bf16.safetensors|675509688|bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9|https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/aacf31a6d48e707f352ecba714cee0680da3479b/vae/qwen_image_2.1_vae_bf16.safetensors
loras/qwen_image_2.1_lightning_4step.safetensors|67144280|53c69320d4163da51032f90275074028cb098556039c72b45f8dce9f0f9e34a3|https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/resolve/6b0a1176327c01522a44276c5da67331a025b6fc/pytorch_lora_weights.safetensors
loras/Qwen-Image-2.1 NSFW Image Edit.safetensors|83943552|9c143cbfe7b016efdce7499506c02d0bfa1842ae6524bb64e05888f2e325e67a|https://huggingface.co/mpasila/Qwen-Image-2.1-LoRAs/resolve/401c3423daed3d678eb7bb6477eb695ba0a01f4d/Qwen-Image-2.1%20NSFW%20Image%20Edit.safetensors
MODELS
fi

while IFS='|' read -r name relative_path bytes digest url; do
    if [[ -n "${selected[$name]-}" ]]; then
        process_model "$relative_path" "$bytes" "$digest" "$url"
    fi
done <<'OPTIONAL_MODELS'
vagina-lora|loras/pussyV2.safetensors|83943904|403121922c6dd013504e126baafa42e5afee49d85422c0cd122b29696063fa61|https://civitai.com/api/download/models/3377365?fileId=3265734
fineview-fp8|diffusion_models/REDQW21-UNLOCKED-v2-FP8-ComfyMCP-builtwithqwen.safetensors|7116801326|0efb5aeb2b372025042e320c2e35c66ec6681ef54ad5de88a652ab19cc63ad92|https://civitai.com/api/download/models/3370753?fileId=3258921
fineview-int8c|diffusion_models/REDQW21-UNLOCKED-v2-INT8C-ComfyMCP-builtwithqwen.safetensors|7256795720|30532b445eabd4b84b108168b6b3fe720535ef400f6bf9d68bd551817d3531a3|https://civitai.com/api/download/models/3370753?fileId=3258922
coachbate-lora|loras/qwen-image-2.1_penis_coachbate_preview1.safetensors|159436160|0f79fae880ba2b89960676e036dfee1fea6020f7acbf6d1a333dc14b477cdeeb|https://civitai.com/api/download/models/3344536?fileId=3232190
OPTIONAL_MODELS

if ! "$plan"; then
    printf 'Qwen-Image-2.1 model files are ready in %s\n' "$models_dir"
fi
