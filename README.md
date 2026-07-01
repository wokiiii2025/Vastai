# Vast.ai 管理脚本

用于在 Vast.ai 实例上管理 Ollama、检查/升级 ComfyUI，并下载常用 ComfyUI 模型。

## 快速使用

在 Vast.ai 机器上直接运行：

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/wokiiii2025/Vastai/main/vastai_manager.sh)
```

如果当前 shell 不支持 `<(...)`，可以改用：

```bash
curl -fsSL https://raw.githubusercontent.com/wokiiii2025/Vastai/main/vastai_manager.sh | bash
```

远程执行时，入口脚本会自动下载 `vastai_manager.py` 到 `/tmp/vastai-manager/vastai_manager.py`，然后进入交互菜单。

本地克隆后运行：

```bash
git clone https://github.com/wokiiii2025/Vastai.git
cd Vastai
./vastai_manager.sh
```

## 依赖要求

基础依赖：

- `bash`
- `curl`
- `python3`

模型下载会优先使用 `aria2c`，没有时依次回退到 `curl`、`wget`。

ComfyUI 相关功能默认面向 Vast.ai 常见目录：

- ComfyUI: `/workspace/ComfyUI`
- ComfyUI 模型目录: `/workspace/ComfyUI/models`
- Ollama 工作目录: `/workspace/ollama`
- Ollama 默认建议端口: `40056`

## 主菜单

运行 `vastai_manager.sh` 后会显示：

```text
1. Ollama 服务与模型管理
2. Vast.ai ComfyUI 本机检查（关键目录/文件/版本/启动链路）
3. Vast.ai ComfyUI 本机升级（检查→备份→升级→补丁→原方式重启→验证）
4. 下载 ComfyUI Qwen-Rapid-AIO 模型（自动跳过已存在的）
5. 下载 ComfyUI Z-Image-Turbo 模型（diffusion/LoRA 可选，其他默认）
0. 退出
```

### Ollama 服务与模型管理

该菜单用于安装/更新 Ollama、配置端口、启动/停止服务，并导入默认 GGUF 模型：

- 模型名: `huihui-qwen3-4b:latest`
- GGUF 文件: `Huihui-Qwen3-4B-Instruct-2507-abliterated.Q8_0.gguf`
- 保存目录: `/workspace/ollama/models/gguf`

脚本会停掉官方安装脚本默认启动的 `ollama.service`，改用脚本配置的端口和模型目录启动 `ollama serve`。

### ComfyUI 检查与升级

菜单 2 用于检查 Vast.ai/ComfyUI 关键路径、Git 状态、Python 环境、启动链路和日志。

菜单 3 用于本机升级 ComfyUI。升级流程会：

- 检查 `/workspace/ComfyUI` 等关键路径
- 备份当前 ComfyUI 状态
- 更新 ComfyUI 代码和依赖
- 保留 Vast.ai 原有启动文件和启动参数
- 按原方式重启并验证日志

## ComfyUI 模型下载

所有 ComfyUI 模型下载都会自动跳过已存在的目标文件。

### Qwen-Rapid-AIO

菜单 4 会下载：

```text
/workspace/ComfyUI/models/checkpoints/Qwen-Rapid-AIO-NSFW-v17.safetensors
```

来源：

```text
https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v17/Qwen-Rapid-AIO-NSFW-v17.safetensors
```

### Z-Image-Turbo

菜单 5 仅使用 Hugging Face 源，不需要 `CIVITAI_API_TOKEN`。

默认下载：

```text
/workspace/ComfyUI/models/text_encoders/qwen_3_4b.safetensors
/workspace/ComfyUI/models/vae/ae.safetensors
```

`diffusion_models` 模型会在运行时二选一，并保留来源文件的原始文件名：

```text
1. 官方 Z-Image-Turbo BF16
   保存到 /workspace/ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors
   https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors

2. 2602 NSFW ZIT BSY BF16
   保存到 /workspace/ComfyUI/models/diffusion_models/2602_NSFW_ZIT_BSY_bf16.safetensors
   https://huggingface.co/wiikoo/checkpoint/resolve/main/tongyi/2602_NSFW_ZIT_BSY_bf16.safetensors
```

LoRA 可选下载，默认不下载：

```text
1. 不下载 LoRA

2. zpenis v9 erect limited
   保存到 /workspace/ComfyUI/models/loras/zpenis_v9_erect_limited_000033300.safetensors
   https://huggingface.co/wiikoo/checkpoint/resolve/main/tongyi/zpenis_v9_erect_limited_000033300.safetensors
```

默认模型对应来源：

```text
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
```

也可以只运行独立下载脚本：

```bash
./download_zimage_turbo_models.sh
```

指定 ComfyUI 目录：

```bash
./download_zimage_turbo_models.sh /workspace/ComfyUI
```

或通过环境变量指定：

```bash
COMFYUI_DIR=/workspace/ComfyUI ./download_zimage_turbo_models.sh
```

自动化时可以通过环境变量跳过交互选择：

```bash
ZIMAGE_TURBO_DIFFUSION_VARIANT=official python3 vastai_manager.py download-zimage-turbo-models
ZIMAGE_TURBO_DIFFUSION_VARIANT=2602-nsfw python3 vastai_manager.py download-zimage-turbo-models
ZIMAGE_TURBO_DIFFUSION_VARIANT=2602-nsfw ZIMAGE_TURBO_LORA_VARIANT=zpenis-v9 python3 vastai_manager.py download-zimage-turbo-models
ZIMAGE_TURBO_DIFFUSION_VARIANT=2602-nsfw ZIMAGE_TURBO_LORA_VARIANT=zpenis-v9 ./download_zimage_turbo_models.sh
```

如果目标文件已存在，脚本会按“自动跳过已存在的”规则跳过，不会覆盖。需要重新下载某个模型时，先自行删除对应目标文件再重新运行。

## 命令行模式

`vastai_manager.py` 支持直接执行具体动作，适合自动化：

```bash
python3 vastai_manager.py ollama-menu
python3 vastai_manager.py vast-comfyui-check
python3 vastai_manager.py vast-comfyui-upgrade
python3 vastai_manager.py download-qwen-rapid-model
python3 vastai_manager.py download-zimage-turbo-models
```

Ollama 服务相关：

```bash
python3 vastai_manager.py serve
python3 vastai_manager.py stop
python3 vastai_manager.py init-model
```

可通过参数覆盖 Ollama 端口和监听地址：

```bash
python3 vastai_manager.py serve --port 40056 --host 0.0.0.0
```

ComfyUI 检查/升级默认在本机执行，也保留可选 SSH 参数用于远程调试：

```bash
python3 vastai_manager.py vast-comfyui-check --ssh-host 1.2.3.4 --ssh-port 22
python3 vastai_manager.py vast-comfyui-upgrade --ssh-host 1.2.3.4 --ssh-port 22
```

## 环境变量

常用：

- `GITHUB_RAW_BASE`: 覆盖远程入口下载 `vastai_manager.py` 的 raw 根地址，默认是 `https://raw.githubusercontent.com/wokiiii2025/Vastai/main`
- `COMFYUI_DIR`: 仅独立 `download_zimage_turbo_models.sh` 使用，默认 `/workspace/ComfyUI`
- `ZIMAGE_TURBO_DIFFUSION_VARIANT`: 指定 Z-Image-Turbo diffusion 模型来源，可选 `official` 或 `2602-nsfw`
- `ZIMAGE_TURBO_LORA_VARIANT`: 指定是否下载 Z-Image-Turbo LoRA，可选 `none` 或 `zpenis-v9`

兼容旧命令：

- `CIVITAI_API_TOKEN`: 仅旧的 `download-zimage-models` 命令需要；当前主菜单已不再提供该入口

## 注意事项

- 不要在生产实例上盲目执行升级；先用菜单 2 做检查。
- 模型文件较大，建议 Vast.ai 实例磁盘空间充足后再下载。
- 如果端口冲突，先在 Ollama 菜单里重新配置端口，不要直接杀不明确的进程。
- 如果远程 `curl` 失败，先检查网络、DNS 和 GitHub raw 是否可访问。
