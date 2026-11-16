# Vast.ai Qwen-Image 2.1 文生图与图生图

这个目录复现现网的 Qwen-Image 2.1 推理栈。它安装固定版本的 ComfyUI、六个模型文件、原生工作流节点和 OpenAI 兼容图像 API。API 源码对应 Qwen-Image-Api 提交 `e04eb1d`，包含单图和多参考图编辑、4/8 步、同步和异步请求，以及按 ComfyUI 后端配置的热载别名。

## 一键部署

在有 NVIDIA GPU 的 Vast.ai Linux 实例中运行。需要 Python 3.12（含 `venv`）、Git、`nvidia-smi` 和可用的 Supervisor。固定环境是 PyTorch `2.7.1+cu126` 和 [ComfyUI `157db2e`](https://github.com/Comfy-Org/ComfyUI/commit/157db2ea2f835917c16191147c11e5896eca01a9)，现网验证硬件为 A100 80 GiB。模型共约 17.5 GiB，安装前建议预留至少 35 GiB，另计实际输出和缓存。

```bash
git clone https://github.com/wokiiii2025/Vastai.git
cd Vastai/qwen21-openai-image-api
bash install.sh --prewarm
```

默认安装到 `/workspace/qwen21-image`，ComfyUI 和 API 分别监听 `127.0.0.1:18188`、`127.0.0.1:18080`。Supervisor 服务名分别是 `qwen21-comfyui`、`qwen21-images-api`。新安装时生成 API Bearer Key，可在工作目录的 `service/.runtime.json` 中查看，权限为 `0600`。重复运行会复用这个 Key，也会重新验证已安装的模型文件。

- `bash install.sh --plan` 查看安装计划，`--prepare-only` 安装文件并生成配置，便于稍后启动服务。
- `--root`、`--comfy-port` 和 `--api-port` 可指定独立目录和可用端口。Supervisor 配置默认位于 `/etc/supervisor/conf.d`，可通过 `--supervisor-dir` 指定实例已有的配置目录。
- 已有验证过的权重可用 `--model-dir /path/to/models`。目录结构与 `models/` 一致。下载或本地导入都会比对大小和 SHA-256；已有文件校验失败会停止并保留原文件。
- Hugging Face 需要认证时，在当前部署会话设置只读 `HF_TOKEN`。`models.lock.json` 固定了六个仓库文件、revision、文件大小和 SHA-256。
- `--prewarm` 会走真实 API 各出一张 512×512 PNG，验证四档实际采样和 CFG，再把对应权重留在可用显存里。可以稍后在项目目录运行 `/workspace/qwen21-image/.venv/bin/python verify.py --prewarm`，重启 ComfyUI 后也可用同样命令预热。

API 文档在 `/docs`，健康检查在 `/health`。本地调用时，把 `service/.runtime.json` 的 `api_key` 用作 `Authorization: Bearer …`。外部访问可通过现有 Vast Portal 或 HTTPS 反向代理，并用 `--public-base-url https://images.example.com` 指定签名图像链接和异步轮询链接的公开根地址。若平台要求服务直接监听映射端口，使用 `--api-host 0.0.0.0 --api-port <实例映射的容器端口>`。

## 现网模型

| 用途 | 文件 | 放置位置 |
|---|---|---|
| 扩散底模 | `NoctQ_V4_int8_convrot.safetensors` | `models/diffusion_models/` |
| Qwen3-VL 文本编码器 | `qwen3vl_8b_int8_convrot_heretic.safetensors` | `models/text_encoders/` |
| 图像 VAE | `qwen_image_2.1_vae_bf16.safetensors` | `models/vae/` |
| Lightning 4 步 LoRA | `qwen_image_2.1_lightning_4step.safetensors` | `models/loras/` |
| Turbo8 8 步 LoRA | `turbo8_lora_step2500.safetensors` | `models/loras/` |
| 图生图编辑 LoRA | `Qwen-Image-2.1 NSFW Image Edit.safetensors` | `models/loras/` |

上游模型按各自条款分发。这套安装脚本使用公开来源和固定校验值。

## 节点与采样

- 文生图 `t2i-v1` 使用 `UNETLoader`、`CLIPLoader`、`VAELoader`、`TextEncodeQwenImage21`、`LoraLoaderModelOnly`、`ModelSamplingFlux`、`EmptyLatentImage`、`KSampler`、`VAEDecode` 和 `SaveImage`。
- 图生图 `i2i-edit-v1` 叠加编辑 LoRA，将 `LoadImage` 的参考图传给 `TextEncodeQwenImage21`。显式尺寸时首图经过 `ImageScale`，`size=auto` 时使用首图尺寸。采样使用编码节点生成的 latent。多参考图按顺序传入 `images.image_1` 到 `images.image_10`，API 请求中重复 multipart 的 `image` 字段。
- 两条接口均使用 **CFG 2.0、Euler、simple、denoise 1.0**，默认 8 步。4/8 步分别加载对应加速 LoRA，强度均为 1.0。流移位 `base_shift=0.5`，4 步 `max_shift=0.9`，8 步 `max_shift=0.6935`。在 [workflows](workflows/) 中可查看四档的 ComfyUI API 格式示例。

所用节点都随固定版本 ComfyUI 提供。专用实例启动时禁用外部 API 节点和第三方自定义节点。

## 热载与显存

实例默认使用 PyTorch attention 和关闭采样预览。现网的四档热载使用不同的 UNET 名称，符号链接都指向同一底模文件。ComfyUI 在显存充足时可以分别保留四个 LoRA 实例，切换文生图、图生图和 4/8 步时减少重载开销。四档全热时，现网 A100 80 GiB 的空闲显存占用约 45 GiB。

`--warm-models auto` 默认在首张可见 GPU 显存至少 72 GiB 时打开别名。共享 GPU 或经常出大图时，可以用 `--warm-models off` 回到共用单一底模的配置。较小显存机器可按实际负载评估 `--warm-models on`。程序重启后模型需要重新加载。现网 704×512 热载测试的 API 耗时约为文生图 4/8 步 6.9/11.2 秒，图生图 4/8 步 11.5/15.1 秒。

## API 独立 Docker 部署

[api](api/) 是同版本 API 的 Docker 部署目录。把 `api/.env.example` 复制为 `api/.env`，设置客户端 Key 和 ComfyUI 地址后，在该目录运行 `docker compose up -d --build`。如果连接远程 ComfyUI，每台服务的地址和认证方式在 `COMFY_BACKENDS` 的独立条目里配置。为支持热载的服务器加入下面的可选字段，名称须能由该机器的 `UNETLoader` 找到。

```json
{
  "model_overrides": {
    "t2i_4": "qwen-warm/t2i-4.safetensors",
    "t2i_8": "qwen-warm/t2i-8.safetensors",
    "edit_4": "qwen-warm/edit-4.safetensors",
    "edit_8": "qwen-warm/edit-8.safetensors"
  }
}
```

公网 API 命令示例：

```bash
curl -fsS "https://images.example.com/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"t2i-v1","prompt":"A blue ceramic cup","size":"704x512","steps":4}'

curl -fsS "https://images.example.com/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F 'model=i2i-edit-v1' -F 'prompt=Change the ceramic cup to green' \
  -F 'steps=8' -F 'image=@reference.png'
```
