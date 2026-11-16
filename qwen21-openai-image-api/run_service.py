"""Launch one service from the installed stack."""

import json
import os
import sys
from pathlib import Path


def main():
    service = Path(__file__).resolve().parent
    root = service.parent
    settings = json.loads((service / ".runtime.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if sys.argv[1:] == ["comfy"]:
        if settings["cuda_visible_devices"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = settings["cuda_visible_devices"]
        os.chdir(root / "ComfyUI")
        cmd = [
            "main.py", "--listen", "127.0.0.1", "--port", str(settings["comfy_port"]),
            "--use-pytorch-cross-attention", "--preview-method", "none",
            "--disable-all-custom-nodes", "--offline",
        ]
    elif sys.argv[1:] == ["api"]:
        os.chdir(service / "api")
        for name in ("QWEN_MODEL", "QWEN_TEXT_ENCODER", "QWEN_VAE", "QWEN_TURBO8_LORA",
                     "QWEN_LIGHTNING4_LORA", "QWEN_NSFW_EDIT_LORA"):
            env.pop(name, None)
        no_proxy = env.get("no_proxy") or env.get("NO_PROXY") or ""
        env["no_proxy"] = env["NO_PROXY"] = ",".join(filter(None, [no_proxy, "127.0.0.1", "localhost"]))
        env.update(
            OPENAI_API_KEY=settings["api_key"],
            COMFY_BACKENDS=json.dumps([{
                "name": "qwen-local", "url": f"http://127.0.0.1:{settings['comfy_port']}",
                "model_overrides": settings["model_overrides"],
            }]),
            COMFY_INPUT_DIR=str(root / "data/inputs"),
            COMFY_OUTPUT_DIR=str(root / "data/output"),
            IMAGE_TASK_DB=str(root / "data/tasks.sqlite3"),
            PUBLIC_BASE_URL=settings["public_base_url"],
        )
        cmd = [
            "-m", "uvicorn", "openai_images_api:app", "--workers", "1",
            "--host", settings["api_host"], "--port", str(settings["api_port"]),
        ]
    else:
        raise SystemExit("Usage: run_service.py [comfy|api]")
    os.execve(sys.executable, [sys.executable, *cmd], env)


if __name__ == "__main__":
    main()
