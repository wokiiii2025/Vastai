"""Check model and node availability, with optional real GPU prewarming."""

import argparse
import base64
import io
import json
import secrets
import time
from pathlib import Path

import httpx
from PIL import Image


PACKAGE = Path(__file__).resolve().parent
MODELS = json.loads((PACKAGE / "models.lock.json").read_text(encoding="utf-8"))
LOADERS = {
    "diffusion_models": ("UNETLoader", "unet_name"),
    "text_encoders": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
    "loras": ("LoraLoaderModelOnly", "lora_name"),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_json(response):
    response.raise_for_status()
    return response.json()


def verify_models(comfy, overrides):
    needed = {}
    for model in MODELS:
        folder, filename = model["target"].split("/", 1)
        needed.setdefault(LOADERS[folder], set()).add(filename)
    needed["UNETLoader", "unet_name"].update(overrides.values())
    for (node, field), names in needed.items():
        available = checked_json(comfy.get(f"object_info/{node}"))[node]["input"]["required"][field][0]
        require(names.issubset(available), f"{node} missing: {', '.join(sorted(names - set(available)))}")
    for node in ("TextEncodeQwenImage21", "ModelSamplingFlux", "EmptyLatentImage",
                 "LoadImage", "ImageScale", "KSampler", "VAEDecode", "SaveImage"):
        require(node in checked_json(comfy.get(f"object_info/{node}")), f"Missing native node {node}")


def recent_workflow(comfy, seed, kind, steps):
    for job in reversed(list(checked_json(comfy.get("history", params={"max_items": 40})).values())):
        graph = job["prompt"][2]
        sampler = graph.get("6", {}).get("inputs", {})
        job_kind = "edit" if sampler.get("latent_image") == ["4", 2] else "t2i"
        if (sampler.get("seed"), job_kind, sampler.get("steps")) == (seed, kind, steps):
            require(job["status"]["status_str"] == "success", f"ComfyUI failed {kind}/{steps}")
            cached = dict(job["status"]["messages"]).get("execution_cached", {}).get("nodes", [])
            require("6" not in cached, f"KSampler was cached for {kind}/{steps}")
            return graph
    raise RuntimeError(f"No ComfyUI history for {kind}/{steps}")


def prewarm(comfy, api, headers, overrides):
    reference = None
    for kind, steps in (("t2i", 4), ("t2i", 8), ("edit", 4), ("edit", 8)):
        seed = secrets.randbelow(2**63)
        request = {"size": "512x512", "steps": steps, "seed": seed, "response_format": "b64_json"}
        started = time.monotonic()
        if kind == "t2i":
            response = api.post("v1/images/generations", headers=headers, json={
                **request, "model": "t2i-v1", "prompt": "A simple studio product photograph of a blue ceramic cup",
            })
        else:
            response = api.post("v1/images/edits", headers=headers, data={
                **request, "model": "i2i-edit-v1", "prompt": "Change only the ceramic cup to green",
            }, files={"image": ("reference.png", reference, "image/png")})
        binary = base64.b64decode(checked_json(response)["data"][0]["b64_json"], validate=True)
        with Image.open(io.BytesIO(binary)) as image:
            require(image.size == (512, 512) and image.format == "PNG", f"Unexpected output from {kind}/{steps}")
            image.verify()
        graph = recent_workflow(comfy, seed, kind, steps)
        sampler = graph["6"]["inputs"]
        require(sampler["cfg"] == 2.0, f"Wrong CFG in {kind}/{steps}")
        require(sampler["sampler_name"] == "euler" and sampler["scheduler"] == "simple",
                f"Unexpected sampler in {kind}/{steps}")
        preset = f"{kind}_{steps}"
        if preset in overrides:
            require(graph["1"]["inputs"]["unet_name"] == overrides[preset], f"Wrong model alias for {preset}")
        if reference is None:
            reference = binary
        print(f"{preset} PASS | 512x512 PNG | CFG 2.0 | {time.monotonic() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/qwen21-image"))
    parser.add_argument("--prewarm", action="store_true")
    args = parser.parse_args()
    settings = json.loads((args.root / "service/.runtime.json").read_text(encoding="utf-8"))
    comfy_url = f"http://127.0.0.1:{settings['comfy_port']}"
    api_url = f"http://127.0.0.1:{settings['api_port']}"
    with (
        httpx.Client(base_url=comfy_url, timeout=15, trust_env=False) as comfy,
        httpx.Client(base_url=api_url, timeout=180, trust_env=False) as api,
    ):
        verify_models(comfy, settings["model_overrides"])
        require(api.get("v1/models").status_code == 401, "Images API must require its Bearer key")
        headers = {"Authorization": "Bearer " + settings["api_key"]}
        models = checked_json(api.get("v1/models", headers=headers))["data"]
        require({m["id"] for m in models} == {"t2i-v1", "i2i-edit-v1"}, "Wrong API models")
        print("PASS: ComfyUI native nodes, model names and API authentication", flush=True)
        if args.prewarm:
            prewarm(comfy, api, headers, settings["model_overrides"])


if __name__ == "__main__":
    main()
