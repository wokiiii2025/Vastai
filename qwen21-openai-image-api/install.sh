#!/usr/bin/env bash
# Vast.ai Qwen-Image 2.1 deployment. Run with --help or --plan for details.
# Set QWEN_API_REPO to your Qwen-Image-Api Git URL or local clone.
# Requires Linux, Python 3.12 with venv, Git, an NVIDIA GPU, and Supervisor.
# Example (from a local clone of Vastai):
#   QWEN_API_REPO=/path/to/Qwen-Image-Api bash qwen21-openai-image-api/install.sh --prewarm
set -euo pipefail

# Keep the installation inputs together while the installer and verifier run.
workdir="$(mktemp -d)"
trap 'rm -rf -- "$workdir"' EXIT

cat > "$workdir/install.py" <<'QWEN_DEPLOY_FILE_1'
"""Install the pinned Qwen stack in a dedicated directory on a Vast.ai instance."""

import argparse
import hashlib
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


PACKAGE = Path(__file__).resolve().parent
API_REV = "e04eb1d1228146cceceaf3de4717fc5f20f91d56"
API_FILES = ("openai_images_api.py", "comfy_backend.py", "requirements.txt")
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_REV = "157db2ea2f835917c16191147c11e5896eca01a9"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
SERVICES = ("qwen21-comfyui", "qwen21-images-api")
PRESETS = ("t2i_4", "t2i_8", "edit_4", "edit_8")
BASE_WEIGHTS = "diffusion_models/NoctQ_V4_int8_convrot.safetensors"
MODELS = json.loads((PACKAGE / "models.lock.json").read_text(encoding="utf-8"))


def run(*cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def output(*cmd):
    return run(*cmd, capture_output=True, text=True).stdout.strip()


def write_file(path, data, mode=0o644):
    if path.exists() and path.read_bytes() == data and path.stat().st_mode & 0o777 == mode:
        return False
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(staging, mode)
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)
    return True


def checkout(comfy):
    if not comfy.exists():
        comfy.mkdir()
        run("git", "-C", comfy, "init", "--quiet")
        run("git", "-C", comfy, "remote", "add", "origin", COMFY_REPO)
    if output("git", "-C", comfy, "remote", "get-url", "origin") != COMFY_REPO:
        raise RuntimeError(f"Unexpected ComfyUI Git origin in {comfy}")
    head = subprocess.run(
        ["git", "-C", comfy, "rev-parse", "--verify", "HEAD"], capture_output=True, text=True,
    )
    if head.returncode:
        run("git", "-C", comfy, "fetch", "--depth=1", "origin", COMFY_REV)
        run("git", "-C", comfy, "checkout", "--detach", COMFY_REV)
    elif head.stdout.strip() != COMFY_REV:
        raise RuntimeError(f"{comfy} is at {head.stdout.strip()}; expected {COMFY_REV}")
    if subprocess.run(["git", "-C", comfy, "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError(f"Tracked ComfyUI files have local changes: {comfy}")


def fetch_api(repo):
    if Path(repo).is_dir():
        repo = str(Path(repo).resolve())
    source = PACKAGE / "api-source"
    run("git", "init", "--quiet", source)
    run("git", "-C", source, "fetch", "--quiet", "--depth=1", "--no-tags", "--", repo, API_REV)
    api = PACKAGE / "api"
    api.mkdir()
    for filename in API_FILES:
        blob = run("git", "-C", source, "show", f"{API_REV}:{filename}", capture_output=True)
        (api / filename).write_bytes(blob.stdout)


def prepare_python(root, comfy):
    python = root / ".venv/bin/python"
    if not python.is_file():
        run(sys.executable, "-m", "venv", root / ".venv")
    pip = (python, "-m", "pip", "install", "--disable-pip-version-check")
    run(*pip, "--index-url", TORCH_INDEX, "torch==2.7.1+cu126",
        "torchvision==0.22.1+cu126", "torchaudio==2.7.1+cu126")
    run(*pip, "-c", PACKAGE / "constraints-comfy.txt", "-r", comfy / "requirements.txt",
        "-r", PACKAGE / "api/requirements.txt", "huggingface-hub==2.4.0")
    run(python, "-m", "pip", "check")
    device = output(python, "-c", "import json, torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                    "torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize(); "
                    "gpu = torch.cuda.get_device_properties(0); "
                    "print(json.dumps({'name': gpu.name, 'bytes': gpu.total_memory, 'cuda': torch.version.cuda}))")
    stats = json.loads(device.splitlines()[-1])
    print(f"{stats['name']} / CUDA {stats['cuda']} / {stats['bytes'] / 1024**3:.1f} GiB", flush=True)
    return python, stats["bytes"]


def matches(path, model):
    if not path.is_file() or path.stat().st_size != model["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == model["sha256"]


def install_models(root, comfy, offline):
    models = comfy / "models"
    missing_bytes = 0
    for model in MODELS:
        target = models / model["target"]
        if target.exists() or target.is_symlink():
            if not matches(target, model):
                raise RuntimeError(f"Existing model failed SHA-256 verification: {target}")
            print(f"已校验 {model['target']}", flush=True)
        else:
            missing_bytes += model["bytes"]
    if shutil.disk_usage(root).free < missing_bytes:
        raise RuntimeError(f"Not enough free disk for {missing_bytes / 1024**3:.1f} GiB of models")
    for model in MODELS:
        target = models / model["target"]
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if offline:
            source = offline / model["target"]
        else:
            stage = root / ".downloads" / model["sha256"][:12]
            stage.mkdir(parents=True, exist_ok=True)
            source = stage / model["filename"]
            if not matches(source, model):
                run(root / ".venv/bin/hf", "download", model["repo"], model["filename"],
                    "--revision", model["revision"], "--local-dir", stage)
        if not matches(source, model):
            raise RuntimeError(f"Downloaded model failed SHA-256 verification: {model['target']}")
        if offline:
            fd, name = tempfile.mkstemp(dir=target.parent, prefix=".model-")
            staging = Path(name)
            try:
                with os.fdopen(fd, "wb") as f:
                    with source.open("rb") as contents:
                        shutil.copyfileobj(contents, f)
                if not matches(staging, model):
                    raise RuntimeError(f"Copied model failed SHA-256 verification: {model['target']}")
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
        else:
            source.replace(target)
        print(f"已安装 {model['target']}", flush=True)


def create_model_aliases(comfy, enabled):
    if not enabled:
        return {}
    models = comfy / "models"
    base = (models / BASE_WEIGHTS).resolve(strict=True)
    folder = models / "diffusion_models/qwen-warm"
    folder.mkdir(exist_ok=True)
    overrides = {}
    for preset in PRESETS:
        name = preset.replace("_", "-") + ".safetensors"
        link = folder / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(base)
        if not link.is_symlink() or link.resolve(strict=True) != base:
            raise RuntimeError(f"Model alias already points elsewhere: {link}")
        overrides[preset] = f"qwen-warm/{name}"
    return overrides


def wait_json(url, verify, timeout=120):
    direct_http = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with direct_http.open(url, timeout=5) as result:
                data = json.load(result)
            if verify(data):
                return data
        except (URLError, OSError, ValueError) as err:
            last_error = err
        time.sleep(1)
    raise RuntimeError(f"Service did not become ready at {url}: {last_error}")


def read_runtime(root):
    secret = root / "service/.runtime.json"
    return json.loads(secret.read_text(encoding="utf-8")) if secret.is_file() else {}


def prepare_runtime(root, args, overrides):
    runtime = root / "service"
    runtime.mkdir(mode=0o700, exist_ok=True)
    secret = runtime / ".runtime.json"
    old_settings = read_runtime(root)
    settings = {
        "api_key": old_settings.get("api_key") or os.getenv("QWEN_API_KEY") or secrets.token_urlsafe(36),
        "api_host": args.api_host,
        "api_port": args.api_port,
        "comfy_port": args.comfy_port,
        "public_base_url": args.public_base_url,
        "model_overrides": overrides,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", old_settings.get("cuda_visible_devices")),
    }
    api = runtime / "api"
    api.mkdir(exist_ok=True)
    for name in ("openai_images_api.py", "comfy_backend.py"):
        write_file(api / name, (PACKAGE / "api" / name).read_bytes())
    write_file(runtime / "run_service.py", (PACKAGE / "run_service.py").read_bytes())
    write_file(secret, (json.dumps(settings, indent=2) + "\n").encode(), 0o600)
    return settings


def supervisor_config(name, python, root, mode, settings):
    command = shlex.join([str(python), str(root / "service/run_service.py"), mode]).replace("%", "%%")
    logs = str(root / f"logs/{name}.log").replace("%", "%%")
    keys = ("comfy_port", "cuda_visible_devices") if mode == "comfy" else (
        "comfy_port", "api_host", "api_port", "public_base_url", "model_overrides",
    )
    digest = hashlib.sha256(json.dumps({key: settings[key] for key in keys}, sort_keys=True).encode())
    digest.update((root / "service/run_service.py").read_bytes())
    if mode == "api":
        for filename in ("openai_images_api.py", "comfy_backend.py"):
            digest.update((root / "service/api" / filename).read_bytes())
    return (
        f"# Managed by Vastai Qwen 2.1: {root}\n"
        f"[program:{name}]\ncommand={command}\nautostart=true\nautorestart=unexpected\n"
        f"environment=QWEN_SERVICE_REVISION=\"{digest.hexdigest()[:16]}\"\n"
        "startsecs=3\nstopasgroup=true\nkillasgroup=true\nstopwaitsecs=60\n"
        f"stdout_logfile={logs}\nstdout_logfile_maxbytes=10MB\nstdout_logfile_backups=2\n"
        "redirect_stderr=true\n"
    )


def service_state(name):
    result = subprocess.run(["supervisorctl", "status", name], capture_output=True, text=True)
    return next(iter(result.stdout.split()[1:2]), "")


def assert_port_free(bind, port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((bind, port))
        except OSError as err:
            raise RuntimeError(f"Port {bind}:{port} is already in use") from err


def check_service_slots(root, args):
    configs = [args.supervisor_dir / f"{name}.conf" for name in SERVICES]
    owner = f"# Managed by Vastai Qwen 2.1: {root}\n"
    old = read_runtime(root)
    for path in configs:
        if path.is_symlink() or (path.exists() and not path.read_text().startswith(owner)):
            raise RuntimeError(f"Refusing to replace a different service: {path}")
    ports = (("127.0.0.1", "comfy_port"), (args.api_host, "api_port"))
    for path, name, (bind, port_key) in zip(configs, SERVICES, ports):
        port = getattr(args, port_key)
        owns_port = path.is_file() and old.get(port_key) == port and service_state(name) == "RUNNING"
        if not owns_port:
            assert_port_free(bind, port)
    return configs


def start_services(root, python, args, settings, configs):
    (root / "logs").mkdir(exist_ok=True)
    for path, name, mode in zip(configs, SERVICES, ("comfy", "api")):
        write_file(path, supervisor_config(name, python, root, mode, settings).encode())
    run("supervisorctl", "reread")
    run("supervisorctl", "update", *SERVICES)
    for name in SERVICES:
        if service_state(name) in ("STOPPED", "EXITED"):
            run("supervisorctl", "start", name)
    comfy = f"http://127.0.0.1:{args.comfy_port}"
    api = f"http://127.0.0.1:{args.api_port}"
    wait_json(f"{comfy}/system_stats", lambda data: "system" in data)
    wait_json(f"{api}/health", lambda data: data.get("backends") == {"healthy": 1, "total": 1})
    run(python, PACKAGE / "verify.py", "--root", root, *(["--prewarm"] if args.prewarm else []))
    print(f"ComfyUI: {comfy}\nOpenAI Images API: {api}/v1\nAPI Key 文件: {root / 'service/.runtime.json'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Set QWEN_API_REPO to the Qwen-Image-Api Git URL or local clone. "
               "Git must already have access if the source repository is private.",
    )
    parser.add_argument("--root", type=Path, default=Path("/workspace/qwen21-image"))
    parser.add_argument("--model-dir", type=Path, help="Use an offline directory mirroring ComfyUI/models")
    parser.add_argument("--comfy-port", type=int, default=18188)
    parser.add_argument("--api-port", type=int, default=18080)
    parser.add_argument("--api-host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--public-base-url", default="")
    parser.add_argument("--supervisor-dir", type=Path, default=Path("/etc/supervisor/conf.d"))
    parser.add_argument("--warm-models", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--prepare-only", action="store_true", help="Install files without registering services")
    parser.add_argument("--prewarm", action="store_true", help="Run four real GPU requests after installation")
    parser.add_argument("--plan", action="store_true", help="Print the pinned install plan without making changes")
    args = parser.parse_args()
    if not all(1 <= p <= 65535 for p in (args.api_port, args.comfy_port)) or args.api_port == args.comfy_port:
        parser.error("Use two distinct TCP ports between 1 and 65535")
    if args.prewarm and args.prepare_only:
        parser.error("--prewarm needs running services")
    if args.plan:
        print(json.dumps({
            "root": str(args.root), "comfy_commit": COMFY_REV, "api_commit": API_REV,
            "torch": "2.7.1+cu126",
            "model_bytes": sum(m["bytes"] for m in MODELS), "warm_models": args.warm_models,
            "comfy_port": args.comfy_port, "api_port": args.api_port, "models": MODELS,
        }, ensure_ascii=False, indent=2))
        return
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        parser.error("Installation requires Linux and Python 3.12")
    if not args.prepare_only:
        if not args.supervisor_dir.is_dir() or not shutil.which("supervisorctl"):
            parser.error("Supervisor is required to start services; use --prepare-only to install files")
    for program in ("git", "nvidia-smi"):
        if not shutil.which(program):
            parser.error(f"Required executable missing: {program}")
    repo = os.getenv("QWEN_API_REPO", "").strip()
    if not repo:
        parser.error("Set QWEN_API_REPO to the Qwen-Image-Api Git URL or local clone")
    fetch_api(repo)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (root / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configs = None if args.prepare_only else check_service_slots(root, args)
        comfy = root / "ComfyUI"
        if shutil.disk_usage(root).free < (8 * 1024**3 if not (root / ".venv/bin/python").is_file() else 0):
            raise RuntimeError("Keep at least 8 GiB of free space for the Python environment")
        checkout(comfy)
        python, gpu_memory = prepare_python(root, comfy)
        warm = args.warm_models == "on" or (args.warm_models == "auto" and gpu_memory >= 72 * 1024**3)
        install_models(root, comfy, args.model_dir)
        overrides = create_model_aliases(comfy, warm)
        runtime = prepare_runtime(root, args, overrides)
        print("模型热载别名：", ", ".join(overrides) if overrides else "共用默认底模", flush=True)
        if configs:
            start_services(root, python, args, runtime, configs)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"安装失败：{error}")
QWEN_DEPLOY_FILE_1

cat > "$workdir/models.lock.json" <<'QWEN_DEPLOY_FILE_2'
[
  {
    "target": "diffusion_models/NoctQ_V4_int8_convrot.safetensors",
    "repo": "Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1",
    "revision": "a81b9af51120a78e285e57906f2250a2a02080e9",
    "filename": "NoctQ_V4_int8_convrot.safetensors",
    "bytes": 7256784368,
    "sha256": "4d92d5538253ab36e6a73b056cce5f697950f4303cb9e6686a3198d0ea13f9e8"
  },
  {
    "target": "text_encoders/qwen3vl_8b_int8_convrot_heretic.safetensors",
    "repo": "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-int8-convrot",
    "revision": "033b4bf23f8816b22c065309bc9b349953193978",
    "filename": "qwen3vl_8b_int8_convrot_heretic.safetensors",
    "bytes": 9350828392,
    "sha256": "f15ce4275428e04f42cdb59e3a253cb290cae5de99259527f01d3d2e51153653"
  },
  {
    "target": "vae/qwen_image_2.1_vae_bf16.safetensors",
    "repo": "Comfy-Org/Qwen-Image-2.1",
    "revision": "aacf31a6d48e707f352ecba714cee0680da3479b",
    "filename": "vae/qwen_image_2.1_vae_bf16.safetensors",
    "bytes": 675509688,
    "sha256": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"
  },
  {
    "target": "loras/qwen_image_2.1_lightning_4step.safetensors",
    "repo": "lightx2v/Qwen-Image-2.1-Lightning",
    "revision": "6b0a1176327c01522a44276c5da67331a025b6fc",
    "filename": "pytorch_lora_weights.safetensors",
    "bytes": 67144280,
    "sha256": "53c69320d4163da51032f90275074028cb098556039c72b45f8dce9f0f9e34a3"
  },
  {
    "target": "loras/turbo8_lora_step2500.safetensors",
    "repo": "chriswritescode/Turbo8-LoRA-Qwen-Image-2.1",
    "revision": "611a3f86d0cc53d21a1a09d07ea7dc82df3d68a1",
    "filename": "turbo8_lora_step2500.safetensors",
    "bytes": 1359148744,
    "sha256": "a0b7a7634ff2f82ccbe209e16b0ba19b25e2d9e18624d89edc490503b6a75d39"
  },
  {
    "target": "loras/Qwen-Image-2.1 NSFW Image Edit.safetensors",
    "repo": "mpasila/Qwen-Image-2.1-LoRAs",
    "revision": "401c3423daed3d678eb7bb6477eb695ba0a01f4d",
    "filename": "Qwen-Image-2.1 NSFW Image Edit.safetensors",
    "bytes": 83943552,
    "sha256": "9c143cbfe7b016efdce7499506c02d0bfa1842ae6524bb64e05888f2e325e67a"
  }
]
QWEN_DEPLOY_FILE_2

cat > "$workdir/constraints-comfy.txt" <<'QWEN_DEPLOY_FILE_3'
torch==2.7.1+cu126
torchvision==0.22.1+cu126
torchaudio==2.7.1+cu126
transformers==5.23.0
safetensors==0.8.0
numpy==2.5.4
huggingface-hub==2.4.0
comfy-aimdo==0.6.0
comfy-kitchen==0.2.41
comfy-angle==0.1.1
pydantic==2.14.3
QWEN_DEPLOY_FILE_3

cat > "$workdir/run_service.py" <<'QWEN_DEPLOY_FILE_4'
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
QWEN_DEPLOY_FILE_4

cat > "$workdir/verify.py" <<'QWEN_DEPLOY_FILE_5'
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
QWEN_DEPLOY_FILE_5

"${PYTHON:-python3}" "$workdir/install.py" "$@"
