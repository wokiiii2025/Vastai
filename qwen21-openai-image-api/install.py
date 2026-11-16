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
    parser = argparse.ArgumentParser(description=__doc__)
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
            "root": str(args.root), "comfy_commit": COMFY_REV, "torch": "2.7.1+cu126",
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
