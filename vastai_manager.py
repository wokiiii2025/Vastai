#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import argparse
import shutil
import urllib.request
import urllib.parse
import json
import re
from pathlib import Path

# ==========================================
# 默认配置
# ==========================================
WORK_DIR = "/workspace/ollama"
LOG_DIR = os.path.join(WORK_DIR, "logs")
MODELS_DIR = os.path.join(WORK_DIR, "models")
GGUF_DIR = os.path.join(MODELS_DIR, "gguf")
CONFIG_FILE = os.path.join(WORK_DIR, "manager_config.json")
OLLAMA_SUGGESTED_PORT = "40056"

# 模型配置
MODEL_URL = "https://huggingface.co/mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF/resolve/main/Huihui-Qwen3-4B-Instruct-2507-abliterated.Q8_0.gguf"
MODEL_NAME = "huihui-qwen3-4b:latest"
GGUF_FILENAME = "Huihui-Qwen3-4B-Instruct-2507-abliterated.Q8_0.gguf"

# ComfyUI Z-Image 模型下载配置
COMFYUI_MODELS_ROOT = "/workspace/ComfyUI/models"
ZIMAGE_MODELS = [
    {
        "subdir": "text_encoders",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    },
    {
        "subdir": "diffusion_models",
        "url": "https://civitai.red/api/download/models/2543657?fileId=2432032",
        "token_env": "CIVITAI_API_TOKEN",
        "token_query_param": "token",
        "filename_from_response": True,
    },
    {
        "subdir": "diffusion_models",
        "url": "https://civitai.red/api/download/models/2903129?fileId=2781142",
        "token_env": "CIVITAI_API_TOKEN",
        "token_query_param": "token",
        "filename_from_response": True,
    },
    {
        "subdir": "vae",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    },
]
ZIMAGE_OBSOLETE_MODELS = [
    os.path.join(COMFYUI_MODELS_ROOT, "model_patches", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
]

# Z-Image-Turbo 模型下载配置（仅 HuggingFace，不涉及 Civitai token）
ZIMAGE_TURBO_MODELS = [
    {
        "subdir": "text_encoders",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    },
    {
        "subdir": "diffusion_models",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    },
    {
        "subdir": "diffusion_models",
        "url": "https://huggingface.co/SeeSee21/Z-Anime/resolve/main/diffusion_models/z-anime-distill-8step-bf16.safetensors",
    },
    {
        "subdir": "vae",
        "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    },
]

# Qwen-Rapid-AIO 模型下载配置
QWEN_RAPID_AIO_MODEL = {
    "subdir": "checkpoints",
    "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v17/Qwen-Rapid-AIO-NSFW-v17.safetensors",
}

# ==========================================
# 视觉美化工具
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    NC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'

def print_banner():
    if sys.stdout.isatty() and os.environ.get("TERM"):
        os.system('clear')
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print(r""" __     __        _      _    _
 \ \   / /_ _ ___| |_   / \  (_)
  \ \ / / _` / __| __| / _ \ | |
   \ V / (_| \__ \ |_ / ___ \| |
    \_/ \__,_|___/\__/_/   \_\_|""")
    print(f"{Colors.NC}")

def print_info(msg): print(f"{Colors.BLUE}[信息]{Colors.NC} {msg}")
def print_success(msg): print(f"{Colors.GREEN}[成功]{Colors.NC} {msg}")
def print_warning(msg): print(f"{Colors.YELLOW}[警告]{Colors.NC} {msg}")
def print_error(msg): print(f"{Colors.RED}[错误]{Colors.NC} {msg}")

# ==========================================
# Vast.ai ComfyUI 本机/可选 SSH 升级工具
# ==========================================
VAST_DEFAULT_USER = "root"
VAST_DEFAULT_KEY = os.path.expanduser("~/.ssh/id_rsa")
VAST_COMFY_DIR = "/workspace/ComfyUI"
VAST_VENV_ACTIVATE = "/venv/main/bin/activate"
VAST_STARTUP_FILES = [
    "/opt/supervisor-scripts/comfyui.sh",
    "/workspace/comfyui_gpus_start.sh",
]

def ensure_private_key_permissions(key_path):
    path = Path(key_path).expanduser()
    if not path.exists():
        print_error(f"SSH 私钥不存在: {path}")
        return False

    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        print_warning(f"SSH 私钥权限过宽 ({oct(mode)})，自动调整为 600。")
        path.chmod(0o600)
    return True

def ssh_base_cmd(host, port, key_path, user=VAST_DEFAULT_USER):
    return [
        "ssh",
        "-i", str(Path(key_path).expanduser()),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        f"{user}@{host}",
    ]

def ssh_run(host, port, key_path, remote_cmd, user=VAST_DEFAULT_USER, stream=True, check=True):
    cmd = ssh_base_cmd(host, port, key_path, user) + [remote_cmd]
    if stream:
        proc = subprocess.run(cmd, text=True)
    else:
        proc = subprocess.run(cmd, text=True, capture_output=True)

    if check and proc.returncode != 0:
        if not stream and proc.stderr:
            print_error(proc.stderr.strip())
        raise RuntimeError(f"SSH 命令执行失败，退出码: {proc.returncode}")
    return proc.stdout.strip() if not stream and proc.stdout else ""

def local_run(script, check=True):
    proc = subprocess.run(["bash", "-lc", script], text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"本机命令执行失败，退出码: {proc.returncode}")
    return proc.returncode == 0

def run_vast_script(script, host=None, port=None, key_path=None, user=VAST_DEFAULT_USER):
    if host and port:
        return ssh_run(host, port, key_path or VAST_DEFAULT_KEY, script, user=user, stream=True)
    return local_run(script)

def print_vast_target(host=None, port=None, key_path=None):
    if host and port:
        print_info(f"执行模式: SSH 远程 root@{host}:{port}")
        print_info(f"SSH 私钥: {Path(key_path or VAST_DEFAULT_KEY).expanduser()}")
    else:
        print_info("执行模式: 本机 Vast.ai 服务器")
        print_info("不会创建 SSH 连接；请在目标服务器上直接运行本脚本。")

def vast_preflight_script():
    return r'''
set -e
get_supervisor_port() {
  grep -E 'COMFYUI_ARGS=.*--port[ =]?[0-9]+' /opt/supervisor-scripts/comfyui.sh 2>/dev/null \
    | sed -E 's/.*--port[ =]?([0-9]+).*/\1/' \
    | tail -1
}
get_gpu_base_port() {
  grep -E '^BASE_PORT=' /workspace/comfyui_gpus_start.sh 2>/dev/null \
    | sed -E 's/.*BASE_PORT=([0-9]+).*/\1/' \
    | tail -1
}
get_gpu_count() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
}
print_expected_ports() {
  supervisor_port=$(get_supervisor_port)
  base_port=$(get_gpu_base_port)
  gpu_count=$(get_gpu_count)
  printf "supervisor_port=%s\n" "${supervisor_port:-unknown}"
  printf "gpu_base_port=%s\n" "${base_port:-unknown}"
  printf "gpu_count=%s\n" "${gpu_count:-0}"
  if [ -n "${base_port:-}" ] && [ "${gpu_count:-0}" -gt 0 ]; then
    printf "gpu_ports="
    i=0
    while [ "$i" -lt "$gpu_count" ]; do
      printf "%s " "$((base_port + i))"
      i=$((i + 1))
    done
    printf "\n"
  fi
}
list_expected_ports() {
  supervisor_port=$(get_supervisor_port)
  base_port=$(get_gpu_base_port)
  gpu_count=$(get_gpu_count)
  [ -n "${supervisor_port:-}" ] && echo "$supervisor_port"
  if [ -n "${base_port:-}" ] && [ "${gpu_count:-0}" -gt 0 ]; then
    i=0
    while [ "$i" -lt "$gpu_count" ]; do
      echo "$((base_port + i))"
      i=$((i + 1))
    done
  fi
}
kill_expected_gpu_listeners() {
  base_port=$(get_gpu_base_port)
  gpu_count=$(get_gpu_count)
  if [ -z "${base_port:-}" ] || [ "${gpu_count:-0}" -le 0 ]; then
    echo "No GPU ports derived from startup script/GPU count."
    return 0
  fi
  i=0
  while [ "$i" -lt "$gpu_count" ]; do
    port=$((base_port + i))
    pid=$(ss -tlnp | grep ":${port} " | sed -E "s/.*pid=([0-9]+).*/\1/" | head -1 || true)
    if [ -n "${pid:-}" ]; then
      echo "port $port pid $pid"
      kill "$pid" || true
    else
      echo "port $port no listener"
    fi
    i=$((i + 1))
  done
}
print_expected_port_listeners() {
  for port in $(list_expected_ports); do
    ss -tlnp | grep ":${port} " || true
  done
}
missing=0
require_path() {
  if [ ! -e "$1" ]; then
    echo "MISSING path: $1"
    missing=1
  else
    echo "OK path: $1"
  fi
}
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MISSING command: $1"
    missing=1
  else
    echo "OK command: $1 -> $(command -v "$1")"
  fi
}

printf "\n== Required paths ==\n"
require_path /workspace
require_path /workspace/ComfyUI
require_path /workspace/ComfyUI/.git
require_path /workspace/ComfyUI/main.py
require_path /workspace/ComfyUI/requirements.txt
require_path /venv/main/bin/activate
require_path /venv/main/bin/python
require_path /opt/supervisor-scripts/comfyui.sh
require_path /workspace/comfyui_gpus_start.sh

printf "\n== Required commands ==\n"
require_cmd git
require_cmd python3
require_cmd curl
require_cmd ss
require_cmd supervisorctl
require_cmd uv

if [ "$missing" -ne 0 ]; then
  echo "Preflight failed: required Vast.ai/ComfyUI paths or commands are missing."
  exit 20
fi

printf "\n== Startup file hashes (must remain unchanged) ==\n"
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh

printf "\n== Ports derived from existing startup files ==\n"
print_expected_ports
'''

def vast_comfyui_check(host=None, port=None, key_path=None, user=VAST_DEFAULT_USER):
    print_banner()
    print_info("Vast.ai ComfyUI 检查")
    print_vast_target(host, port, key_path)

    if host and port and not ensure_private_key_permissions(key_path or VAST_DEFAULT_KEY):
        return False

    check_script = vast_preflight_script() + r'''
set -e
printf "\n== Host ==\n"
hostname
whoami
pwd

printf "\n== Workspace ==\n"
ls -la /workspace 2>/dev/null || true

printf "\n== ComfyUI paths ==\n"
ls -ld /workspace/ComfyUI /opt/workspace-internal/ComfyUI 2>/dev/null || true
realpath /workspace/ComfyUI /opt/workspace-internal/ComfyUI 2>/dev/null || true

printf "\n== Git status ==\n"
cd /workspace/ComfyUI
git rev-parse --is-inside-work-tree
git remote -v
printf "current_head="
git rev-parse --short HEAD
printf "branch="
git branch --show-current || true
git status --short
git fetch --prune origin master
printf "origin_master="
git rev-parse --short origin/master
printf "ahead_behind="
git rev-list --left-right --count HEAD...origin/master

printf "\n== Python packages ==\n"
. /venv/main/bin/activate
python --version
uv --version 2>/dev/null || true
python -m pip show comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs torch 2>/dev/null | awk '/^(Name|Version):/ {print}'

printf "\n== Startup scripts (read-only) ==\n"
printf "注意：以下仅打印，不修改 Vast.ai 启动文件和启动参数。\n"
sed -n '1,220p' /opt/supervisor-scripts/comfyui.sh 2>/dev/null || true
printf "\n-- /workspace/comfyui_gpus_start.sh --\n"
sed -n '1,220p' /workspace/comfyui_gpus_start.sh 2>/dev/null || true

printf "\n== Processes and ports ==\n"
supervisorctl status comfyui 2>/dev/null || true
ps -eo pid,ppid,cmd | grep -E '[p]ython main.py|[b]ash -c .*main.py' | sed -n '1,120p' || true
print_expected_port_listeners
'''
    run_vast_script(check_script, host=host, port=port, key_path=key_path, user=user)
    return True

def vast_comfyui_upgrade(host=None, port=None, key_path=None, user=VAST_DEFAULT_USER):
    print_banner()
    print_info("Vast.ai ComfyUI 升级")
    print_vast_target(host, port, key_path)

    if host and port and not ensure_private_key_permissions(key_path or VAST_DEFAULT_KEY):
        return False

    print_info("执行升级前检查...")
    vast_comfyui_check(host, port, key_path, user=user)

    print_info("开始升级 ComfyUI；不会修改 /opt/supervisor-scripts 或 /workspace/comfyui_gpus_start.sh。")
    upgrade_script = vast_preflight_script() + r'''
set -euo pipefail
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh > /tmp/vast_comfyui_startup_hash_before.txt
cd /workspace/ComfyUI

git fetch --prune origin master
backup_branch="vastai-pre-upgrade-$(date +%Y%m%d-%H%M%S)"
old_head=$(git rev-parse --short HEAD)
new_head=$(git rev-parse --short origin/master)

printf "Creating backup branch %s at %s\n" "$backup_branch" "$old_head"
git branch "$backup_branch" HEAD

git diff --name-only --diff-filter=D > /tmp/comfyui_deleted_paths.txt
if [ -s /tmp/comfyui_deleted_paths.txt ]; then
  printf "Preserving deleted paths:\n"
  sed -n '1,120p' /tmp/comfyui_deleted_paths.txt
  xargs -a /tmp/comfyui_deleted_paths.txt git stash push -m "vastai-upgrade-preserve-deletions-$old_head" --
fi

printf "Stopping Vast supervisor comfyui...\n"
supervisorctl stop comfyui || true

printf "Killing GPU listeners by port...\n"
kill_expected_gpu_listeners
sleep 5

printf "Switching to master and fast-forwarding to origin/master %s...\n" "$new_head"
git switch master
git merge --ff-only origin/master

if [ -s /tmp/comfyui_deleted_paths.txt ]; then
  while IFS= read -r f; do
    [ -n "$f" ] && rm -f -- "$f"
  done < /tmp/comfyui_deleted_paths.txt
  git stash list | grep -q "vastai-upgrade-preserve-deletions-$old_head" && git stash drop stash@{0} || true
fi

printf "Installing requirements in /venv/main with uv...\n"
. /venv/main/bin/activate
uv pip --no-cache-dir install -r requirements.txt

printf "Restarting Vast supervisor comfyui...\n"
supervisorctl start comfyui || supervisorctl restart comfyui || true
sleep 10

printf "Verifying Vast.ai startup files were not modified before GPU start...\n"
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh > /tmp/vast_comfyui_startup_hash_after.txt
if ! cmp -s /tmp/vast_comfyui_startup_hash_before.txt /tmp/vast_comfyui_startup_hash_after.txt; then
  echo "ERROR: Vast.ai startup files changed. Aborting before GPU helper start."
  diff -u /tmp/vast_comfyui_startup_hash_before.txt /tmp/vast_comfyui_startup_hash_after.txt || true
  exit 30
fi

printf "Starting GPU helper script with original startup path...\n"
WORKSPACE=/workspace /workspace/comfyui_gpus_start.sh || true

printf "\nUpgrade summary:\n"
printf "old_head=%s\nnew_head=%s\nbackup_branch=%s\n" "$old_head" "$(git rev-parse --short HEAD)" "$backup_branch"
printf "frontend="; python -m pip show comfyui-frontend-package | awk '/^Version:/ {print $2}'
printf "templates="; python -m pip show comfyui-workflow-templates | awk '/^Version:/ {print $2}'
printf "embedded_docs="; python -m pip show comfyui-embedded-docs | awk '/^Version:/ {print $2}'
printf "status:\n"; git status --short
'''
    run_vast_script(upgrade_script, host=host, port=port, key_path=key_path, user=user)

    print_info("检查并修复已知 LTXVideo 自定义节点兼容问题...")
    patch_script = r'''
set -e
node_dir="/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo-Registry_Mattabyte"
file="$node_dir/tricks/modules/ltx_model.py"

if [ -f "$file" ]; then
  cd "$node_dir"
  git fetch --prune origin || true
  printf "LTXVideo node status before patch:\n"
  git status --short || true
  if grep -q "precompute_freqs_cis" "$file"; then
    cp "$file" /tmp/ltx_model.py.vastai-prepatch
    perl -0pi -e 's/,\n\s*precompute_freqs_cis//' "$file"
    perl -0pi -e 's/pe = precompute_freqs_cis\(/pe = self._precompute_freqs_cis(/' "$file"
    printf "Applied LTXVideo compatibility patch if needed.\n"
    git diff -- tricks/modules/ltx_model.py || true
  else
    printf "LTXVideo compatibility patch not needed.\n"
  fi
else
  printf "LTXVideo custom node not present, skipping patch.\n"
fi
'''
    run_vast_script(patch_script, host=host, port=port, key_path=key_path, user=user)

    print_info("重启 ComfyUI，让自定义节点补丁生效...")
    restart_script = r'''
set -e
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh > /tmp/vast_comfyui_startup_hash_before_restart.txt
supervisorctl restart comfyui || true
kill_expected_gpu_listeners
sleep 5
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh > /tmp/vast_comfyui_startup_hash_after_restart.txt
if ! cmp -s /tmp/vast_comfyui_startup_hash_before_restart.txt /tmp/vast_comfyui_startup_hash_after_restart.txt; then
  echo "ERROR: Vast.ai startup files changed during restart."
  diff -u /tmp/vast_comfyui_startup_hash_before_restart.txt /tmp/vast_comfyui_startup_hash_after_restart.txt || true
  exit 31
fi
WORKSPACE=/workspace /workspace/comfyui_gpus_start.sh || true
'''
    run_vast_script(restart_script, host=host, port=port, key_path=key_path, user=user)

    print_info("执行最终验证...")
    verify_script = r'''
set -e
expected_version="0.21.1"
cd /workspace/ComfyUI
printf "\n== Versions ==\n"
head_now=$(git rev-parse --short HEAD)
echo "$head_now"
git status --short
. /venv/main/bin/activate
python - <<'PY'
import comfyui_version
print("comfyui_version", getattr(comfyui_version, "__version__", "unknown"))
PY
python -m pip show comfyui-frontend-package comfyui-workflow-templates comfyui-embedded-docs | awk '/^(Name|Version):/ {print}'

printf "\n== Waiting for HTTP checks ==\n"
failed=0
ports=$(list_expected_ports | sort -n | uniq)
if [ -z "$ports" ]; then
  echo "ERROR: Could not derive any ComfyUI ports from existing startup files."
  exit 40
fi
for port in $ports; do
  ok=0
  for i in $(seq 1 24); do
    body=$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/system_stats" 2>/dev/null || true)
    version=$(printf "%s" "$body" | sed -n 's/.*"comfyui_version": "\([^"]*\)".*/\1/p')
    if [ -n "$version" ]; then
      echo "$port comfyui_version=$version"
      ok=1
      break
    fi
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    echo "$port FAILED"
    failed=1
  fi
done

printf "\n== Supervisor and ports ==\n"
supervisorctl status comfyui || true
print_expected_port_listeners

printf "\n== Recent log signals ==\n"
tail -n 280 /workspace/comfyui_gpu0.log | grep -Ei 'ComfyUI-LTXVideo-Registry|IMPORT FAILED|precompute_freqs_cis|traceback|cannot import|error|failed' || true
tail -n 380 /var/log/portal/comfyui.log | grep -Ei 'ComfyUI-LTXVideo-Registry|IMPORT FAILED|precompute_freqs_cis|traceback|cannot import|error|failed' || true

if tail -n 280 /workspace/comfyui_gpu0.log | grep -Eq 'IMPORT FAILED.*ComfyUI-LTXVideo-Registry|precompute_freqs_cis'; then
  echo "ERROR: LTXVideo compatibility failure is still present in fresh GPU log."
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo "ERROR: Final verification failed."
  exit 40
fi

printf "\n== Startup file hashes after completion ==\n"
sha256sum /opt/supervisor-scripts/comfyui.sh /workspace/comfyui_gpus_start.sh

printf "\n== Backup branches ==\n"
git branch --list 'vastai-pre-upgrade-*' -v || true

printf "\n== Custom node diff ==\n"
git -C custom_nodes/ComfyUI-LTXVideo-Registry_Mattabyte status --short 2>/dev/null || true
git -C custom_nodes/ComfyUI-LTXVideo-Registry_Mattabyte diff -- tricks/modules/ltx_model.py 2>/dev/null || true
'''
    run_vast_script(verify_script, host=host, port=port, key_path=key_path, user=user)
    print_success("Vast.ai ComfyUI 升级流程已完成。")
    return True

# ==========================================
# 核心逻辑
# ==========================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        return {"port": str(cfg.get("port", "") or ""), "host": cfg.get("host", "0.0.0.0") or "0.0.0.0"}
    return {"port": "", "host": "0.0.0.0"}

def save_config(port, host):
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f: json.dump({"port": str(port), "host": host}, f)

def prompt_for_ollama_port(config=None):
    config = config or load_config()
    current = config.get("port") or ""
    while True:
        if current:
            prompt = f"{Colors.BOLD}请输入 Ollama 服务端口 (当前: {current}，回车沿用): {Colors.NC}"
        else:
            prompt = f"{Colors.BOLD}请输入 Ollama 服务端口 (建议: {OLLAMA_SUGGESTED_PORT}，不可留空): {Colors.NC}"
        new_port = input(prompt).strip()
        if not new_port and current:
            new_port = current
        if not new_port:
            print_warning("首次配置必须输入端口；脚本不会擅自使用固定默认端口。")
            continue
        if not new_port.isdigit():
            print_warning("端口必须是数字。")
            continue
        if not (1 <= int(new_port) <= 65535):
            print_warning("端口范围必须是 1-65535。")
            continue
        return new_port

def require_ollama_config():
    config = load_config()
    if config.get("port"):
        return config
    print_warning("Ollama 服务端口尚未配置。")
    if not sys.stdin.isatty():
        raise RuntimeError("非交互模式下请先通过 --port 指定 Ollama 端口。")
    port = prompt_for_ollama_port(config)
    save_config(port, config["host"])
    print_success(f"Ollama 端口已配置为: {port}")
    return load_config()

def is_root():
    return os.getuid() == 0

def run_cmd(cmd, env=None, capture=False, check=True):
    try:
        e = os.environ.copy()
        if env: e.update(env)
        res = subprocess.run(cmd, shell=True, env=e, check=check, capture_output=capture, text=True)
        return res.stdout.strip() if capture else True
    except: return False

def format_bytes(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024

def get_zimage_model_specs():
    return ZIMAGE_MODELS

def resolve_model_target_path(model_spec):
    filename = model_spec.get("filename")
    if model_spec.get("filename_from_response") and not filename:
        return os.path.join(COMFYUI_MODELS_ROOT, model_spec["subdir"])
    if not filename:
        parsed = urllib.parse.urlparse(model_spec["url"])
        filename = urllib.parse.unquote(os.path.basename(parsed.path))
    if not filename:
        raise ValueError(f"无法从 URL 解析文件名: {model_spec['url']}")
    return os.path.join(COMFYUI_MODELS_ROOT, model_spec["subdir"], filename)

def build_model_download_url(model_spec):
    url = model_spec["url"]
    token_env = model_spec.get("token_env")
    token = os.environ.get(token_env) if token_env else None
    if token_env and not token:
        raise RuntimeError(
            f"下载 {url} 需要先设置环境变量 {token_env}。\n"
            f"示例: export {token_env}=\"your_token_here\""
        )
    token_query_param = model_spec.get("token_query_param")
    if token and token_query_param:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = [(k, v) for k, v in query if k != token_query_param]
        query.append((token_query_param, token))
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))
    return url

def get_response_filename(response):
    content_disposition = response.headers.get("Content-Disposition") or ""
    match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition)
    if match:
        return urllib.parse.unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return urllib.parse.unquote(match.group(1).strip())
    return ""

def cleanup_obsolete_zimage_models():
    removed = 0
    for path in ZIMAGE_OBSOLETE_MODELS:
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
                print_success(f"已删除多余旧模型: {path}")
            except Exception as e:
                print_error(f"删除多余旧模型失败: {path} - {e}")
        else:
            print_info(f"多余旧模型不存在，跳过: {path}")
    return removed

def download_file_with_progress(url, dest_path, timeout=60, filename_from_response=False):
    dest_is_dir = filename_from_response or os.path.isdir(dest_path)
    if not dest_is_dir and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print_success(f"已存在，跳过: {dest_path} ({format_bytes(os.path.getsize(dest_path))})")
        return "skipped"

    if dest_is_dir:
        os.makedirs(dest_path, exist_ok=True)
        download_dir = dest_path
    else:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        download_dir = os.path.dirname(dest_path)

    filename = os.path.basename(dest_path) if not dest_is_dir else "等待服务端返回文件名"
    part_path = os.path.join(download_dir, f".{int(time.time())}.download.part") if dest_is_dir else dest_path + ".part"
    if os.path.exists(part_path):
        print_warning(f"检测到未完成下载，重新下载: {part_path}")
        os.remove(part_path)

    print_info(f"开始下载: {filename}")
    print_info(f"保存目录: {download_dir}")

    downloaded = 0
    start_time = time.time()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "vastai-manager/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if dest_is_dir:
                response_filename = get_response_filename(response)
                if not response_filename:
                    parsed = urllib.parse.urlparse(response.geturl())
                    response_filename = urllib.parse.unquote(os.path.basename(parsed.path))
                if not response_filename or response_filename == "download":
                    raise RuntimeError("服务端未返回有效文件名，无法按原始文件名保存。")
                dest_path = os.path.join(download_dir, response_filename)
                filename = response_filename
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    print_success(f"已存在，跳过: {dest_path} ({format_bytes(os.path.getsize(dest_path))})")
                    return "skipped"
                print_info(f"服务端文件名: {filename}")
                print_info(f"保存路径: {dest_path}")

            with open(part_path, "wb") as out_file:
                total = int(response.headers.get("Content-Length") or 0)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    elapsed = max(time.time() - start_time, 0.001)
                    speed = downloaded / elapsed
                    if total:
                        percent = downloaded / total * 100
                        status = f"进度: {percent:6.2f}% ({format_bytes(downloaded)} / {format_bytes(total)}) | {format_bytes(speed)}/s"
                    else:
                        status = f"进度: {format_bytes(downloaded)} | {format_bytes(speed)}/s"
                    print(f"\r{Colors.BLUE}[信息]{Colors.NC} {status}", end="", flush=True)
        print()
        os.replace(part_path, dest_path)
        print_success(f"下载完成: {dest_path} ({format_bytes(os.path.getsize(dest_path))})")
        return "success"
    except Exception as e:
        print()
        if os.path.exists(part_path):
            os.remove(part_path)
        print_error(f"下载失败: {filename} - {e}")
        return "failed"

def _download_zimage_models_impl(model_specs):
    cleanup_obsolete_zimage_models()
    results = {"success": 0, "skipped": 0, "failed": 0}
    for index, model_spec in enumerate(model_specs, start=1):
        try:
            dest_path = resolve_model_target_path(model_spec)
        except Exception as e:
            print_error(str(e))
            results["failed"] += 1
            continue

        try:
            download_url = build_model_download_url(model_spec)
        except Exception as e:
            print_error(str(e))
            results["failed"] += 1
            continue

        display_name = os.path.basename(dest_path) if os.path.basename(dest_path) else "按服务端原始文件名保存"
        print_info(f"[{index}/{len(model_specs)}] {model_spec['subdir']}/{display_name}")
        status = download_file_with_progress(
            download_url,
            dest_path,
            filename_from_response=model_spec.get("filename_from_response", False),
        )
        results[status] += 1

    print("-" * 60)
    print_success(
        f"Z-Image 模型下载任务完成：成功 {results['success']}，跳过 {results['skipped']}，失败 {results['failed']}。"
    )
    return results["failed"] == 0

def download_zimage_models():
    print_banner()
    print_info("下载 ComfyUI Z-Image 模型")
    print_info(f"目标根目录: {COMFYUI_MODELS_ROOT}")
    _download_zimage_models_impl(get_zimage_model_specs())
    input("\n按回车键返回菜单...")

def download_zimage_turbo_models():
    print_banner()
    print_info("下载 ComfyUI Z-Image-Turbo 模型（仅 HuggingFace 源）")
    print_info(f"目标根目录: {COMFYUI_MODELS_ROOT}")
    _download_zimage_models_impl(ZIMAGE_TURBO_MODELS)
    input("\n按回车键返回菜单...")

def is_running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=1) as r:
            return r.status == 200
    except: return False

# ==========================================
# 功能函数
# ==========================================
def _has_systemd():
    """检测当前环境是否运行 systemd（容器环境通常没有）。"""
    return os.path.isdir("/run/systemd/system")

def _stop_existing_ollama():
    """停止官方脚本/历史进程拉起的 ollama serve，避免与自定义端口/目录冲突。"""
    if _has_systemd():
        run_cmd("systemctl stop ollama.service 2>/dev/null", check=False)
    else:
        print_info("未检测到 systemd，跳过 systemctl 操作。")
    run_cmd("pkill -9 -f 'ollama serve'", check=False)


def _start_ollama_serve(port, host, wait_seconds=10):
    """以脚本配置的端口和模型目录启动 ollama serve；启动前先清理旧进程。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    _stop_existing_ollama()
    time.sleep(1)
    env = {
        "OLLAMA_HOST": f"{host}:{port}",
        "OLLAMA_MODELS": MODELS_DIR,
        "OLLAMA_CONTEXT_LENGTH": "8192",
    }
    subprocess.Popen(
        f"nohup ollama serve >> {os.path.join(LOG_DIR, 'ollama.log')} 2>&1 &",
        shell=True,
        env={**os.environ, **env},
        preexec_fn=os.setpgrp,
    )
    for _ in range(max(1, wait_seconds)):
        time.sleep(1)
        if is_running(port):
            return True
    return is_running(port)


def install_ollama():
    print_banner()
    print_info("步骤 1: 安装/更新 Ollama 并设置端口")

    if not shutil.which("ollama"):
        print_info("正在下载并安装 Ollama (官方脚本)...")
        run_cmd("curl -fsSL https://ollama.com/install.sh | sh")
    else:
        print_success("检测到 Ollama 已安装。")

    # 官方脚本会自动注册并启动 ollama.service（默认端口 11434、ollama 用户、默认模型目录），
    # 与本脚本自定义端口/模型目录会冲突，禁用并停掉，由本脚本自行管理。
    print_info("正在停止并禁用官方脚本默认启动的 Ollama 服务，避免与自定义端口/目录冲突...")
    if _has_systemd():
        run_cmd("systemctl disable ollama.service 2>/dev/null", check=False)
    else:
        print_info("未检测到 systemd，跳过 systemctl 操作。")
    _stop_existing_ollama()

    config = load_config()
    new_port = prompt_for_ollama_port(config)
    save_config(new_port, config['host'])
    print_success(f"端口已配置为: {new_port}")

    config = load_config()
    port, host = config['port'], config['host']

    # 端口确认后开始自动化流水线：启动服务 -> 下载并导入模型
    print_info(f"开始自动化流水线：启动 Ollama 服务并导入模型 {MODEL_NAME}...")
    if not _start_ollama_serve(port, host):
        print_error(f"Ollama 服务启动失败，已中止后续自动化。请查看 {os.path.join(LOG_DIR, 'ollama.log')}。")
        input("\n按回车键返回菜单...")
        return
    print_success(f"Ollama 服务已在端口 {port} 启动。")

    _import_qwen_model(port)
    print_success("步骤 1 自动化流水线已完成。")
    input("\n按回车键返回菜单...")


def _import_qwen_model(port):
    """下载 GGUF 并注册到当前 Ollama，已存在则跳过。不做服务启停，调用方自行保证服务可用。"""
    print_info("正在检查模型列表...")
    models = run_cmd(f"OLLAMA_HOST=127.0.0.1:{port} ollama list", capture=True)
    if models and MODEL_NAME in models:
        print_success(f"模型 {MODEL_NAME} 已存在，跳过导入。")
        return True

    os.makedirs(GGUF_DIR, exist_ok=True)
    gguf_path = os.path.join(GGUF_DIR, GGUF_FILENAME)
    if not os.path.exists(gguf_path):
        print_info("正在从 Hugging Face 下载 GGUF 文件 (4.3GB)...")
        if not run_cmd(f"curl -L -o {gguf_path} {MODEL_URL}"):
            print_error("GGUF 下载失败。")
            return False

    modelfile = "/tmp/Modelfile_qwen"
    with open(modelfile, 'w') as f:
        f.write(f'FROM {gguf_path}\n')
        f.write('TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"""\n')

    print_info("正在注册模型到 Ollama...")
    if not run_cmd(f"ollama create {MODEL_NAME} -f {modelfile}", env={"OLLAMA_HOST": f"127.0.0.1:{port}"}):
        print_error("模型注册失败。")
        return False
    print_success("模型导入完成！")
    return True

def init_qwen_model():
    print_banner()
    print_info("步骤 2: 下载并注册 Qwen 模型")
    config = require_ollama_config()
    port = config['port']

    if not is_running(port):
        print_error(f"Ollama 服务未在端口 {port} 运行，请先执行菜单 4 启动服务。")
        input("\n按回车键返回菜单...")
        return

    _import_qwen_model(port)
    input("\n按回车键返回菜单...")

def download_qwen_rapid_model():
    print_banner()
    print_info("下载 ComfyUI Qwen-Rapid-AIO 模型")
    print_info(f"目标目录: {os.path.join(COMFYUI_MODELS_ROOT, 'checkpoints')}")
    _download_zimage_models_impl([QWEN_RAPID_AIO_MODEL])
    input("\n按回车键返回菜单...")

def manage_autostart():
    print_banner()
    print_info("步骤 3: 设置 Ollama 开机自动启动")
    
    # 检测是否支持 systemd
    has_systemd = os.path.isdir("/run/systemd/system")
    
    if is_root() and has_systemd:
        print_info("检测到 Systemd，正在配置服务...")
        config = require_ollama_config()
        port, host = config['port'], config['host']
        content = f"""[Unit]
Description=Ollama AI Service
After=network.target

[Service]
Type=simple
User={os.getenv('SUDO_USER', 'root')}
Environment="OLLAMA_HOST={host}:{port}"
Environment="OLLAMA_MODELS={MODELS_DIR}"
ExecStart={shutil.which('ollama')} serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
        try:
            with open("/etc/systemd/system/ollama.service", 'w') as f: f.write(content)
            if run_cmd("systemctl daemon-reload && systemctl enable ollama.service"):
                print_success("开机自启动服务已配置成功 (Systemd)！")
            else:
                raise Exception("Systemctl 执行失败")
        except Exception as e:
            print_error(f"Systemd 配置失败: {e}")
            has_systemd = False # 强制进入 crontab 逻辑

    if not has_systemd:
        print_info("环境不支持 Systemd，正在尝试通过 Crontab 配置...")
        config = require_ollama_config()
        port = config['port']
        # 构造重启时运行的命令
        script_path = os.path.abspath(__file__)
        cron_cmd = f"@reboot python3 {script_path} serve --port {port}\n"
        
        try:
            # 读取现有 crontab
            current_cron = run_cmd("crontab -l", capture=True) or ""
            if script_path not in current_cron:
                new_cron = current_cron + "\n" + cron_cmd
                process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE)
                process.communicate(input=new_cron.encode())
                print_success("开机自启动已通过 Crontab 配置成功！")
            else:
                print_success("Crontab 任务已存在。")
        except Exception as e:
            print_error(f"Crontab 配置失败: {e}")
            print_info("请手动尝试: (crontab -l ; echo '@reboot python3 ...') | crontab -")
    
    input("\n按回车键返回菜单...")

def manage_service(action):
    config = require_ollama_config() if action == "start" else load_config()
    port, host = config['port'], config['host']
    if action == "start":
        if is_running(port):
            print_warning(f"服务已在 {port} 运行中。")
        else:
            print_info(f"开启 Ollama 服务 (端口: {port})...")
            if _start_ollama_serve(port, host):
                print_success("服务已启动。")
            else:
                print_error("启动失败，请查看日志。")
    elif action == "stop":
        print_info("正在停止服务...")
        _stop_existing_ollama()
        print_success("服务已关闭。")
    
    input("\n按回车键返回菜单...")

# ==========================================
# 菜单主入口
# ==========================================
def ollama_menu():
    # 首次进入先执行一次全面体检
    print_banner()
    print_info("正在执行系统环境体检...")
    
    while True:
        config = load_config()
        port = config.get('port') or ""
        
        # 1. 检查安装状态
        ollama_path = shutil.which('ollama')
        ollama_ver = run_cmd("ollama --version", capture=True) if ollama_path else "未安装"
        
        # 2. 检查服务状态
        is_up = is_running(port) if port else False
        if port:
            status_str = f"{Colors.GREEN}● 运行中{Colors.NC}" if is_up else f"{Colors.RED}○ 已停止{Colors.NC}"
            port_label = port
        else:
            status_str = f"{Colors.YELLOW}○ 未配置端口{Colors.NC}"
            port_label = "未配置"
        
        # 3. 检查已有的模型
        model_list = "无"
        if is_up:
            raw_models = run_cmd(f"OLLAMA_HOST=127.0.0.1:{port} ollama list", capture=True)
            if raw_models:
                # 提取模型名称列
                lines = raw_models.split('\n')[1:]
                names = [l.split()[0] for l in lines if l.strip()]
                model_list = ", ".join(names) if names else "无"
        
        print_banner()
        print(f"{Colors.BOLD}当前系统状态:{Colors.NC}")
        print(f"  - Ollama 主程序: {Colors.CYAN}{ollama_ver}{Colors.NC} ({ollama_path or 'N/A'})")
        print(f"  - 后台服务状态: {status_str} (端口: {port_label})")
        print(f"  - 已导入的模型: {Colors.YELLOW}{model_list}{Colors.NC}")
        print("-" * 60)
        
        print(f"  {Colors.BOLD}1.{Colors.NC} 安装/更新 Ollama 并配置端口")
        print(f"  {Colors.BOLD}2.{Colors.NC} 下载并导入 Qwen 模型 (自动跳过已存在的)")
        print(f"  {Colors.BOLD}3.{Colors.NC} 配置 Ollama 开机自启动")
        print(f"  {Colors.BOLD}4.{Colors.NC} 启动后台服务")
        print(f"  {Colors.BOLD}5.{Colors.NC} 停止所有 Ollama 进程")
        print(f"  {Colors.BOLD}6.{Colors.NC} 查看日志 (Tail)")
        print(f"  {Colors.BOLD}0.{Colors.NC} 退出")
        print("-" * 60)
        
        choice = input(f"{Colors.BOLD}请选择操作 [0-6]: {Colors.NC}").strip()
        
        if choice == '1': install_ollama()
        elif choice == '2': init_qwen_model()
        elif choice == '3': manage_autostart()
        elif choice == '4': manage_service("start")
        elif choice == '5': manage_service("stop")
        elif choice == '6': 
            log_path = os.path.join(LOG_DIR, 'ollama.log')
            if os.path.exists(log_path):
                os.system(f"tail -n 50 -f {log_path}")
            else:
                print_error("日志文件尚不存在。")
                time.sleep(2)
        elif choice == '0': break

def main_menu():
    while True:
        print_banner()
        print(f"  {Colors.BOLD}1.{Colors.NC} Ollama 服务与模型管理")
        print(f"  {Colors.BOLD}2.{Colors.NC} Vast.ai ComfyUI 本机检查（关键目录/文件/版本/启动链路）")
        print(f"  {Colors.BOLD}3.{Colors.NC} Vast.ai ComfyUI 本机升级（检查→备份→升级→补丁→原方式重启→验证）")
        print(f"  {Colors.BOLD}4.{Colors.NC} 下载 ComfyUI Qwen-Rapid-AIO 模型 (自动跳过已存在的)")
        print(f"  {Colors.BOLD}5.{Colors.NC} 下载 ComfyUI Z-Image-Turbo 模型 (仅 HuggingFace，自动跳过已存在的)")
        print(f"  {Colors.BOLD}0.{Colors.NC} 退出")
        print("-" * 60)

        choice = input(f"{Colors.BOLD}请选择操作 [0-5]: {Colors.NC}").strip()

        if choice == '1': ollama_menu()
        elif choice == '2':
            vast_comfyui_check()
            input("\n按回车键返回菜单...")
        elif choice == '3':
            vast_comfyui_upgrade()
            input("\n按回车键返回菜单...")
        elif choice == '4': download_qwen_rapid_model()
        elif choice == '5': download_zimage_turbo_models()
        elif choice == '0': break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行模式
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=[
            "serve",
            "ollama-menu",
            "init-model",
            "stop",
            "download-zimage-models",
            "download-zimage-turbo-models",
            "download-qwen-rapid-model",
            "vast-comfyui-check",
            "vast-comfyui-upgrade",
        ])
        parser.add_argument("--port")
        parser.add_argument("--host")
        parser.add_argument("--ssh-host", help="可选：远程调试时使用；默认不填则在本机执行")
        parser.add_argument("--ssh-port", help="可选：远程调试时使用；需要和 --ssh-host 同时提供")
        parser.add_argument("--ssh-key", default=VAST_DEFAULT_KEY, help="可选：远程调试 SSH 私钥")
        parser.add_argument("--ssh-user", default=VAST_DEFAULT_USER, help="可选：远程调试 SSH 用户")
        args = parser.parse_args()

        # 如果命令行传了端口，覆盖配置
        if args.port:
            cfg = load_config()
            save_config(args.port, args.host or cfg['host'])

        if args.action == "serve": manage_service("start")
        elif args.action == "ollama-menu": ollama_menu()
        elif args.action == "init-model": init_qwen_model()
        elif args.action == "stop": manage_service("stop")
        elif args.action == "download-zimage-models":
            ok = _download_zimage_models_impl(get_zimage_model_specs())
            sys.exit(0 if ok else 1)
        elif args.action == "download-zimage-turbo-models":
            ok = _download_zimage_models_impl(ZIMAGE_TURBO_MODELS)
            sys.exit(0 if ok else 1)
        elif args.action == "download-qwen-rapid-model":
            ok = _download_zimage_models_impl([QWEN_RAPID_AIO_MODEL])
            sys.exit(0 if ok else 1)
        elif args.action in ("vast-comfyui-check", "vast-comfyui-upgrade"):
            if bool(args.ssh_host) != bool(args.ssh_port):
                parser.error("--ssh-host 和 --ssh-port 需要同时提供；不提供则默认本机执行")
            if args.action == "vast-comfyui-check":
                vast_comfyui_check(args.ssh_host, args.ssh_port, args.ssh_key, user=args.ssh_user)
            else:
                vast_comfyui_upgrade(args.ssh_host, args.ssh_port, args.ssh_key, user=args.ssh_user)
    else:
        main_menu()
