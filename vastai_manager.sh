#!/bin/bash
# 工具入口菜单：Shell 负责选择/收集参数，Python 负责具体业务命令。
#
# 用法：
#   本地：./vastai_manager.sh
#   远程：bash <(curl -fsSL https://raw.githubusercontent.com/wokiiii2025/Vastai/main/vastai_manager.sh)
#
# 环境变量（按需设置）：
#   CIVITAI_API_TOKEN    Civitai API 访问令牌（下载 Civitai 源模型时需要）
#   GITHUB_RAW_BASE      仓库 raw 文件根 URL（已内置默认值，通常无需手动设置）

# ---- 远程引导（首次通过 curl|bash 执行时自动下载 Python 脚本） ----
GITHUB_RAW_BASE="${GITHUB_RAW_BASE:-https://raw.githubusercontent.com/wokiiii2025/Vastai/main}"

resolve_script_dir() {
    local src
    src="${BASH_SOURCE[0]:-$0}"
    if [[ "$src" == /dev/fd/* ]] || [[ "$src" == bash ]] || [[ "$src" == /proc/self/fd/* ]]; then
        echo "/tmp/vastai-manager"
        return
    fi
    cd "$(dirname "$src")" 2>/dev/null && pwd || echo "/tmp/vastai-manager"
}

SCRIPT_DIR="$(resolve_script_dir)"
PYTHON_SCRIPT="$SCRIPT_DIR/vastai_manager.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[引导] 检测到远程执行模式，正在下载管理脚本..."
    mkdir -p "$SCRIPT_DIR"
    curl -fsSL "$GITHUB_RAW_BASE/vastai_manager.py" -o "$PYTHON_SCRIPT" || {
        echo "[错误] 下载 Python 脚本失败。"
        echo "  当前 raw 地址: $GITHUB_RAW_BASE/vastai_manager.py"
        echo "  请检查网络连接，或手动设置 GITHUB_RAW_BASE 环境变量后重试。"
        echo "  亦可手动克隆仓库后本地运行: git clone https://github.com/wokiiii2025/Vastai.git && cd Vastai && ./vastai_manager.sh"
        exit 1
    }
    chmod +x "$PYTHON_SCRIPT" 2>/dev/null || true
    echo "[引导] 脚本就绪: $PYTHON_SCRIPT"
fi

while true; do
    if [[ -t 1 && -n "$TERM" ]]; then
        clear
    fi
    cat <<'EOF'
 __     __        _      _    _
 \ \   / /_ _ ___| |_   / \  (_)
  \ \ / / _` / __| __| / _ \ | |
   \ V / (_| \__ \ |_ / ___ \| |
    \_/ \__,_|___/\__/_/   \_\_|
EOF
    echo
    echo "  1. Ollama 服务与模型管理"
    echo "  2. Vast.ai ComfyUI 本机检查（关键目录/文件/版本/启动链路）"
    echo "  3. Vast.ai ComfyUI 本机升级（检查→备份→升级→补丁→原方式重启→验证）"
    echo "  4. 下载 ComfyUI Qwen-Rapid-AIO 模型（自动跳过已存在的）"
    echo "  5. 下载 ComfyUI Z-Image-Turbo 模型（仅 HuggingFace，自动跳过已存在的）"
    echo "  0. 退出"
    echo
    read -r -p "请选择操作 [0-5]: " choice

    case "$choice" in
        1)
            python3 "$PYTHON_SCRIPT" ollama-menu
            ;;
        2|3)
            if [[ "$choice" == "2" ]]; then
                python3 "$PYTHON_SCRIPT" vast-comfyui-check
            else
                python3 "$PYTHON_SCRIPT" vast-comfyui-upgrade
            fi

            read -r -p "按回车键返回菜单..."
            ;;
        4)
            python3 "$PYTHON_SCRIPT" download-qwen-rapid-model
            read -r -p "按回车键返回菜单..."
            ;;
        5)
            python3 "$PYTHON_SCRIPT" download-zimage-turbo-models
            read -r -p "按回车键返回菜单..."
            ;;
        0)
            exit 0
            ;;
        *)
            echo "无效选项。"
            sleep 1
            ;;
    esac
done
