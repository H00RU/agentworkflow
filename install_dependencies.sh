#!/bin/bash
# AgentWorkflow 完整依赖安装脚本
# Complete dependency installation script for AgentWorkflow
#
# 功能:
# 1. 安装 Python 基础依赖
# 2. 安装 PyTorch 和 transformers
# 3. 安装 AFlow 依赖
# 4. 配置 Qwen 模型路径
# 5. 验证安装

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "AgentWorkflow 依赖安装脚本"
echo "AgentWorkflow Dependency Installation Script"
echo "========================================================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}$1${NC}"
    echo "========================================================================"
}

# 检查是否在正确的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "当前目录: $SCRIPT_DIR"
print_info "工作目录: $(pwd)"

# ============================================================================
# 步骤 1: 检查 Python 版本
# ============================================================================
print_step "[1/8] 检查 Python 版本"

if ! command -v python3 &> /dev/null; then
    print_error "Python3 未安装！"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_info "Python 版本: $PYTHON_VERSION"

PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "需要 Python 3.8 或更高版本！当前版本: $PYTHON_VERSION"
    exit 1
fi

print_info "✓ Python 版本满足要求"

# ============================================================================
# 步骤 2: 升级 pip
# ============================================================================
print_step "[2/8] 升级 pip"

python3 -m pip install --upgrade pip
print_info "✓ pip 升级完成"

# ============================================================================
# 步骤 3: 安装基础依赖
# ============================================================================
print_step "[3/8] 安装基础 Python 依赖"

print_info "安装 numpy, pandas, pyyaml..."
pip3 install numpy pandas pyyaml

print_info "安装 tqdm, tenacity..."
pip3 install tqdm tenacity

print_info "安装 openai, anthropic (API 客户端)..."
pip3 install openai anthropic

print_info "✓ 基础依赖安装完成"

# ============================================================================
# 步骤 4: 安装 PyTorch
# ============================================================================
print_step "[4/8] 安装 PyTorch"

if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    print_warning "PyTorch 已安装 (版本: $TORCH_VERSION)"
    read -p "是否重新安装? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "重新安装 PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "跳过 PyTorch 安装"
    fi
else
    print_info "安装 PyTorch (CUDA 11.8)..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

print_info "✓ PyTorch 安装完成"

# ============================================================================
# 步骤 5: 安装 Transformers 和相关库
# ============================================================================
print_step "[5/8] 安装 Transformers 和模型相关库"

print_info "安装 transformers, accelerate, peft..."
pip3 install transformers accelerate peft

print_info "安装 sentencepiece, protobuf (tokenizer 依赖)..."
pip3 install sentencepiece protobuf

print_info "安装 safetensors (模型加载)..."
pip3 install safetensors

print_info "✓ Transformers 相关库安装完成"

# ============================================================================
# 步骤 6: 安装 AFlow 依赖
# ============================================================================
print_step "[6/8] 安装 AFlow 依赖"

if [ -f "AFlow/requirements.txt" ]; then
    print_info "从 AFlow/requirements.txt 安装依赖..."
    pip3 install -r AFlow/requirements.txt
    print_info "✓ AFlow 依赖安装完成"
else
    print_warning "未找到 AFlow/requirements.txt，跳过"
fi

# 安装额外的 AFlow 可能需要的包
print_info "安装额外依赖: aiofiles, tree-sitter, sympy..."
pip3 install aiofiles tree-sitter sympy

print_info "✓ AFlow 依赖安装完成"

# ============================================================================
# 步骤 7: 配置 Qwen 模型
# ============================================================================
print_step "[7/8] 配置 Qwen 模型"

MODEL_PATH="/root/models/Qwen2.5-7B-Instruct"

if [ -d "$MODEL_PATH" ]; then
    print_info "✓ 发现本地 Qwen 模型: $MODEL_PATH"

    # 检查模型文件完整性
    if [ -f "$MODEL_PATH/config.json" ] && \
       [ -f "$MODEL_PATH/tokenizer.json" ] && \
       [ -f "$MODEL_PATH/model.safetensors.index.json" ]; then
        print_info "✓ 模型文件完整"

        # 创建软链接到项目目录
        if [ ! -L "./models" ]; then
            print_info "创建模型软链接..."
            ln -s "$MODEL_PATH" ./models
            print_info "✓ 软链接创建完成: ./models -> $MODEL_PATH"
        else
            print_info "软链接已存在: ./models"
        fi

        # 设置环境变量
        echo "export QWEN_MODEL_PATH=\"$MODEL_PATH\"" >> ~/.bashrc
        export QWEN_MODEL_PATH="$MODEL_PATH"
        print_info "✓ 环境变量已设置: QWEN_MODEL_PATH"

        # 创建配置文件
        cat > ./model_config.yaml << EOF
# Qwen 模型配置
model:
  name: Qwen2.5-7B-Instruct
  path: $MODEL_PATH
  local: true
  use_cache: true

# 如果本地模型不可用，使用 HuggingFace
fallback:
  name: Qwen/Qwen2.5-7B-Instruct
  local: false
EOF
        print_info "✓ 模型配置文件已创建: ./model_config.yaml"

    else
        print_warning "模型文件不完整，训练时将从 HuggingFace 下载"
    fi
else
    print_warning "本地模型路径不存在: $MODEL_PATH"
    print_warning "训练时将从 HuggingFace 下载模型"

    # 创建配置文件
    cat > ./model_config.yaml << EOF
# Qwen 模型配置
model:
  name: Qwen/Qwen2.5-7B-Instruct
  local: false
  use_cache: true
EOF
    print_info "✓ 模型配置文件已创建: ./model_config.yaml (使用 HuggingFace)"
fi

# ============================================================================
# 步骤 8: 验证安装
# ============================================================================
print_step "[8/8] 验证安装"

print_info "检查关键依赖..."

# 验证 Python 包
PACKAGES=(
    "numpy"
    "pandas"
    "torch"
    "transformers"
    "peft"
    "yaml"
    "openai"
)

ALL_GOOD=true
for package in "${PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))")
        print_info "✓ $package (version: $VERSION)"
    else
        print_error "✗ $package 未安装或导入失败"
        ALL_GOOD=false
    fi
done

# 验证 CUDA 可用性
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    print_info "✓ CUDA 可用 (版本: $CUDA_VERSION, GPU 数量: $GPU_COUNT)"
    print_info "  GPU: $GPU_NAME"
else
    print_warning "CUDA 不可用，将使用 CPU 训练"
fi

# 验证 AFlow
if [ -f "AFlow/scripts/optimizer.py" ]; then
    print_info "✓ AFlow 可用"
else
    print_error "✗ AFlow 未找到"
    ALL_GOOD=false
fi

# 验证数据集
if [ -f "data/aime24/data.json" ]; then
    print_info "✓ AIME24 数据集可用"
else
    print_warning "AIME24 数据集未找到: data/aime24/data.json"
fi

# ============================================================================
# 安装完成
# ============================================================================
echo ""
echo "========================================================================"
if [ "$ALL_GOOD" = true ]; then
    print_info "✓✓✓ 所有依赖安装成功！✓✓✓"
    echo "========================================================================"
    echo ""
    echo "下一步操作:"
    echo "  1. 运行测试: python test_e2e.py"
    echo "  2. 验证整合: python verify_integration.py"
    echo "  3. 快速测试: python train.py --config config/minimal_test.yaml"
    echo "  4. 完整训练: python train.py --config config/training_config.yaml"
    echo ""
    if [ -d "$MODEL_PATH" ]; then
        echo "本地模型路径: $MODEL_PATH"
        echo "模型软链接: ./models -> $MODEL_PATH"
    fi
    echo ""
else
    print_error "某些依赖安装失败，请检查上面的错误信息"
    echo "========================================================================"
    exit 1
fi

# 创建快速验证脚本
cat > ./quick_verify.sh << 'EOFVERIFY'
#!/bin/bash
# 快速验证脚本
echo "========================================"
echo "快速验证 AgentWorkflow 安装"
echo "========================================"
echo ""
echo "[1] Python 版本:"
python3 --version
echo ""
echo "[2] PyTorch 和 CUDA:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "[3] Transformers:"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""
echo "[4] 模型路径:"
if [ -L "./models" ]; then
    echo "✓ 模型软链接存在: $(readlink -f ./models)"
elif [ -n "$QWEN_MODEL_PATH" ]; then
    echo "✓ 环境变量: QWEN_MODEL_PATH=$QWEN_MODEL_PATH"
else
    echo "⚠ 使用 HuggingFace 在线模型"
fi
echo ""
echo "[5] AFlow:"
if [ -f "AFlow/scripts/optimizer.py" ]; then
    echo "✓ AFlow 可用"
else
    echo "✗ AFlow 未找到"
fi
echo ""
echo "[6] 数据集:"
if [ -f "data/aime24/data.json" ]; then
    echo "✓ AIME24 数据集可用"
else
    echo "⚠ AIME24 数据集未找到"
fi
echo ""
echo "========================================"
EOFVERIFY

chmod +x ./quick_verify.sh
print_info "✓ 快速验证脚本已创建: ./quick_verify.sh"

echo ""
print_info "安装完成！现在可以运行: ./quick_verify.sh 进行快速验证"
