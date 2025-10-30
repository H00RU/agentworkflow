# AgentWorkflow 安装指南

## 快速安装

### 一键安装脚本

```bash
cd /content/agentworkflow
chmod +x install_dependencies.sh
./install_dependencies.sh
```

这个脚本会自动:
1. ✅ 检查 Python 版本 (需要 >= 3.8)
2. ✅ 升级 pip
3. ✅ 安装基础依赖 (numpy, pandas, pyyaml, etc.)
4. ✅ 安装 PyTorch (CUDA 11.8)
5. ✅ 安装 Transformers 和模型库
6. ✅ 安装 AFlow 依赖
7. ✅ 配置 Qwen 模型路径
8. ✅ 验证所有安装

## 安装内容详解

### 1. 基础 Python 依赖

```bash
numpy
pandas
pyyaml
tqdm
tenacity
openai
anthropic
```

### 2. PyTorch 深度学习框架

```bash
torch
torchvision
torchaudio
```

**注意**: 默认安装 CUDA 11.8 版本。如果需要 CPU 版本或其他 CUDA 版本，请修改脚本。

### 3. Transformers 和模型相关

```bash
transformers       # HuggingFace 模型库
accelerate        # 加速训练
peft              # 参数高效微调 (LoRA)
sentencepiece     # Tokenizer
safetensors       # 安全的模型格式
```

### 4. AFlow 依赖

从 `AFlow/requirements.txt` 安装:
```bash
aiofiles
tree-sitter
sympy
regex
... (更多依赖)
```

### 5. Qwen 模型配置

脚本会自动检测本地模型路径: `/root/models/Qwen2.5-7B-Instruct`

#### 如果本地模型存在:
- ✅ 创建软链接: `./models -> /root/models/Qwen2.5-7B-Instruct`
- ✅ 设置环境变量: `QWEN_MODEL_PATH`
- ✅ 生成配置文件: `./model_config.yaml`

#### 如果本地模型不存在:
- ⚠️ 训练时会从 HuggingFace 自动下载
- ⚠️ 模型大小约 15GB，首次运行需要时间

## 验证安装

### 方法 1: 快速验证脚本

```bash
./quick_verify.sh
```

输出示例:
```
========================================
快速验证 AgentWorkflow 安装
========================================

[1] Python 版本:
Python 3.10.12

[2] PyTorch 和 CUDA:
PyTorch: 2.1.0
CUDA Available: True
CUDA Version: 11.8

[3] Transformers:
Transformers: 4.35.0

[4] 模型路径:
✓ 模型软链接存在: /root/models/Qwen2.5-7B-Instruct

[5] AFlow:
✓ AFlow 可用

[6] 数据集:
✓ AIME24 数据集可用
```

### 方法 2: 运行整合验证

```bash
python verify_integration.py
```

### 方法 3: 运行端到端测试

```bash
python test_e2e.py
```

预期输出:
```
✓ Data loading test PASSED
✓ Validators test PASSED
✓ Config loading test PASSED
✓ Qwen policy test PASSED
✓ Metrics logger test PASSED
✓ Backup manager test PASSED
```

## 手动安装 (可选)

如果自动安装脚本失败，可以手动安装:

### Step 1: 基础依赖

```bash
pip install numpy pandas pyyaml tqdm tenacity openai
```

### Step 2: PyTorch

```bash
# CUDA 11.8 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch torchvision torchaudio
```

### Step 3: Transformers

```bash
pip install transformers accelerate peft sentencepiece safetensors
```

### Step 4: AFlow 依赖

```bash
cd AFlow
pip install -r requirements.txt
cd ..
```

### Step 5: 配置模型

如果有本地 Qwen 模型:
```bash
# 创建软链接
ln -s /root/models/Qwen2.5-7B-Instruct ./models

# 设置环境变量
export QWEN_MODEL_PATH="/root/models/Qwen2.5-7B-Instruct"
```

## 常见问题

### Q1: PyTorch CUDA 安装失败

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决**:
```bash
# 检查 CUDA 版本
nvcc --version

# 重新安装对应版本的 PyTorch
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q2: Transformers 下载模型很慢

**问题**: HuggingFace 模型下载缓慢

**解决**:
```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后放到本地
# 然后在配置中使用本地路径
```

### Q3: 内存不足 (CUDA OOM)

**问题**: 训练时 GPU 内存不足

**解决**:
```yaml
# 编辑 config/training_config.yaml
grpo:
  batch_size: 1              # 减小批次大小
  gradient_accumulation_steps: 8  # 增加梯度累积

model:
  use_lora: true             # 使用 LoRA 减少内存
  lora_rank: 4               # 减小 LoRA rank
```

或使用 CPU:
```bash
python train.py --config config/minimal_test.yaml --device cpu
```

### Q4: AFlow 导入失败

**问题**: `ImportError: No module named 'scripts.optimizer'`

**解决**:
```bash
# 检查 AFlow 目录
ls AFlow/scripts/optimizer.py

# 如果不存在，重新复制 AFlow
cp -r /content/AFlow ./

# 或重新运行整合验证
python verify_integration.py
```

### Q5: 本地模型找不到

**问题**: 本地 Qwen 模型不存在

**解决**:

**选项 1**: 从 HuggingFace 下载
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

**选项 2**: 手动下载并配置
```bash
# 下载到本地
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# 创建配置
cd ..
echo "export QWEN_MODEL_PATH='$(pwd)/models/Qwen2.5-7B-Instruct'" >> ~/.bashrc
```

## 不同环境的安装

### Google Colab

```bash
# Colab 已预装 PyTorch 和基础库
!cd /content
!git clone <your-repo>
!cd agentworkflow
!./install_dependencies.sh
```

### 本地服务器

```bash
# 确保有 CUDA 驱动
nvidia-smi

# 克隆项目
git clone <your-repo>
cd agentworkflow

# 运行安装脚本
./install_dependencies.sh
```

### Docker 环境

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y git
RUN ./install_dependencies.sh
```

## 验证清单

安装成功后，确认以下内容:

- [ ] Python 版本 >= 3.8
- [ ] PyTorch 已安装且 CUDA 可用 (如有 GPU)
- [ ] Transformers 版本 >= 4.30
- [ ] AFlow 目录存在且可导入
- [ ] AIME24 数据集存在
- [ ] Qwen 模型配置正确 (本地或在线)
- [ ] 所有测试通过: `python test_e2e.py`
- [ ] 整合验证通过: `python verify_integration.py`

## 卸载

如果需要卸载:

```bash
# 卸载 Python 包
pip uninstall -y torch transformers peft accelerate

# 删除模型软链接
rm -f ./models

# 清理配置
rm -f ./model_config.yaml

# 清理生成的脚本
rm -f ./quick_verify.sh
```

## 下一步

安装完成后:

1. ✅ 运行快速验证: `./quick_verify.sh`
2. ✅ 运行端到端测试: `python test_e2e.py`
3. ✅ 运行最小化训练测试: `python train.py --config config/minimal_test.yaml`
4. ✅ 查看结果: `cat outputs_test/results.json`

## 获取帮助

- 查看项目文档: `README.md`
- 查看快速入门: `QUICKSTART.md`
- 查看项目总结: `PROJECT_SUMMARY.md`
- 检查日志文件: `logs/training.log`

## 技术支持

如果遇到问题:
1. 检查 Python 和依赖版本
2. 查看详细错误日志
3. 运行 `./quick_verify.sh` 诊断
4. 参考上面的常见问题部分
