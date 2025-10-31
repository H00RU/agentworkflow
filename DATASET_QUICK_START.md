# 数据集使用快速指南

## 快速开始（5 分钟）

### 1. 查看支持的数据集

```bash
python test_dataset_config.py
```

输出会列出所有支持的 5 个数据集及其配置。

### 2. 切换训练数据集

编辑 `config/training_config.yaml`，修改第 8 行：

```yaml
# 改成想要的数据集
dataset:
  name: MATH    # 改为: AIME24, MATH, GSM8K, HumanEval, MBPP
```

### 3. 运行训练

```bash
python train.py --config config/training_config.yaml
```

系统会自动：
- ✓ 验证数据集是否支持
- ✓ 找到正确的数据文件路径
- ✓ 选择合适的评估器
- ✓ 配置正确的问题类型

## 数据集信息速查表

| 数据集 | 类型 | 问题类型 | 数据格式 | 推荐用途 |
|--------|------|---------|---------|---------|
| **AIME24** | MATH | Math | JSON | 数学竞赛问题 |
| **MATH** | MATH | Math | JSONL | 数学推理 |
| **GSM8K** | GSM8K | Math | JSONL | 算术推理 |
| **HumanEval** | HumanEval | Code | JSONL | 代码生成 |
| **MBPP** | MBPP | Code | JSONL | 编程问题 |

## 常见问题解答

### Q1: 如何在不修改代码的情况下切换数据集？

**答**: 只需修改 `config/training_config.yaml` 中的 `dataset.name` 字段：

```yaml
dataset:
  name: GSM8K  # 改这一行
```

### Q2: 如果不指定 `data_path` 会怎样？

**答**: 系统会自动使用该数据集的默认路径。例如：
- MATH → `./data/math/math_validate.jsonl`
- GSM8K → `./data/gsm8k/gsm8k_validate.jsonl`

### Q3: 如何使用自定义的数据文件？

**答**: 在 `config/training_config.yaml` 中指定 `data_path`：

```yaml
dataset:
  name: MATH
  data_path: /path/to/my/custom/math_data.jsonl
```

### Q4: 如何添加新的数据集？

**答**: 编辑 `src/utils/dataset_config.py`，在 `DATASET_REGISTRY` 中添加：

```python
DATASET_REGISTRY['MyDataset'] = DatasetMetadata(
    name='MyDataset',
    aflow_type='MATH',
    question_type='math',
    validator_type='numeric',
    data_path_template='my_data/dataset.jsonl'
)
```

然后在配置中使用：
```yaml
dataset:
  name: MyDataset
```

### Q5: 支持的数据格式是什么？

**答**:
- **JSON** (.json): 数组格式
  ```json
  [
    {"question": "...", "answer": 42, ...},
    {"question": "...", "answer": 123, ...}
  ]
  ```

- **JSONL** (.jsonl): 每行一个 JSON 对象
  ```
  {"problem": "...", "solution": "42"}
  {"problem": "...", "solution": "123"}
  ```

系统会自动规范化字段名（`problem` → `question`，`solution` → `answer`）。

### Q6: 错误提示"数据集不支持"怎么办？

**答**: 检查：
1. 数据集名称拼写是否正确
2. 数据文件是否存在
3. 运行 `test_dataset_config.py` 查看支持的数据集

### Q7: 我能同时使用两个数据集训练吗？

**答**: 目前配置系统一次只支持一个数据集。如需多数据集训练，可以创建多个配置文件：

```bash
# config/math_config.yaml
dataset:
  name: MATH

# config/code_config.yaml
dataset:
  name: HumanEval

# 分别训练
python train.py --config config/math_config.yaml
python train.py --config config/code_config.yaml
```

## 系统架构速览

```
配置文件 (training_config.yaml)
    ↓
config_loader.py (验证和加载)
    ↓
dataset_config.py (获取配置)
    ↓
workflow_evaluator.py (选择合适的评估器)
    ↓
train.py (启动训练)
```

## 验证数据集配置

要验证某个数据集的配置是否正确：

```python
from src.utils.dataset_config import get_dataset_config

config = get_dataset_config('MATH')
print(f"Question type: {config.question_type}")
print(f"AFlow type: {config.aflow_type}")
print(f"Data path: {config.get_data_path()}")
print(f"Description: {config.description}")
```

## 调试技巧

### 查看系统如何解析配置

编辑 `train.py`，在初始化前添加：

```python
from utils.dataset_config import get_dataset_config

dataset_name = config['dataset']['name']
dataset_config = get_dataset_config(dataset_name)

logger.info(f"Dataset: {dataset_config.name}")
logger.info(f"Question Type: {dataset_config.question_type}")
logger.info(f"AFlow Type: {dataset_config.aflow_type}")
logger.info(f"Data Path: {dataset_config.get_data_path()}")
```

### 检查数据文件格式

```python
import json

# 对于 JSON 文件
with open('data.json') as f:
    data = json.load(f)
    print(f"Total problems: {len(data)}")
    print(f"First problem: {data[0]}")

# 对于 JSONL 文件
with open('data.jsonl') as f:
    for i, line in enumerate(f):
        if i == 0:
            print(f"First problem: {json.loads(line)}")
            break
```

## 性能提示

- **数据集大小**: MATH (100K+) > GSM8K (8K) > AIME (30)
- **建议**:
  - 开发测试用 AIME (快速)
  - 完整训练用 MATH 或 GSM8K (平衡)
  - 代码任务用 HumanEval 或 MBPP

## 获取帮助

1. 查看 `IMPLEMENTATION_COMPLETE.md` 了解详细的技术实现
2. 查看 `GENERALIZATION_PLAN.md` 了解设计理念
3. 运行 `test_dataset_config.py` 验证系统
4. 检查 `src/utils/dataset_config.py` 了解所有可用函数
