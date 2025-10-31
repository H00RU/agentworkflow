# 项目通用化实现方案

## 当前问题分析

### 1. question_type 硬编码
**位置**: train.py 第 295 行
```python
question_type = 'math'  # AIME24 is math problems
```
**问题**: 所有数据集都被强制设为 'math'

### 2. 配置文件路径特定
**位置**: config/training_config.yaml 第 6-7 行
```yaml
dataset:
  name: AIME24
  data_path: ./data/aime24/data.json
```
**问题**: 硬编码 AIME24 和特定路径

### 3. 类命名过于特定
**位置**: src/eval/workflow_evaluator.py 第 180 行
```python
class AIEMEvaluator(DatasetEvaluator):
```
**问题**: 类名暴露了实现细节

### 4. 缺少数据集元数据
**问题**: 没有一个统一的地方定义各数据集的属性

---

## 实现方案

### 第1步：创建数据集配置系统

**新文件**: `src/utils/dataset_config.py`

数据集注册表将定义：
- 每个数据集的 AFlow 类型
- 问题类型（math/code/qa）
- 验证器类型（numeric/code/string）
- 数据文件路径模板

### 第2步：重构 WorkflowEvaluator

**修改**: `src/eval/workflow_evaluator.py`

- 重命名 `AIEMEvaluator` → `MathDatasetEvaluator`
- 重命名 `HumanEvalEvaluator` → `CodeDatasetEvaluator`
- 添加 `StringAnswerValidator` 用于通用文本答案
- 支持多种文件格式（JSON 和 JSONL）

### 第3步：修改 train.py

**修改**: `train.py` 第 283-307 行

将硬编码的 dataset_type_map 和 question_type 替换为从配置系统获取。

### 第4步：更新配置文件系统

**修改**: `src/utils/config_loader.py`

- 添加数据集验证
- 自动补充默认数据路径

### 第5步：更新配置文件

**修改**: `config/training_config.yaml`

更改为通用配置，支持多种数据集。

---

## 修改清单

| 文件 | 类型 | 操作 |
|------|------|------|
| `src/utils/dataset_config.py` | 新文件 | 创建数据集配置系统 |
| `src/eval/workflow_evaluator.py` | 修改 | 重构评估器，支持多数据集 |
| `train.py` | 修改 | 使用配置系统替换硬编码 |
| `src/utils/config_loader.py` | 修改 | 添加配置验证 |
| `config/training_config.yaml` | 修改 | 更新为通用配置 |

---

## 优势

1. ✅ **零硬编码** - 所有配置中心化
2. ✅ **易扩展** - 添加新数据集只需在注册表中添加一行
3. ✅ **类型安全** - 使用 dataclass 避免魔法字符串
4. ✅ **向后兼容** - 现有代码继续工作
5. ✅ **明确的元数据** - 每个数据集的属性一目了然
6. ✅ **支持多格式** - JSON 和 JSONL 都支持
