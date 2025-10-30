# AgentWorkflow Training Pipeline

A complete training pipeline combining **MCTS (Monte Carlo Tree Search)** with **GRPO (Group Refined Policy Optimization)** for fine-tuning Qwen models to optimize workflow code generation.

## Architecture

```
┌──────────────────────────────────────────────┐
│      MCTS-based Workflow Optimization       │
│  (AFlow native Optimizer + WorkflowEvaluator)│
└──────────────┬───────────────────────────────┘
               │ (Generate candidates + pass@k scores)
               ▼
┌──────────────────────────────────────────────┐
│   GRPO Policy Training (VeRL-based)         │
│   (Fine-tune Qwen with pass@k rewards)      │
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Checkpoints + Logs to Google Drive          │
└──────────────────────────────────────────────┘
```

## Key Features

- **MCTS Optimization**: Uses AFlow's native Monte Carlo Tree Search for workflow optimization
- **Qwen Fine-tuning**: LoRA-based efficient fine-tuning of Qwen models
- **GRPO Training**: Group Refined Policy Optimization from VeRL framework
- **AIME Evaluation**: Standard AIME24 dataset with 30 problems
- **Checkpoint Management**: Automatic checkpoint saving and Google Drive backup
- **Comprehensive Logging**: Metrics tracking, CSV export, and result logging

## Project Structure

```
agentworkflow/
├── config/
│   └── training_config.yaml          # Training configuration
├── data/
│   └── aime24/                       # AIME24 dataset
│       └── data.json
├── src/
│   ├── eval/
│   │   ├── __init__.py
│   │   └── workflow_evaluator.py     # Evaluation system
│   ├── mcts/
│   │   ├── __init__.py
│   │   ├── mcts_optimizer.py         # High-level MCTS optimizer
│   │   └── aflow_wrapper.py          # AFlow interface
│   ├── grpo/
│   │   ├── __init__.py
│   │   ├── qwen_policy.py            # Qwen model with LoRA
│   │   └── grpo_trainer.py           # GRPO training loop
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # Configuration management
│       ├── backup_manager.py         # Drive backup
│       └── metrics_logger.py         # Metrics tracking
├── train.py                          # Main training script
├── test_e2e.py                       # End-to-end tests
├── checkpoints/                      # Model checkpoints
├── outputs/                          # Training outputs
└── logs/                             # Training logs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- AFlow (available at `/content/AFlow`)
- VeRL (from `/content/verl-agent`)

### Setup

```bash
cd /content/agentworkflow

# Install dependencies
pip install torch transformers peft pyyaml

# Optional: Install for faster training
pip install flash-attn  # For fast attention
```

## Configuration

Edit `config/training_config.yaml` to customize:

```yaml
# Number of training epochs and episodes
training:
  num_epochs: 3              # Total epochs
  num_episodes: 3            # Episodes per epoch
  problems_per_episode: 5    # Problems per episode

# MCTS parameters
mcts:
  num_iterations: 10         # Iterations per round
  num_samples_per_iteration: 3
  num_search_rounds: 5       # Search rounds per problem

# GRPO parameters
grpo:
  learning_rate: 1.0e-5      # Base learning rate
  lora_learning_rate: 5.0e-4 # LoRA learning rate
  num_epochs: 3              # GRPO epochs per episode

# Model
model:
  name: Qwen/Qwen2-7B        # Model name
  use_lora: true             # Use LoRA fine-tuning

# Paths
paths:
  aflow_path: /content/AFlow
  workspace_path: ./outputs
  drive_path: /content/drive/MyDrive/agentworkflow
```

## Usage

### Basic Training

```bash
# Run training with default config
python train.py

# Run with custom config
python train.py --config config/training_config.yaml

# Run on CPU (for testing)
python train.py --device cpu
```

### Testing

```bash
# Run end-to-end tests
python test_e2e.py
```

This tests:
1. Data loading and evaluation
2. Answer validators
3. Configuration loading
4. Qwen policy initialization
5. Metrics logging
6. Backup management

### Output Structure

After training, outputs are organized as:

```
outputs/
├── results.json                    # Training results
checkpoints/
├── epoch_0/
│   ├── mcts/checkpoint.json
│   ├── grpo/                       # GRPO model
│   └── policy/                     # Qwen model
├── epoch_1/
...
logs/
└── training.log                    # Training log
```

Google Drive backup:
```
/content/drive/MyDrive/agentworkflow/
├── checkpoints/                    # Backup checkpoints
├── results/                        # Result JSONs
└── logs/                          # Log files
```

## Training Flow

### 1. Data Loading
- Load AIME24 dataset (30 problems)
- Split 80/20 into train/test
- Initialize WorkflowEvaluator

### 2. MCTS Optimization (Per Problem)
- Initialize AFlow Optimizer
- Run MCTS tree search (5 rounds)
- Generate multiple workflow candidates
- Evaluate candidates against ground truth
- Compute pass@k scores

### 3. Trajectory Collection
- Collect problem + solutions + rewards
- Build trajectory dataset

### 4. GRPO Training
- Initialize Qwen policy with LoRA
- Train on collected trajectories
- Optimize with group-level policy gradients
- Save checkpoint

### 5. Repeat
- Next epoch: Continue with more problems
- Cumulative learning from MCTS-generated data

## Metrics

### MCTS Metrics
- `pass@k`: Percentage of problems solved by top-k candidates
- `num_rounds`: Number of MCTS rounds
- `success`: Whether optimization completed

### GRPO Metrics
- `loss`: Policy training loss
- `entropy`: Action entropy (exploration)
- `value_loss`: Value function loss

### Overall
- Average pass@k across all problems
- Training loss convergence
- Checkpoint quality

## Evaluation

The system uses AIME24 problems with standard evaluation:

- **Answer Extraction**: From `<answer>...</answer>` tags or last number
- **Validation**: Exact numeric match for integer answers
- **Pass@k**: Computed from top-k solutions

Example:
```python
evaluator = WorkflowEvaluator('AIME24', './data/aime24/data.json')
result = evaluator.evaluate_workflow_response(
    generated_text="The answer is <answer>33</answer>",
    problem_id=0,
    split='test'
)
# result: {'problem_id': 0, 'correct': True, ...}
```

## Advanced Usage

### Custom LLM Policy

Replace Qwen with custom model:

```python
from src.grpo import QwenPolicy

class CustomPolicy(QwenPolicy):
    def __init__(self, ...):
        # Load your custom model
        pass
```

### Resume from Checkpoint

```python
# Load checkpoint
trainer = GRPOTrainer(policy, config)
trainer.load_checkpoint('./checkpoints/epoch_2/grpo')

# Continue training
trainer.train(trajectories, num_epochs=3)
```

### Manual Evaluation

```python
from src.eval import WorkflowEvaluator

evaluator = WorkflowEvaluator('AIME24', './data/aime24/data.json')

# Evaluate single response
result = evaluator.evaluate_workflow_response(response, problem_id=0)

# Evaluate batch
batch_result = evaluator.evaluate_batch(responses, [0, 1, 2])
print(f"Accuracy: {batch_result['accuracy']:.2%}")
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use `device: cpu` for debugging
- Enable gradient checkpointing

### Model Download Slow
- Pre-download model: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-7B')"`
- Use `--cache-dir` to specify cache location

### AFlow Import Errors
- Ensure AFlow is at `/content/AFlow`
- Check `paths.aflow_path` in config

## Performance Tips

1. **Batch Size**: Larger batches (8-16) for faster training
2. **LoRA Rank**: Increase to 16-32 for better quality
3. **Learning Rate**: Adjust based on loss curves
4. **MCTS Rounds**: Increase for more thorough search

## Contributing

To extend the pipeline:

1. **Add new dataset**: Subclass `DatasetEvaluator` in `src/eval/workflow_evaluator.py`
2. **Custom policy**: Inherit from `QwenPolicy` in `src/grpo/qwen_policy.py`
3. **New metrics**: Add to `src/utils/metrics_logger.py`

## References

- **AFlow**: Native workflow optimization with MCTS
- **VeRL**: Group Refined Policy Optimization framework
- **AIME**: American Invitational Mathematics Examination
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning

## License

Internal use only

## Contact

For issues or questions, refer to the summary from the training session.
