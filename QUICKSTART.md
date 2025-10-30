# Quick Start Guide

Get the training pipeline running in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (recommended, CPU works too)
- AFlow available at `/content/AFlow`
- AIME dataset already loaded

## 1. Verify Setup

```bash
# Check AFlow exists
ls /content/AFlow

# Check AIME data
ls ./data/aime24/data.json

# Check config
ls ./config/
```

## 2. Run End-to-End Tests

```bash
# Test all components
python test_e2e.py

# Expected output:
# ✓ Data loading test PASSED
# ✓ Validators test PASSED
# ✓ Config loading test PASSED
# ✓ Qwen policy test PASSED
# ✓ Metrics logger test PASSED
# ✓ Backup manager test PASSED
```

## 3. Quick Test Run (5 min)

```bash
# Run minimal test configuration
python train.py --config config/minimal_test.yaml

# Configuration:
# - 1 epoch, 1 episode, 2 problems
# - MCTS: 3 iterations, 2 rounds
# - GRPO: 1 epoch training
# - Output to ./outputs_test/
```

## 4. Full Training (hours)

```bash
# Run full training
python train.py --config config/training_config.yaml

# Configuration:
# - 3 epochs, 3 episodes per epoch, 5 problems per episode
# - MCTS: 10 iterations, 5 rounds
# - GRPO: 3 epochs training per episode
# - Output to ./outputs/ and Google Drive
```

## 5. Monitor Training

```bash
# View logs in real-time
tail -f logs/training.log

# Check outputs
ls outputs/
ls checkpoints/

# View results
cat outputs/results.json | python -m json.tool
```

## Common Commands

### Run with CPU (for testing)
```bash
python train.py --config config/minimal_test.yaml --device cpu
```

### Use custom config
```bash
python train.py --config config/your_config.yaml
```

### Check dataset
```bash
python -c "
from src.eval import WorkflowEvaluator
ev = WorkflowEvaluator('AIME24', './data/aime24/data.json')
print(ev.get_dataset_info())
"
```

### Evaluate single problem
```bash
python -c "
from src.eval import WorkflowEvaluator
ev = WorkflowEvaluator('AIME24', './data/aime24/data.json')
problem = ev.get_problem(0, split='test')
print(f'Problem 0: {problem[\"question\"][:100]}...')
print(f'Answer: {problem[\"answer\"]}')
"
```

## File Structure After Training

```
agentworkflow/
├── outputs/                       # Training outputs
│   └── results.json
├── outputs_test/                  # Test run outputs
│   └── results.json
├── checkpoints/                   # Model checkpoints
│   ├── epoch_0/
│   │   ├── mcts/checkpoint.json
│   │   ├── grpo/                  # GRPO model
│   │   └── policy/                # Qwen model
│   └── ...
├── logs/                          # Training logs
│   ├── training.log
│   └── test/
└── test_results.json              # Test results
```

## Configuration Tips

### For Faster Training
```yaml
training:
  num_epochs: 1           # Fewer epochs
  num_episodes: 1         # Fewer episodes
  problems_per_episode: 3 # Fewer problems

mcts:
  num_iterations: 5       # Fewer iterations
  num_search_rounds: 3    # Fewer rounds

grpo:
  num_epochs: 1           # Fewer GRPO epochs
  batch_size: 8           # Larger batches
```

### For Better Quality
```yaml
training:
  num_epochs: 5           # More epochs
  num_episodes: 5         # More episodes
  problems_per_episode: 10 # More problems

mcts:
  num_iterations: 20      # More iterations
  num_search_rounds: 10   # More rounds

grpo:
  num_epochs: 5           # More GRPO epochs
  learning_rate: 2.0e-5   # Higher learning rate
```

### For Low Memory (CPU)
```yaml
training:
  device: cpu
  num_epochs: 1
  num_episodes: 1
  problems_per_episode: 1

grpo:
  batch_size: 1
  gradient_accumulation_steps: 1

model:
  name: Qwen/Qwen2-7B     # Still 7B, but runs on CPU (slow)
```

## Troubleshooting

### ImportError: No module named 'src'
```bash
# Make sure you're in agentworkflow directory
cd /content/agentworkflow
python train.py
```

### CUDA out of memory
```bash
# Use CPU or reduce batch size
python train.py --device cpu

# Or edit config:
# grpo:
#   batch_size: 1
```

### AFlow not found
```bash
# Check AFlow path in config
# Should be: /content/AFlow
ls /content/AFlow/scripts/optimizer.py

# Update config if needed:
# paths:
#   aflow_path: /content/AFlow
```

### Model download fails
```bash
# Pre-download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-7B')"

# Or use cached version if available
```

### Google Drive backup fails
```bash
# Check mount point
ls /content/drive/MyDrive/

# Ensure drive_path in config is correct:
# paths:
#   drive_path: /content/drive/MyDrive/agentworkflow
```

## Understanding the Output

### training.log
```
2024-10-30 10:00:00 - __main__ - INFO - Epoch 1/3
2024-10-30 10:00:10 - __main__ - INFO - Episode 1/3
2024-10-30 10:00:20 - __main__ - INFO - Problem 1/5
2024-10-30 10:00:30 - __main__ - INFO - MCTS optimization result: pass@k=0.75
2024-10-30 10:01:00 - __main__ - INFO - Training GRPO on 5 trajectories
2024-10-30 10:02:00 - __main__ - INFO - Epoch 0 complete: Loss=0.3421
```

### results.json
```json
{
  "epochs": [
    {
      "epoch": 0,
      "episodes": [
        {
          "episode": 0,
          "mcts_results": [...],
          "trajectories": [...]
        }
      ],
      "grpo_results": {...}
    }
  ],
  "total_mcts_problems": 15,
  "total_grpo_steps": 450
}
```

## Performance Expectations

- **Quick Test**: ~5-10 minutes
- **Full Training**: ~2-4 hours (with GPU)
- **CPU Training**: ~8-12 hours
- **Checkpoint Size**: ~2GB per epoch

## Next Steps

1. ✅ Verify with `test_e2e.py`
2. ✅ Run quick test: `python train.py --config config/minimal_test.yaml`
3. ✅ Check results: `cat outputs_test/results.json`
4. ✅ Run full training: `python train.py --config config/training_config.yaml`
5. ✅ Monitor training: `tail -f logs/training.log`
6. ✅ Review checkpoints: `ls checkpoints/`
7. ✅ Check Drive backup: `ls /content/drive/MyDrive/agentworkflow/`

## Getting Help

- Check `README.md` for detailed documentation
- Check `PROJECT_SUMMARY.md` for implementation details
- Review logs for error messages
- Run `test_e2e.py` to verify all components work

## Success Criteria

Training is successful if:
- ✅ Tests pass: `python test_e2e.py` outputs all PASSED
- ✅ Training starts: No import or initialization errors
- ✅ Checkpoints save: Files appear in `./checkpoints/`
- ✅ Results logged: `results.json` contains training metrics
- ✅ Drive backup: Files copied to Google Drive

Good luck! 🚀
