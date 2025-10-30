#!/bin/bash
# Training Monitor Script

echo "========================================"
echo "AgentWorkflow Training Monitor"
echo "========================================"
echo ""

# Check if training process is running
if ps -p 29507 > /dev/null 2>&1; then
    echo "✓ Training process (PID 29507) is RUNNING"
else
    echo "✗ Training process NOT RUNNING"
fi
echo ""

# Show recent log output
echo "--- Last 20 lines of training log ---"
tail -20 /content/agentworkflow/training_full.log
echo ""

# Check progress
echo "--- Training Progress ---"
grep -E "Problem \d+/\d+|Episode \d+/\d+|Epoch \d+/\d+" /content/agentworkflow/training_full.log | tail -5
echo ""

# Check for errors
echo "--- Recent Errors (if any) ---"
grep -i "error" /content/agentworkflow/training_full.log | tail -3
echo ""

# Check outputs
echo "--- Generated Files ---"
if [ -d "/content/agentworkflow/outputs" ]; then
    echo "Outputs directory: $(du -sh /content/agentworkflow/outputs 2>/dev/null | cut -f1)"
    echo "Results: $(find /content/agentworkflow/outputs/results -name "*.json" 2>/dev/null | wc -l) files"
fi

if [ -d "/content/agentworkflow/checkpoints" ]; then
    echo "Checkpoints: $(find /content/agentworkflow/checkpoints -type f 2>/dev/null | wc -l) files"
fi
echo ""

echo "========================================"
echo "Monitor script completed"
echo "========================================"
