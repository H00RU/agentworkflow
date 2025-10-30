#!/bin/bash
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        AgentWorkflow Training Status Report              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check process
if ps -p 29507 > /dev/null 2>&1; then
    RUNTIME=$(ps -p 29507 -o etime= | tr -d ' ')
    echo "✓ Training Process: RUNNING (Runtime: $RUNTIME)"
else
    echo "✗ Training Process: STOPPED"
fi
echo ""

# Extract progress info
echo "📊 Training Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LAST_EPOCH=$(grep "Epoch [0-9]" /content/agentworkflow/training_full.log | tail -1)
LAST_EPISODE=$(grep "Episode [0-9]" /content/agentworkflow/training_full.log | tail -1)
LAST_PROBLEM=$(grep "Problem [0-9]" /content/agentworkflow/training_full.log | tail -1)

echo "  Last Epoch:   $LAST_EPOCH"
echo "  Last Episode: $LAST_EPISODE"
echo "  Last Problem: $LAST_PROBLEM"
echo ""

# Count completed
TOTAL_PROBLEMS=$(grep -c "Evaluation complete" /content/agentworkflow/training_full.log)
echo "  Completed Problems: $TOTAL_PROBLEMS"
echo ""

# File statistics
echo "📁 Generated Outputs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "/content/agentworkflow/outputs" ]; then
    RESULTS_COUNT=$(find /content/agentworkflow/outputs/results -name "*.json" 2>/dev/null | wc -l)
    OUTPUTS_SIZE=$(du -sh /content/agentworkflow/outputs 2>/dev/null | cut -f1)
    echo "  Results Files: $RESULTS_COUNT"
    echo "  Total Size: $OUTPUTS_SIZE"
fi
echo ""

# Recent activity
echo "🔄 Recent Activity (Last 5 log entries):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -E "(Problem|Episode|Training|GRPO|complete)" /content/agentworkflow/training_full.log | tail -5
echo ""

echo "╚════════════════════════════════════════════════════════════╝"
