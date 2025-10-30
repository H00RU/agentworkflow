#!/bin/bash
# 训练监控脚本

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        AgentWorkflow Training Monitor (修复版)            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查训练进程
if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo "✓ 训练进程: 运行中 (PID: $PID, 运行时长: $RUNTIME)"
    else
        echo "✗ 训练进程: 已停止 (PID: $PID)"
    fi
else
    echo "⚠ 未找到训练进程ID文件"
fi
echo ""

# 显示最新进度
echo "📊 训练进度 (最近10行):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -10 training.log | grep -E "(Epoch|Episode|Problem|Evaluation|Pass@k|Training GRPO)" || echo "等待训练日志..."
echo ""

# 统计已完成
if [ -f "training.log" ]; then
    PROBLEMS_DONE=$(grep -c "Evaluation complete" training.log 2>/dev/null || echo "0")
    echo "  已完成问题数: $PROBLEMS_DONE"
fi
echo ""

# 检查输出文件
echo "📁 生成的输出:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "outputs" ]; then
    RESULTS=$(find outputs/results -name "*.json" 2>/dev/null | wc -l)
    OUTPUTS_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
    echo "  结果文件: $RESULTS"
    echo "  输出目录大小: $OUTPUTS_SIZE"
fi

if [ -d "checkpoints" ]; then
    CHECKPOINTS=$(find checkpoints -type f 2>/dev/null | wc -l)
    echo "  检查点文件: $CHECKPOINTS"
fi
echo ""

# 检查错误
echo "⚠️  近期错误 (如有):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -i "error\|exception\|failed" training.log 2>/dev/null | tail -3 | grep -v "Unable to register" || echo "无错误"
echo ""

echo "╚════════════════════════════════════════════════════════════╝"
echo "提示: 查看完整日志: tail -f training.log"
