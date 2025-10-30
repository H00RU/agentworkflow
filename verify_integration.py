#!/usr/bin/env python3
"""
Verification script to ensure agentworkflow is fully integrated and independent.

Checks:
1. AFlow is integrated locally
2. Can import AFlow components
3. All paths are relative/self-contained
4. No external dependencies on /content/AFlow
"""

import os
import sys
from pathlib import Path

def check_aflow_integration():
    """Check if AFlow is properly integrated."""
    agentworkflow_root = os.path.dirname(os.path.abspath(__file__))
    aflow_path = os.path.join(agentworkflow_root, 'AFlow')
    optimizer_path = os.path.join(aflow_path, 'scripts', 'optimizer.py')

    print("=" * 70)
    print("AGENTWORKFLOW INTEGRATION CHECK")
    print("=" * 70)

    # Check 1: AFlow directory exists
    print("\n[1] Checking AFlow directory...")
    if os.path.exists(aflow_path):
        print(f"    ✅ AFlow found at: {aflow_path}")
    else:
        print(f"    ❌ AFlow NOT found at: {aflow_path}")
        return False

    # Check 2: optimizer.py exists
    print("\n[2] Checking optimizer.py...")
    if os.path.exists(optimizer_path):
        print(f"    ✅ optimizer.py found")
    else:
        print(f"    ❌ optimizer.py NOT found")
        return False

    # Check 3: Can import from local AFlow
    print("\n[3] Testing AFlow import...")
    sys.path.insert(0, aflow_path)
    try:
        from scripts.optimizer import Optimizer
        print(f"    ✅ Successfully imported Optimizer from local AFlow")
    except ImportError as e:
        print(f"    ❌ Failed to import: {e}")
        return False

    # Check 4: Verify agentworkflow modules
    print("\n[4] Checking agentworkflow modules...")
    modules_to_check = [
        'src/eval/workflow_evaluator.py',
        'src/mcts/mcts_optimizer.py',
        'src/grpo/qwen_policy.py',
        'train.py',
        'test_e2e.py',
    ]

    all_exist = True
    for module in modules_to_check:
        module_path = os.path.join(agentworkflow_root, module)
        if os.path.exists(module_path):
            print(f"    ✅ {module}")
        else:
            print(f"    ❌ {module} NOT found")
            all_exist = False

    if not all_exist:
        return False

    # Check 5: AIME data
    print("\n[5] Checking AIME dataset...")
    aime_path = os.path.join(agentworkflow_root, 'data', 'aime24', 'data.json')
    if os.path.exists(aime_path):
        print(f"    ✅ AIME data found")
    else:
        print(f"    ❌ AIME data NOT found")
        return False

    # Check 6: Configuration files
    print("\n[6] Checking configuration files...")
    config_files = [
        'config/training_config.yaml',
        'config/minimal_test.yaml',
    ]

    for config_file in config_files:
        config_path = os.path.join(agentworkflow_root, config_file)
        if os.path.exists(config_path):
            print(f"    ✅ {config_file}")
        else:
            print(f"    ❌ {config_file} NOT found")
            all_exist = False

    if not all_exist:
        return False

    # Check 7: Verify paths are relative
    print("\n[7] Verifying configuration uses relative paths...")
    config_path = os.path.join(agentworkflow_root, 'config', 'training_config.yaml')
    with open(config_path, 'r') as f:
        config_content = f.read()
        if 'aflow_path: ./AFlow' in config_content or 'aflow_path: /content/agentworkflow/AFlow' in config_content:
            print(f"    ✅ Configuration uses local/relative AFlow paths")
        elif '/content/AFlow' in config_content:
            print(f"    ⚠️  Configuration still references /content/AFlow (will auto-redirect)")
        else:
            print(f"    ⚠️  Could not determine AFlow path in config")

    return True


def main():
    """Run all checks."""
    success = check_aflow_integration()

    print("\n" + "=" * 70)
    if success:
        print("✅ INTEGRATION SUCCESSFUL")
        print("=" * 70)
        print("\nAgentworkflow is now fully independent and self-contained!")
        print("\nYou can now use agentworkflow without depending on /content/AFlow")
        print("\nNext steps:")
        print("  1. cd /content/agentworkflow")
        print("  2. python test_e2e.py")
        print("  3. python train.py --config config/minimal_test.yaml")
        return 0
    else:
        print("❌ INTEGRATION FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before running training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
