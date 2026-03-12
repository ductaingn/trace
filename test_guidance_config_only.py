#!/usr/bin/env python
"""
Quick validation test for TRACE guidance configs.
This tests just the configuration validity without running full inference.
"""

import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from tbsim.utils.guidance_loss import GuidanceConfig, verify_guidance_config_list


def test_guidance_config_structure():
    """Test that guidance configs have the correct structure."""
    print("=" * 80)
    print("TEST: Guidance Config Structure")
    print("=" * 80)
    print()
    
    # Create a valid guidance config
    guidance_config = {
        'name': 'global_target_pos',
        'weight': 1.0,  # CRITICAL: Must have weight key
        'params': {
            'target_pos': [[10.5, 0.0]],
            'urgency': [0.8],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0]
    }
    
    # Test 1: Check GuidanceConfig.from_dict() works
    print("Test 1: Creating GuidanceConfig from dict...")
    try:
        config_obj = GuidanceConfig.from_dict(guidance_config)
        print(f"✓ GuidanceConfig created successfully")
        print(f"  Config name: {config_obj.name}")
        print(f"  Config weight: {config_obj.weight}")
        print(f"  Config agents: {config_obj.agents}")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 2: Check verify_guidance_config_list() with correct structure
    print("Test 2: Verifying config list with correct structure [[config]]...")
    guidance_config_list = [[guidance_config]]  # CRITICAL: List of lists
    try:
        is_valid = verify_guidance_config_list(guidance_config_list)
        print(f"✓ Config list is valid: {is_valid}")
        print(f"  Structure: {len(guidance_config_list)} scene(s)")
        print(f"           : {len(guidance_config_list[0])} config(s) per scene")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 3: Verify all required keys are present
    print("Test 3: Checking all required keys...")
    required_keys = {'name', 'weight', 'params', 'agents'}
    actual_keys = set(guidance_config.keys())
    
    if actual_keys == required_keys:
        print(f"✓ All required keys present: {required_keys}")
    else:
        print(f"✗ Key mismatch!")
        print(f"  Required: {required_keys}")
        print(f"  Actual: {actual_keys}")
        print(f"  Missing: {required_keys - actual_keys}")
        print(f"  Extra: {actual_keys - required_keys}")
        return False
    
    print()
    
    # Test 4: Multiple configs per scene
    print("Test 4: Multiple guidance configs for same scene...")
    guidance_config_2 = {
        'name': 'agent_collision',
        'weight': 1.0,
        'params': {
            'num_disks': 5,
            'buffer_dist': 0.2,
        },
        'agents': None  # All agents
    }
    
    multi_config_list = [[guidance_config, guidance_config_2]]  # One scene, two configs
    try:
        is_valid = verify_guidance_config_list(multi_config_list)
        print(f"✓ Multi-config list is valid: {is_valid}")
        print(f"  Scene 0: {len(multi_config_list[0])} configs")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print()
    print("=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  1. ✓ GuidanceConfig.from_dict() works with all 4 required keys")
    print("  2. ✓ verify_guidance_config_list() validates [[config]] structure")
    print("  3. ✓ Config has all required keys: name, weight, params, agents")
    print("  4. ✓ Multiple configs per scene work correctly")
    print()
    print("Guidance configs are now VALID for TRACE inference!")
    
    return True


if __name__ == "__main__":
    success = test_guidance_config_structure()
    sys.exit(0 if success else 1)
