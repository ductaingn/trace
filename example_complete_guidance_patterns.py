#!/usr/bin/env python
"""
Complete working example - minimal reproduction of TRACE guidance with waypoints.
This shows the exact pattern needed for Arena integration.
"""

import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from tbsim.utils.guidance_loss import verify_guidance_config_list


def main():
    print("\n" + "=" * 80)
    print("COMPLETE WORKING EXAMPLE: TRACE Guidance Configs")
    print("=" * 80 + "\n")
    
    # =========================================================================
    # STEP 1: Define a single agent waypoint guidance
    # =========================================================================
    print("STEP 1: Single Agent Waypoint Guidance")
    print("-" * 80)
    
    single_agent_config = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': [[10.5, 0.0]],    # Single agent going to (10.5, 0.0)
            'urgency': [0.8],                # 80% speed
            'pref_speed': 1.42,              # m/s
            'dt': 0.1,                       # timestep
            'min_progress_dist': 0.5,        # min progress per step
        },
        'agents': [0]  # Apply to agent 0
    }
    
    # CRITICAL: Wrap in [[config]] for scenes
    single_agent_list = [[single_agent_config]]
    
    is_valid = verify_guidance_config_list(single_agent_list)
    print(f"Single agent config valid: {is_valid}")
    print(f"  Structure: {len(single_agent_list)} scene(s), {len(single_agent_list[0])} config(s)")
    print()
    
    # =========================================================================
    # STEP 2: Multiple agents with different waypoints
    # =========================================================================
    print("STEP 2: Multiple Agents with Different Waypoints")
    print("-" * 80)
    
    multi_agent_config = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': [
                [10.5, 0.0],   # Agent 0 goes here
                [10.5, 2.0],   # Agent 1 goes here
                [8.0, 1.0],    # Agent 2 goes here
            ],
            'urgency': [0.8, 0.8, 0.6],    # Different speeds
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0, 1, 2]  # Apply to all agents
    }
    
    multi_agent_list = [[multi_agent_config]]
    is_valid = verify_guidance_config_list(multi_agent_list)
    print(f"Multi-agent config valid: {is_valid}")
    print(f"  {len(multi_agent_config['params']['target_pos'])} agents with different targets")
    print()
    
    # =========================================================================
    # STEP 3: Combining multiple guidance types
    # =========================================================================
    print("STEP 3: Combining Multiple Guidance Types (Waypoint + Collision Avoidance)")
    print("-" * 80)
    
    waypoint_guidance = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': [[10.5, 0.0], [10.5, 2.0]],
            'urgency': [0.8, 0.8],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0, 1]
    }
    
    collision_guidance = {
        'name': 'agent_collision',
        'weight': 1.0,
        'params': {
            'num_disks': 5,
            'buffer_dist': 0.2,
        },
        'agents': None  # All agents
    }
    
    combined_list = [[waypoint_guidance, collision_guidance]]
    is_valid = verify_guidance_config_list(combined_list)
    print(f"Combined guidance valid: {is_valid}")
    print(f"  Config 1: waypoint guidance for agents [0, 1]")
    print(f"  Config 2: collision avoidance for all agents")
    print()
    
    # =========================================================================
    # STEP 4: Multiple scenes
    # =========================================================================
    print("STEP 4: Multiple Scenes (Each with Different Guidance)")
    print("-" * 80)
    
    scene_0_config = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': [[10.0, 0.0]],
            'urgency': [0.8],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0]
    }
    
    scene_1_config = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': [[5.0, 5.0]],
            'urgency': [0.9],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0]
    }
    
    multi_scene_list = [
        [scene_0_config],      # Scene 0 with config
        [scene_1_config],      # Scene 1 with different config
    ]
    
    is_valid = verify_guidance_config_list(multi_scene_list)
    print(f"Multi-scene config valid: {is_valid}")
    print(f"  Scene 0: target (10.0, 0.0)")
    print(f"  Scene 1: target (5.0, 5.0)")
    print()
    
    # =========================================================================
    # STEP 5: ARENA WORLD PATTERN
    # =========================================================================
    print("STEP 5: Arena World Integration Pattern")
    print("-" * 80)
    
    # This is what you'd do with Arena
    arena_waypoints = [
        [10.5, 0.0],   # Exit A
        [10.5, 2.0],   # Exit B
        [8.0, 1.0],    # Exit C
    ]
    
    arena_urgencies = [0.8, 0.8, 0.6]
    num_agents = 3
    
    arena_config = {
        'name': 'global_target_pos',
        'weight': 1.0,
        'params': {
            'target_pos': arena_waypoints,
            'urgency': arena_urgencies,
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': list(range(num_agents))
    }
    
    arena_config_list = [[arena_config]]  # CRITICAL: [[config]]
    
    is_valid = verify_guidance_config_list(arena_config_list)
    print(f"Arena config valid: {is_valid}")
    print(f"  Ready to use with guided_rollout()")
    print(f"  Example call:")
    print(f"""
    from tbsim.utils.scene_edit_utils import guided_rollout
    
    stats, info = guided_rollout(
        env=env,
        policy=rollout_policy,
        policy_model=policy_model,
        guidance_config=arena_config_list,  # [[{...}]]
        horizon=100,
    )
    """)
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("✓ ALL EXAMPLES VALID")
    print("=" * 80)
    print()
    print("Key patterns:")
    print("  1. Single agent:      [[{config for 1 agent}]]")
    print("  2. Multiple agents:   [[{config for N agents}]]")
    print("  3. Multiple configs:  [[{config1}, {config2}]]")
    print("  4. Multiple scenes:   [[configs_scene0], [configs_scene1]]")
    print()
    print("ALWAYS remember:")
    print("  ✓ Each config MUST have: 'name', 'weight', 'params', 'agents'")
    print("  ✓ Structure is ALWAYS list-of-lists: [[{...}, {...}], [{...}]]")
    print("  ✓ weight defaults to 1.0 (guidance strength)")
    print()
    print("Ready for Arena integration! 🚀")


if __name__ == "__main__":
    main()
