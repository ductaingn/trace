"""
QUICK START: Copy-Paste Code Patterns

This file contains ready-to-use code patterns for integrating
TRACE with synthetic waypoint guidance into your Arena pipeline.
"""

# ============================================================================
# PATTERN 1: Creating Valid Guidance Config
# ============================================================================

def create_waypoint_guidance_config(
    num_agents: int,
    waypoints: list,  # List of (x, y) tuples, one per agent
    urgency_values: list = None,  # List of 0-1 values, or None for default
    pref_speed: float = 1.42,
    dt: float = 0.1,
    min_progress_dist: float = 0.5,
    weight: float = 1.0,  # Guidance weight - how much to affect denoising
) -> list:
    """
    Create a VALID guidance config for TRACE model.
    
    CRITICAL: Must return [[{config}]] structure (list of lists)
    - Outer list: scenes
    - Inner list: configs for that scene
    - Dict: config with 'name', 'weight', 'params', 'agents' keys
    """
    
    if urgency_values is None:
        urgency_values = [0.8] * num_agents
    
    # Single guidance config for this scene
    guidance_config = {
        'name': 'global_target_pos',  # Must be in GUIDANCE_FUNC_MAP
        'weight': weight,  # REQUIRED: was missing in original examples!
        'params': {
            'target_pos': waypoints,           # (N, 2) list of positions
            'urgency': urgency_values,         # (N,) list of 0-1 values
            'pref_speed': pref_speed,          # Speed in m/s
            'dt': dt,                          # Match model's timestep!
            'min_progress_dist': min_progress_dist,  # Min progress per step
        },
        'agents': list(range(num_agents))  # Apply to all agents
    }
    
    # CRITICAL: Return [[config]] not [config]
    # DiffuserGuidance expects guidance_config_list[scene_idx][config_idx]
    return [[guidance_config]]


# ============================================================================
# PATTERN 2: Validating Guidance Config
# ============================================================================

def validate_guidance_config(guidance_config_list: list) -> bool:
    """Verify config is valid before using it."""
    from tbsim.utils.guidance_loss import verify_guidance_config_list
    
    try:
        is_valid = verify_guidance_config_list(guidance_config_list)
        if is_valid:
            print("✓ Guidance config is valid")
            return True
        else:
            print("✗ Guidance config validation returned False")
            return False
    except Exception as e:
        print(f"✗ Config validation error: {e}")
        return False


# ============================================================================
# PATTERN 3: Loading TRACE Model and Environment
# ============================================================================

def setup_trace_inference(
    ckpt_dir: str = "./ckpt/trace/orca_mixed",
    ckpt_key: str = "iter40000",
    data_dir: str = "./datasets/orca_sim",
    num_simulation_steps: int = 100,
    num_action_samples: int = 20,
) -> tuple:
    """
    Setup TRACE model, policy, and environment.
    
    Returns:
        (env, rollout_policy, policy_model, exp_config)
    """
    import torch
    import importlib
    from tbsim.configs.scene_edit_config import SceneEditingConfig
    from tbsim.evaluation.env_builders import EnvUnifiedBuilder
    from tbsim.policies.wrappers import RolloutWrapper
    from tbsim.utils.trajdata_utils import (
        set_global_trajdata_batch_env,
        set_global_trajdata_batch_raster_cfg,
    )
    
    # Create config
    cfg = SceneEditingConfig()
    cfg.eval_class = "Diffuser"
    cfg.trajdata_source_test = ["orca_maps"]
    cfg.trajdata_data_dirs = {"orca_maps": data_dir}
    cfg.num_simulation_steps = num_simulation_steps
    cfg.ckpt.policy.ckpt_dir = ckpt_dir
    cfg.ckpt.policy.ckpt_key = ckpt_key
    
    # Guidance settings
    cfg.policy.num_action_samples = num_action_samples
    cfg.policy.class_free_guide_w = 1.0  # Enable guidance
    cfg.policy.guide_as_filter_only = False
    cfg.policy.guide_clean = True
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Setup trajdata
    set_global_trajdata_batch_env(cfg.trajdata_source_test[0])
    
    # Load policy
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, cfg.eval_class)
    composer = composer_class(cfg, device)
    policy, exp_config = composer.get_policy()
    policy_model = policy.model
    
    # Setup rasterization
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # Create rollout wrapper
    rollout_policy = RolloutWrapper(agents_policy=policy)
    
    # Create environment
    env_builder = EnvUnifiedBuilder(
        eval_config=cfg, exp_config=exp_config, device=device
    )
    env = env_builder.get_env()
    
    return env, rollout_policy, policy_model, exp_config


# ============================================================================
# PATTERN 4: Running Inference with Guidance
# ============================================================================

def run_inference_with_waypoints(
    env,
    rollout_policy,
    policy_model,
    guidance_config_list: list,
    scene_index: int = 0,
    num_simulation_steps: int = 100,
) -> dict:
    """
    Run TRACE inference with synthetic waypoint guidance.
    
    Returns:
        Dictionary with results including trajectories
    """
    from tbsim.utils.scene_edit_utils import guided_rollout
    import torch
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Reset environment to scene
    scenes_valid = env.reset(
        scene_indices=[scene_index],
        start_frame_index=None,
    )
    
    if not scenes_valid[0]:
        print(f"✗ Scene {scene_index} is not valid")
        return None
    
    num_agents = env.current_num_agents
    print(f"✓ Scene loaded with {num_agents} agents")
    
    # Run inference with guidance
    stats, info = guided_rollout(
        env=env,
        policy=rollout_policy,
        policy_model=policy_model,
        n_step_action=1,
        guidance_config=guidance_config_list,
        scene_indices=[scene_index],
        device=device,
        obs_to_torch=True,
        horizon=num_simulation_steps,
        use_gt=False,
        start_frames=None,
    )
    
    return {
        'stats': stats,
        'info': info,
        'num_agents': num_agents,
    }


# ============================================================================
# PATTERN 5: Visualizing Results with draw_scene_data
# ============================================================================

def visualize_inference_results(
    results: dict,
    guidance_config_list: list,
    rasterizer,
    output_path: str = "trace_results.png",
) -> None:
    """
    Visualize TRACE inference results using draw_scene_data.
    """
    import matplotlib.pyplot as plt
    from tbsim.utils.viz_utils import draw_scene_data
    
    if results is None or 'info' not in results:
        print("✗ No results to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw scene data
    draw_scene_data(
        ax=ax,
        scene_name="trace_inference",
        scene_data=results['info'],
        starting_frame=0,
        rasterizer=rasterizer,
        guidance_config=guidance_config_list,  # Show guidance
        draw_trajectory=True,      # Show agent paths
        draw_action=True,          # Show predicted actions
        draw_diffusion_step=None,
        n_step_action=5,
        draw_action_sample=False,  # Cleaner visualization
        traj_len=200,
        ras_pos=None,
        linewidth=2.5,
    )
    
    ax.set_title(f"TRACE Inference - {results['num_agents']} Agents")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


# ============================================================================
# PATTERN 6: Complete Integration Example
# ============================================================================

def complete_example():
    """
    Complete example showing all steps together.
    """
    import os
    import numpy as np
    
    # Step 1: Setup TRACE
    print("Step 1: Setting up TRACE...")
    env, rollout_policy, policy_model, exp_config = setup_trace_inference(
        num_simulation_steps=100,
        num_action_samples=20,
    )
    
    # Step 2: Define waypoints
    print("\nStep 2: Creating waypoint guidance...")
    num_agents = 3  # From your Arena scenario
    waypoints = [
        (10.5, 0.0),   # Agent 0 target
        (10.5, 2.0),   # Agent 1 target
        (8.0, 1.0),    # Agent 2 target
    ]
    urgencies = [0.8, 0.8, 0.6]  # Different speeds
    
    guidance_config_list = create_waypoint_guidance_config(
        num_agents=num_agents,
        waypoints=waypoints,
        urgency_values=urgencies,
    )
    
    # Step 3: Validate config
    print("\nStep 3: Validating config...")
    if not validate_guidance_config(guidance_config_list):
        print("✗ Config validation failed!")
        return
    
    # Step 4: Run inference
    print("\nStep 4: Running inference with guidance...")
    results = run_inference_with_waypoints(
        env=env,
        rollout_policy=rollout_policy,
        policy_model=policy_model,
        guidance_config_list=guidance_config_list,
        scene_index=0,
        num_simulation_steps=100,
    )
    
    if results is None:
        print("✗ Inference failed!")
        return
    
    # Step 5: Visualize
    print("\nStep 5: Visualizing results...")
    try:
        from tbsim.utils.viz_utils import get_trajdata_renderer
        
        rasterizer = get_trajdata_renderer(
            desired_data=["orca_maps"],
            data_dirs={"orca_maps": "./datasets/orca_sim"},
            raster_size=400,
            px_per_m=12.0,
            rebuild_maps=False,
            cache_location="~/.unified_data_cache",
        )
        
        os.makedirs("results", exist_ok=True)
        visualize_inference_results(
            results=results,
            guidance_config_list=guidance_config_list,
            rasterizer=rasterizer,
            output_path="results/trace_visualization.png",
        )
    except Exception as e:
        print(f"Note: Visualization skipped ({e})")
    
    print("\n✓ Example complete!")
    return results


# ============================================================================
# PATTERN 7: Arena World Integration
# ============================================================================

def arena_world_integration_example():
    """
    Example showing how to integrate with Arena world.
    
    This shows the pattern you'd use in arena_trace_pipeline_v2.py
    """
    import numpy as np
    from pathlib import Path
    
    # Step 1: Load Arena world (from arena_trace_pipeline_v2.py)
    print("Loading Arena world...")
    # from arena_simulation_setup.tree.World import World
    # world = World(path=Path(arena_world_path))
    # world_descr = world.load()
    
    # Step 2: Extract initial agent positions
    # initial_positions = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    # num_agents = initial_positions.shape[0]
    
    # For this example, use dummy data
    num_agents = 3
    initial_positions = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [0.0, 4.0],
    ])
    
    # Step 3: Define Arena waypoints
    print(f"Creating guidance for {num_agents} agents in Arena world...")
    waypoints = [
        (10.5, 0.0),   # Exit A
        (10.5, 2.0),   # Exit B
        (8.0, 1.0),    # Exit C (different)
    ]
    
    guidance_config_list = create_waypoint_guidance_config(
        num_agents=num_agents,
        waypoints=waypoints,
        urgency_values=[0.8] * num_agents,
    )
    
    # Step 4: Validate
    if not validate_guidance_config(guidance_config_list):
        print("Config validation failed!")
        return
    
    # Step 5: Setup TRACE and run
    print("Setting up TRACE model...")
    env, rollout_policy, policy_model, exp_config = setup_trace_inference()
    
    print("Running inference on Arena scenario...")
    # Note: In real usage, you'd need to create an environment with Arena data
    # For now, we're using ORCA data as example
    results = run_inference_with_waypoints(
        env=env,
        rollout_policy=rollout_policy,
        policy_model=policy_model,
        guidance_config_list=guidance_config_list,
        scene_index=0,
    )
    
    if results:
        print("✓ Inference successful!")
        print(f"  Generated trajectories for {results['num_agents']} agents")
        print(f"  Trajectories shape: {results['info']['centroid'].shape if results['info'] else 'unknown'}")
    else:
        print("✗ Inference failed")


# ============================================================================
# QUICK REFERENCE: All Guidance Types
# ============================================================================

GUIDANCE_EXAMPLES = {
    'global_target_pos': {
        'name': 'global_target_pos',
        'weight': 1.0,  # REQUIRED!
        'params': {
            'target_pos': [[10.5, 0.0]],
            'urgency': [0.8],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0],
    },
    
    'target_speed': {
        'name': 'target_speed',
        'weight': 1.0,  # REQUIRED!
        'params': {
            'target_speed': 1.5,  # m/s
            'dt': 0.1,
            'mode': 'use_action',
        },
        'agents': None,  # All agents
    },
    
    'agent_collision': {
        'name': 'agent_collision',
        'weight': 1.0,  # REQUIRED!
        'params': {
            'num_disks': 5,
            'buffer_dist': 0.2,
        },
        'agents': None,  # All agents
    },
    
    'map_collision': {
        'name': 'map_collision',
        'weight': 1.0,  # REQUIRED!
        'params': {
            'num_points_lw': (10, 10),
        },
        'agents': None,  # All agents
    },
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\nQuick Start Code Patterns for TRACE + Waypoint Guidance\n")
    
    # Uncomment to run examples:
    # complete_example()
    # arena_world_integration_example()
    
    print("Available functions:")
    print("  • create_waypoint_guidance_config()")
    print("  • validate_guidance_config()")
    print("  • setup_trace_inference()")
    print("  • run_inference_with_waypoints()")
    print("  • visualize_inference_results()")
    print("  • complete_example()")
    print("  • arena_world_integration_example()")
    print("\nUse these patterns in your code!")
