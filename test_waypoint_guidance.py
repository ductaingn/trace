"""
Test script to verify if TRACE model can follow synthetic waypoints using GlobalTargetPosGuidance.

Key questions to answer:
1. Does GlobalTargetPosGuidance need ground truth trajectories? NO - only needs target positions
2. Can we synthesize trajectories by providing sparse waypoints? YES
3. Does the number of inferred frames depend on ORCA dataset? Partially - depends on scene horizon config
"""
import matplotlib
matplotlib.use("Agg")
import csv
import datetime
import pickle

import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from arena_trace_pipeline_v2 import create_predefined_agents, arena_world_to_orca_map
from orca_sim.gen_dataset import USE_BOUNDARY, run_sim, viz_scene

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Import TRACE components
import importlib
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.guidance_loss import GuidanceConfig, verify_guidance_config_list
from tbsim.utils.scene_edit_utils import compute_heuristic_guidance, merge_guidance_configs
from arena_simulation_setup.tree.World import WorldDescription, World


def trace_output_to_socnavbench_scenario(trace_agents_trajectories, output_csv_path, target_fps=25, original_fps=2):
    """
    Convert trace output to SocNavBench format with interpolation to target FPS.
    
    Args:
        trace_agents_trajectories: trajectory data with 'pred_pos' key
        output_csv_path: path to save the CSV file
        target_fps: target frame rate (default 25 fps)
        original_fps: original frame rate (default 0.2 fps, i.e., 5 seconds per frame)
    """
    print("Collecting and interpolating trajectory points from TRACE output...")
    T, N, _ = trace_agents_trajectories.shape
    
    # Calculate interpolation factor
    interp_factor = int(target_fps / original_fps)  # 25 / 0.2 = 125 frames per original frame
    
    all_samples = []
    frame_offset = 200 # For ease of loading
    # frame_counter = 1
    
    for agent_id in range(N):
        frame_counter = 1
        for frame_idx in range(T - 1):
            # Get current and next position
            pos_current = trace_agents_trajectories[frame_idx, agent_id]
            pos_next = trace_agents_trajectories[frame_idx + 1, agent_id]
            
            # Interpolate between current and next position
            for interp_step in range(int(interp_factor)):
                alpha = interp_step / interp_factor
                x = pos_current[0] + alpha * (pos_next[0] - pos_current[0])
                y = pos_current[1] + alpha * (pos_next[1] - pos_current[1])
                
                all_samples.append({
                    'frame': frame_counter+frame_offset,
                    'id': agent_id,
                    'x': x,
                    'y': y
                })
                frame_counter += 1
        
        # Add the final position for this agent
        x, y = trace_agents_trajectories[T - 1, agent_id]
        all_samples.append({
            'frame': frame_counter + frame_offset,
            'id': agent_id,
            'x': x,
            'y': y
        })
    all_samples.sort(key=lambda s: (s['frame'], s['id']))
    # Prepare the four attribute rows
    row_frames = [s['frame'] for s in all_samples]
    row_ids = [s['id'] for s in all_samples]
    row_xs = [round(s['x'], 4) for s in all_samples]
    row_ys = [round(s['y'], 4) for s in all_samples]
    
    # Write to CSV
    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_frames)
        writer.writerow(row_ids)
        writer.writerow(row_xs)
        writer.writerow(row_ys)
    
    print(f"Successfully converted trajectories to {output_csv_path}")
    scenario_duration = max(row_frames) / target_fps
    print(f"Total frames: {max(row_frames)}, Total agents: {len(set(row_ids))}, Scenario duration: {scenario_duration:.2f}s")

def create_synthetic_guidance_config(
    num_agents: int,
    waypoints: List[Tuple[float, float]],
    urgency: float = 1.0,
    pref_speed: float = 1.42,
    dt: float = 0.1,
    min_progress_dist: float = 0.5,
    weight: float = 10000.0,
) -> Dict:
    """
    Create a synthetic guidance config using dummy waypoints.
    
    Args:
        num_agents: Number of agents in scene
        waypoint: Target position as (x, y)
        urgency: How much progress to make [0, 1]
        pref_speed: Preferred speed in m/s
        dt: Timestep in seconds
        min_progress_dist: Minimum distance to progress per step
        weight: Guidance weight (how much to affect denoising) - typically 1.0
    
    Returns:
        List of lists: guidance_config_list[scene_idx][config_idx]
    """
    print("=" * 80)
    print("CREATING SYNTHETIC GUIDANCE CONFIG")
    print("=" * 80)
    
    # Create a single guidance config applied to all agents
    # IMPORTANT: Each config must have 'name', 'weight', 'params', 'agents'
    guidance_config_global_target_pos = {
        'name': 'global_target_pos',
        'weight': weight,  # How much this guidance affects denoising
        'params': {
            'target_pos': waypoints,
            'urgency': [urgency] * num_agents,
            'pref_speed': pref_speed,
            'dt': dt,
            'min_progress_dist': min_progress_dist,
        },
        'agents': None  # Apply to all agents
    }

    guidance_config_global_obstacle_avoidance = {
        "name": "map_collision",
        "weight": 10.0,
        "params": {
            "num_points_lw": [
                10,
                10
            ]
        },
        "agents": None
    }
    
    # CRITICAL: Structure is list of lists: [scenes][configs_per_scene]
    # For 1 scene with 1 config: [[{config}]]
    guidance_config_list = [[guidance_config_global_target_pos, guidance_config_global_obstacle_avoidance]]
    
    print(f"✓ Created guidance config for {num_agents} agents")
    print(f"  Target waypoints: {waypoints}")
    print(f"  Urgency: {urgency}")
    print(f"  Weight: {weight}")
    print(f"  Pref speed: {pref_speed} m/s")
    print(f"  Structure: {len(guidance_config_list)} scene(s), {len(guidance_config_list[0])} config(s)")
    print(f"  Config: {list(guidance_config_list[0])}")
    
    return guidance_config_list

def apply_heuristic_guidance(
    env,
    manual_guidance_config: List[List[Dict]],
    scene_indices: List[int],
    start_frames: Optional[List[int]] = None,
) -> List[List[Dict]]:
    """
    Apply heuristic guidance on top of manual guidance using the same pattern as scene_editor.py.
    This is critical for making guidance actually work!
    
    Args:
        env: The environment
        manual_guidance_config: Manual guidance configs created by create_synthetic_guidance_config()
        scene_indices: Which scenes to apply heuristic to
        start_frames: Starting frame indices
    
    Returns:
        Merged guidance config (heuristic + manual)
    """
    print("=" * 80)
    print("APPLYING HEURISTIC GUIDANCE")
    print("=" * 80)
    
    # IMPORTANT: Reset env with scene_indices FIRST before computing heuristic
    # This ensures env._current_scenes is populated
    if start_frames is None:
        start_frames = [None] * len(scene_indices)
    
    print(f"Resetting environment with scenes {scene_indices}...")
    valid_scenes = env.reset(scene_indices=scene_indices, start_frame_index=start_frames)
    
    # Filter out invalid scenes
    scene_indices = [si for si, valid in zip(scene_indices, valid_scenes) if valid]
    start_frames = [sf for sf, valid in zip(start_frames, valid_scenes) if valid]
    
    if not scene_indices:
        print("⚠ Warning: No valid scenes after reset, returning manual guidance only")
        return manual_guidance_config
    
    # Heuristic config: use target_pos heuristic to generate configs automatically
    # This matches scene_editor.py pattern
    heuristic_config = [
        {
            'name': 'global_target_pos',
            'weight': 1.0,
            'params': {
                # Use the same target that was in manual config
                'target_time': 50,  # Predict 50 steps ahead
                'urgency': 1.0,
                'pref_speed': 1.42,
                'min_progress_dist': 0.5,
            }
        }
    ]
    
    # Compute heuristic guidance using scene data (like scene_editor.py does)
    print(f"Computing heuristic guidance for {len(scene_indices)} scene(s)...")
    heuristic_guidance_cfg = compute_heuristic_guidance(
        heuristic_config=heuristic_config,
        env=env,
        scene_indices=scene_indices,
        start_frames=start_frames,
    )
    
    print(f"✓ Heuristic guidance computed")
    print(f"  Number of configs per scene: {[len(cfg) for cfg in heuristic_guidance_cfg]}")
    
    # Merge heuristic with manual guidance (like scene_editor.py does)
    print(f"Merging heuristic and manual guidance configs...")
    merged_guidance_config = merge_guidance_configs(manual_guidance_config, heuristic_guidance_cfg)
    
    print(f"✓ Guidance configs merged")
    print(f"  Final number of configs per scene: {[len(cfg) for cfg in merged_guidance_config]}")
    
    return merged_guidance_config

def verify_guidance_config_validity(guidance_config_list: List[Dict]) -> bool:
    """
    Verify that guidance configs are valid before using them.
    """
    print("=" * 80)
    print("VERIFYING GUIDANCE CONFIG VALIDITY")
    print("=" * 80)
    
    try:
        # Check that config is valid
        valid = verify_guidance_config_list(guidance_config_list)
        print(f"✓ Guidance config is valid: {valid}")
        return valid
    except Exception as e:
        print(f"✗ Guidance config validation failed: {e}")
        return False

def load_orca_config(data_dir: str) -> SceneEditingConfig:
    """Load configuration for ORCA evaluation."""
    cfg = SceneEditingConfig()
    cfg.eval_class = "Diffuser"
    cfg.trajdata_source_test = ["orca_maps"]
    cfg.trajdata_data_dirs = {"orca_maps": data_dir}
    cfg.num_scenes_per_batch = 1
    cfg.num_simulation_steps = 200
    cfg.ckpt.policy.ckpt_dir = (
        "/home/linh/ductai_nguyen_ws/trace/ckpt/trace/orca_mixed"
    )
    cfg.ckpt.policy.ckpt_key = "iter40000"
    cfg.trajdata_cache_location = "~/.unified_data_cache"
    
    # Guidance settings
    cfg.policy.num_action_samples = 20
    cfg.policy.class_free_guide_w = 1.0  # Enable guidance
    cfg.policy.guide_as_filter_only = False
    cfg.policy.guide_clean = True

    print(f"✓ Config loaded")
    print(f"  num_simulation_steps: {cfg.num_simulation_steps}")
    print(f"  num_scenes_per_batch: {cfg.num_scenes_per_batch}")
    
    return cfg

def create_inference_env(eval_cfg: SceneEditingConfig, device: torch.device):
    """Create environment and policy for inference."""
    print("\n" + "=" * 80)
    print("CREATING INFERENCE ENVIRONMENT")
    print("=" * 80)
    
    # Set global trajdata environment
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    
    # Load policy and model
    print(f"Loading policy with eval_class: {eval_cfg.eval_class}")
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    policy_model = policy.model
    
    # Set up rasterization config
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # Create rollout wrapper
    rollout_policy = RolloutWrapper(agents_policy=policy)
    
    # Create environment
    print("Building environment with ORCA dataset...")
    env_builder = EnvUnifiedBuilder(
        eval_config=eval_cfg, exp_config=exp_config, device=device
    )
    env = env_builder.get_env()
    
    # Enable trajectory logging so we can extract positions for visualization
    env._log_data = True
    
    print(f"✓ Model loaded: {type(policy_model).__name__}")
    print(f"✓ Environment ready with {env.total_num_scenes} ORCA scenes")
    print(f"✓ Trajectory logging enabled for visualization")
    
    return env, rollout_policy, policy_model, exp_config

def run_inference_with_guidance(
    env,
    rollout_policy: RolloutWrapper,
    policy_model,
    guidance_config_list: List[Dict],
    scene_index: int = 0,
    horizon: int = 100,
    device=None,
) -> Dict:
    """
    Run inference on a single ORCA scene with synthetic guidance.
    
    Args:
        env: Simulation environment
        rollout_policy: Policy wrapper
        policy_model: The TRACE model
        guidance_config_list: List of guidance configs (one per scene)
        scene_index: Which scene to use
        horizon: Number of steps to predict
        device: PyTorch device
    
    Returns:
        Dictionary with trajectories and stats
    """
    from tbsim.utils.scene_edit_utils import guided_rollout
    
    print("\n" + "=" * 80)
    print(f"RUNNING INFERENCE ON SCENE {scene_index} WITH GUIDANCE")
    print("=" * 80)
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Use the proper guided_rollout function from scene_edit_utils
    print(f"Starting guided rollout on scene {scene_index}...")
    start_time = time.time()
    
    try:
        stats, info = guided_rollout(
            env=env,
            policy=rollout_policy,
            policy_model=policy_model,
            n_step_action=1,
            guidance_config=guidance_config_list,
            scene_indices=[scene_index],
            device=device,
            obs_to_torch=True,
            horizon=horizon,
            use_gt=False,
            start_frames=None,
        )
        
        elapsed_time = time.time() - start_time
        
        num_agents = env.current_num_agents
        
        print(f"✓ Rollout finished in {elapsed_time:.2f}s")
        print(f"  Scene index: {scene_index}")
        print(f"  Number of agents: {num_agents}")
        
        return {
            'stats': stats,
            'info': info,
            'elapsed_time': elapsed_time,
            'num_agents': num_agents,
            'guidance_config': guidance_config_list,
        }
    except Exception as e:
        print(f"✗ Guided rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_trajectories_from_buffer(info: Dict, num_agents: int) -> Optional[np.ndarray]:
    """
    Extract trajectory positions from the logged scene buffer.
    
    Args:
        info: Dictionary returned by guided_rollout() containing 'buffer'
        num_agents: Number of agents in the scene
    
    Returns:
        Array of shape (num_timesteps, num_agents, 2) or None if buffer not available
    """
    if 'buffer' not in info:
        print("⚠ Warning: No trajectory buffer found in info. Enable _log_data=True in environment")
        return None
    
    try:
        buffer_list = info['buffer']
        if not buffer_list or len(buffer_list) == 0:
            print("⚠ Warning: Buffer list is empty")
            return None
        
        # buffer is a list of scene buffers (we have 1 scene)
        scene_buffer = buffer_list[0]
        
        # scene_buffer is a dict with keys like 'centroid', 'yaw', 'extent', etc.
        if 'centroid' not in scene_buffer:
            print("⚠ Warning: 'centroid' not found in scene buffer")
            print(f"  Available keys: {list(scene_buffer.keys())}")
            return None
        
        # centroid shape is (num_agents, num_timesteps, 2)
        centroids = scene_buffer['centroid']
        
        # Transpose to (num_timesteps, num_agents, 2)
        if len(centroids.shape) == 3:
            trajectories = np.transpose(centroids, (1, 0, 2))
            print(f"✓ Extracted trajectories from buffer: shape {trajectories.shape}")
            return trajectories
        else:
            print(f"⚠ Warning: Unexpected centroid shape {centroids.shape}")
            return None
            
    except Exception as e:
        print(f"✗ Error extracting trajectories from buffer: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_results(results: Dict, output_dir: str = "./waypoint_guidance_results"):
    """
    Visualize the inference results with trajectory plots.
    Creates matplotlib visualizations showing:
    - Agent trajectories (colored paths)
    - Start positions (green circles)
    - Target waypoints (red stars)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch
    
    print("\n" + "=" * 80)
    print("VISUALIZING RESULTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if results is None:
        print("✗ No results to visualize")
        return
    
    try:
        info = results.get('info', {})
        num_agents = results['num_agents']
        # Extract trajectories from buffer
        pred_pos = extract_trajectories_from_buffer(info, num_agents)
        
        now = datetime.datetime.now().strftime("%H:%M:%S")
        with open(os.path.join(output_dir, f'trajectory_data_{now}.npz'), 'wb') as f:
            np.savez(f, pred_pos=pred_pos)
            trace_output_to_socnavbench_scenario(pred_pos, os.path.join(output_dir, f'trace_trajectory_{now}.csv'))

        print(f"✓ Trajectory data saved to {output_dir}/trajectory_data.npz")
        # Save results summary
        summary_json = {
            'elapsed_time': results['elapsed_time'],
            'num_agents': results['num_agents'],
            'guidance_target': results['guidance_config'][0][0]['params']['target_pos']
        }
        
        summary_file = os.path.join(output_dir, f'results_summary_{now}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        print(f"✓ Results summary saved to {summary_file}")
        
        # Extract trajectory data from results
        guidance_config = results['guidance_config'][0][0] if results['guidance_config'] else {}
        target_pos_raw = guidance_config.get('params', {}).get('target_pos', [[0, 0]])
        
        # Ensure target_pos is a proper [x, y] numpy array (not ragged)
        if isinstance(target_pos_raw, list) and len(target_pos_raw) > 0:
            target_pos = np.array(target_pos_raw[0] if isinstance(target_pos_raw[0], (list, tuple)) else target_pos_raw, dtype=np.float64)
        else:
            target_pos = np.array(target_pos_raw, dtype=np.float64).flatten()[:2]
        
        # Ensure it's 1D with shape (2,)
        if target_pos.ndim > 1:
            target_pos = target_pos.flatten()
        if len(target_pos) < 2:
            target_pos = np.array([0.0, 0.0])
        else:
            target_pos = target_pos[:2]
        
        if pred_pos is None:
            print("✗ Could not extract trajectory data for visualization")
            return
        
        # Create trajectory visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each agent's trajectory
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_agents, 10)))
        
        for agent_id in range(num_agents):
            if agent_id < pred_pos.shape[1]:
                traj = pred_pos[:, agent_id, :]
                
                # Plot trajectory line
                ax.plot(traj[:, 0], traj[:, 1], 
                       color=colors[agent_id], 
                       linewidth=2.5, 
                       alpha=0.7,
                       label=f'Agent {agent_id}')
                
                # Plot start point (green circle)
                ax.plot(traj[0, 0], traj[0, 1], 
                       'go', markersize=12, 
                       markeredgecolor='darkgreen', 
                       markeredgewidth=2,
                       zorder=5)
                
                # Plot end point (blue square)
                ax.plot(traj[-1, 0], traj[-1, 1], 
                       's', color=colors[agent_id], 
                       markersize=10,
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       zorder=5)
        
        # Plot target waypoint (red star) - same for all agents
        ax.plot(target_pos[0], target_pos[1], 
               'r*', markersize=25, 
               markeredgecolor='darkred', 
               markeredgewidth=2,
               label='Target Waypoint',
               zorder=10)
        
        # Add arrow from start to target to show guidance direction
        if num_agents > 0:
            start_pos = np.array(pred_pos[0, 0, :], dtype=np.float64)
            target_pos_arrow = np.array(target_pos, dtype=np.float64)
            arrow = FancyArrowPatch(
                tuple(start_pos), 
                tuple(target_pos_arrow),
                arrowstyle='->', 
                mutation_scale=30, 
                color='red',
                alpha=0.3,
                linewidth=2,
                zorder=2
            )
            ax.add_patch(arrow)
        
        # Customize plot
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'TRACE Waypoint Guidance: {num_agents} Agents Following Target [{target_pos[0]:.1f}, {target_pos[1]:.1f}]', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        
        # Add text annotations
        textstr = f'Elapsed: {results["elapsed_time"]:.1f}s\nAgents: {num_agents}\nSteps: {pred_pos.shape[0]}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save trajectory plot
        traj_plot_file = os.path.join(output_dir, 'agent_trajectories.png')
        plt.tight_layout()
        plt.savefig(traj_plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Trajectory visualization saved to {traj_plot_file}")
        plt.close()
        
        # Create animation/step-by-step visualization
        num_steps = pred_pos.shape[0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_agents, 10)))
        
        # Create frames for animation
        for step in range(0, num_steps, max(1, num_steps // 10)):  # 10 frames
            ax.clear()
            
            # Plot trajectories up to this step
            for agent_id in range(num_agents):
                if agent_id < pred_pos.shape[1]:
                    traj_up_to_step = pred_pos[:step+1, agent_id, :]
                    
                    # Plot trajectory line
                    if len(traj_up_to_step) > 1:
                        ax.plot(traj_up_to_step[:, 0], traj_up_to_step[:, 1],
                               color=colors[agent_id], linewidth=2.5, alpha=0.7)
                    
                    # Plot current position (larger circle)
                    ax.plot(traj_up_to_step[-1, 0], traj_up_to_step[-1, 1],
                           'o', color=colors[agent_id], markersize=10,
                           markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                    
                    # Plot start position (green)
                    ax.plot(traj_up_to_step[0, 0], traj_up_to_step[0, 1],
                           'go', markersize=12, markeredgecolor='darkgreen',
                           markeredgewidth=2, zorder=5)
            
            # Plot target waypoint
            ax.plot(target_pos[0], target_pos[1],
                   'r*', markersize=25, markeredgecolor='darkred',
                   markeredgewidth=2, zorder=10)
            
            # Customize plot
            ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
            ax.set_title(f'TRACE Waypoint Guidance - Step {step}/{num_steps}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # Get axis limits from full trajectory
            all_positions = np.vstack([pred_pos[:, i, :] for i in range(min(num_agents, pred_pos.shape[1]))])
            margin = 2.0
            ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
            ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
            
            # Save frame
            frame_file = os.path.join(output_dir, f'trajectory_frame_{step:03d}.png')
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        
        plt.close()
        print(f"✓ Trajectory animation frames saved to {output_dir}/")
        print(f"  Created {(num_steps + 9) // 10} frames for visualization")
        
        print(f"\n✓ All visualizations complete!")
        print(f"  Elapsed time: {summary_json['elapsed_time']:.2f}s")
        print(f"  Number of agents: {summary_json['num_agents']}")
        print(f"  Guidance target: {summary_json['guidance_target']}")
        print(f"  Output directory: {output_dir}")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

def main(scenario_name: str = "scenario_queuing"):
    """Main test script."""
    # Check if cache directory exists and remove it
    cache_dir = "/home/linh/.unified_data_cache"
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ Removed cache directory: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")

    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "TRACE Model: Synthetic Waypoint Guidance Test".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Step 3: Create synthetic waypoint config
    with open(f"./datasets/orca_arena/maps/scene_000000/{scenario_name}.json", "r") as f:
        scenario = json.load(f)
        spawn_pos = []
        waypoints = []
        for agent in scenario["agents"]:
            spawn_pos.append([agent["pos"][0], agent["pos"][1]])
            waypoints.append(agent["waypoint"])

    agent_dict = create_predefined_agents(
        num_agents=len(spawn_pos),
        initial_positions=np.array(spawn_pos)
    )
    world_path = "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"
    world = World(path=Path(world_path))
    orca_map = arena_world_to_orca_map(world)

    sim_data = run_sim(agent_dict, orca_map, 100, 10, False)
    np.savez(os.path.join("/home/linh/ductai_nguyen_ws/trace/datasets/orca_arena/maps/scene_000000", 'sim.npz'), **sim_data)
    with open(os.path.join("/home/linh/ductai_nguyen_ws/trace/datasets/orca_arena/maps/scene_000000", 'map.pkl'), 'wb') as mapf:
        pickle.dump(orca_map, mapf)

    guidance_config_list = create_synthetic_guidance_config(
        num_agents=len(spawn_pos),  # Will be set to actual scene size
        waypoints=waypoints,
        urgency=1.0,
        pref_speed=1.42,
        dt=0.1,
        min_progress_dist=0.5,
    )
    
    # Step 4: Verify config validity
    is_valid = verify_guidance_config_validity(guidance_config_list)
    if not is_valid:
        print("✗ Guidance config validation failed!")
        return
    
    # Step 5: Run inference
    print("\n" + "=" * 80)
    print("INFERENCE SETUP")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    eval_cfg = load_orca_config("./datasets/orca_arena")

    try:
        env, rollout_policy, policy_model, exp_config = create_inference_env(eval_cfg, device)
        
        # IMPORTANT: Apply heuristic guidance like scene_editor.py does
        # This is critical for making guidance actually work!
        print("\n" + "=" * 80)
        print("APPLYING HEURISTIC GUIDANCE")
        print("=" * 80)
        guidance_config_list_with_heuristic = apply_heuristic_guidance(
            env=env,
            manual_guidance_config=guidance_config_list,
            scene_indices=[0],
            start_frames=None,
        )
        
        # Run inference on first scene with proper device parameter
        results = run_inference_with_guidance(
            env,
            rollout_policy,
            policy_model,
            guidance_config_list_with_heuristic,
            scene_index=0,
            horizon=eval_cfg.num_simulation_steps,
            device=device,
        )
        
        # Visualize results
        visualize_results(results)
        
    except Exception as e:
        print(f"\n✗ Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    for scenario_name in ["scenario_emergency"]:
        main(scenario_name)
