"""
ORCA Map Inference Pipeline with Arena World Support
=====================================================
Convert an Arena simulation world to ORCA map format on-the-fly,
use predefined agents, and run inference without saving intermediate files.

Usage:
    python orca_inference_example.py \\
        --config configs/eval/orca/eval.json \\
        --arena-world /path/to/arena/world \\
        --num-agents 5 \\
        --horizon 50 \\
        --output-dir ./results

Optional: Provide specific initial positions:
    python orca_inference_example.py \\
        --arena-world /path/to/arena/world \\
        --num-agents 5 \\
        --agent-positions agent_positions.json \\
        --horizon 50

Agent positions JSON format (agent_positions.json):
    [
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ]
"""

import argparse
import json
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import yaml
import asyncio

# Arena imports
from arena_simulation_setup.tree.World import World

# tbsim imports
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.tensor_utils import map_ndarray
import tbsim.evaluation.policy_composers as policy_composers
import tbsim.utils.tensor_utils as TensorUtils


# ==============================================================================
# ARENA WORLD CONVERSION FUNCTIONS
# ==============================================================================

def load_arena_world(world_path: str) -> "World":
    """
    Load an Arena simulation world.
    
    Args:
        world_path: Path to Arena world directory
        
    Returns:
        World object
    """
    print(f"Loading Arena world from: {world_path}")
    world = World(path=Path(world_path))
    return world


def arena_world_to_orca_map(
    world: World, wall_thickness: float = 0.2
) -> List[List[Tuple[float, float]]]:
    """
    Convert Arena World obstacles and walls to ORCA map format.
    
    Args:
        world: Arena World object
        wall_thickness: Thickness of walls in the conversion
        
    Returns:
        ORCA map format: List of polygons (each polygon is list of (x, y) tuples)
    """
    print("Converting Arena world to ORCA map format...")
    orca_map = []
    world_descr = world.load()
    
    obstacle_count = 0
    wall_count = 0
    
    for obstacle in world_descr.all_static_entities:
        model = obstacle.model
        annotation_path = asyncio.run(model.resolve_path()) / "annotation.yaml"

        with open(annotation_path, "r") as f:
            annotation: dict = yaml.safe_load(f)

        # Bounding box in model-local frame
        (min_x, max_x), (min_y, max_y), _ = annotation["bounding_box"]

        # 2D corners in CCW order (local frame)
        local_corners = np.array(
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ]
        )

        # Rotation (yaw only, ORCA is 2D)
        yaw = obstacle.pose.orientation.to_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])

        # Rotate around model origin
        world_corners = local_corners @ R.T

        # Translate to world position
        world_corners += np.array(
            [obstacle.pose.position.x, obstacle.pose.position.y]
        )

        # Append as CCW polygon
        orca_map.append([(float(x), float(y)) for x, y in world_corners])
        obstacle_count += 1

    # Convert walls
    for wall in world_descr.all_walls:
        try:
            start = np.array([wall.start.x, wall.start.y])
            end = np.array([wall.end.x, wall.end.y])

            d = end - start
            length = np.linalg.norm(d)

            if length < 1e-6:
                continue  # skip degenerate walls

            # Unit direction
            t = d / length

            # Perpendicular (normal)
            n = np.array([-t[1], t[0]])

            half_w = wall_thickness / 2

            # Rectangle corners (CCW)
            corners = [
                start + n * half_w,
                end + n * half_w,
                end - n * half_w,
                start - n * half_w,
            ]

            orca_map.append([(float(x), float(y)) for x, y in corners])
            wall_count += 1
        except Exception as e:
            print(f"Warning: Could not convert wall: {e}")
            continue

    print(f"✓ Converted {obstacle_count} obstacles and {wall_count} walls to ORCA map")
    return orca_map


def create_predefined_agents(
    num_agents: int,
    initial_positions: Optional[np.ndarray] = None,
    initial_velocities: Optional[np.ndarray] = None,
    agent_radius: float = 0.4,
    max_speed: float = 2.0,
    preferred_velocity_range: Tuple[float, float] = (1.0, 2.0),
) -> Dict:
    """
    Create predefined agent properties for ORCA simulation.
    
    Args:
        num_agents: Number of agents to create
        initial_positions: (num_agents, 2) array of x, y positions. If None, randomly sampled.
        initial_velocities: (num_agents, 2) array of velocities. If None, zero.
        agent_radius: Radius of each agent
        max_speed: Maximum speed of each agent
        preferred_velocity_range: Range to sample preferred velocity magnitude from
        
    Returns:
        Dictionary with agent properties
    """
    print(f"Creating {num_agents} predefined agents...")
    
    # Default: random positions if not provided
    if initial_positions is None:
        initial_positions = np.random.uniform(-5, 5, (num_agents, 2))
    
    if initial_velocities is None:
        initial_velocities = np.zeros((num_agents, 2))
    
    # Standard RVO2 agent properties
    agent_dict = {
        'numAgents': num_agents,
        'initPos': initial_positions,
        'initVelocity': initial_velocities,
        'radius': np.ones((num_agents, 1)) * agent_radius,
        'maxSpeed': np.ones((num_agents, 1)) * max_speed,
        'neighborDist': np.ones((num_agents, 1)) * 15.0,  # Look-ahead distance
        'maxNeighbors': np.ones((num_agents, 1), dtype=int) * 10,
        'timeHorizon': np.ones((num_agents, 1)) * 3.0,
        'timeHorizonObst': np.ones((num_agents, 1)) * 3.0,
    }
    
    # Sample preferred velocities (direction + magnitude)
    pref_vel_magnitude = np.random.uniform(
        preferred_velocity_range[0],
        preferred_velocity_range[1],
        (num_agents, 1)
    )
    pref_vel_direction = np.random.uniform(-np.pi, np.pi, (num_agents, 1))
    agent_dict['prefVelocity'] = np.concatenate([
        pref_vel_magnitude * np.cos(pref_vel_direction),
        pref_vel_magnitude * np.sin(pref_vel_direction)
    ], axis=1)
    
    print(f"✓ Agent properties created:")
    print(f"  - Positions: {agent_dict['initPos'].shape}")
    print(f"  - Preferred velocities: {agent_dict['prefVelocity'].shape}")
    
    return agent_dict


# ==============================================================================
# INFERENCE PIPELINE FUNCTIONS (ORIGINAL WITH MODIFICATIONS)
# ==============================================================================

def load_orca_config(config_path: str) -> SceneEditingConfig:
    """Load configuration from JSON file."""
    cfg = SceneEditingConfig()
    if config_path and Path(config_path).exists():
        print(f"Loading config from {config_path}")
        external_cfg = json.load(open(config_path))
        cfg.update(**external_cfg)
    else:
        # Default ORCA config
        print("Using default ORCA configuration")
        cfg.eval_class = "Diffuser"
        cfg.trajdata_source_test = ["orca_maps"]
        cfg.trajdata_data_dirs = {"orca_maps": "./datasets/orca_sim"}
        cfg.num_scenes_per_batch = 1
        cfg.num_simulation_steps = 50
        cfg.ckpt.policy.ckpt_dir = "/home/linh/ductai_nguyen_ws/trace/ckpt/trace/orca_mixed"
        cfg.ckpt.policy.ckpt_key = "iter40000"
        cfg.trajdata_cache_location = "~/.unified_data_cache"
        
    return cfg


def create_inference_env(eval_cfg: SceneEditingConfig, device: torch.device):
    """
    Create environment and policy for inference.
    This mirrors the setup in scene_editor.py
    """
    print("\n" + "="*60)
    print("STEP 1: Initialize Model and Environment")
    print("="*60)
    
    # Set global trajdata environment
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    
    # Load policy and model (same as scene_editor.py)
    print(f"Loading policy with eval_class: {eval_cfg.eval_class}")
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
        eval_config=eval_cfg,
        exp_config=exp_config,
        device=device
    )
    env = env_builder.get_env()
    
    print(f"✓ Model loaded: {type(policy_model).__name__}")
    print(f"✓ Environment ready with {env.total_num_scenes} ORCA scenes")
    
    return env, rollout_policy, exp_config


def rollout_scene(
    env,
    rollout_policy: RolloutWrapper,
    scene_index: int,
    horizon: int = 50,
    n_step_action: int = 1,
) -> Tuple[List[Dict], Dict]:
    """
    Run inference on a single ORCA scene.
    
    Args:
        env: Simulation environment
        rollout_policy: Policy wrapper for action queries
        scene_index: Which ORCA scene to use
        horizon: Number of timesteps to predict
        n_step_action: Steps between policy queries
        
    Returns:
        (trajectories_list, observation_at_each_step)
    """
    print("\n" + "="*60)
    print(f"STEP 2: Run Inference on Scene {scene_index}")
    print("="*60)
    
    # Reset environment to scene
    print(f"Loading ORCA scene {scene_index}...")
    scenes_valid = env.reset(
        scene_indices=[scene_index],
        start_frame_index=None  # Use default start frame
    )
    
    if not scenes_valid[0]:
        print(f"❌ Scene {scene_index} invalid!")
        return None, None
    
    print(f"✓ Scene loaded with {env.current_num_agents} agents")
    
    # Store trajectory data at each timestep
    trajectories = []
    
    # Run rollout loop (same as guided_rollout in scene_edit_utils.py)
    print("Running rollout...")
    done = env.is_done()
    counter = 0
    
    while not done and counter < horizon:
        # Get observation (state of all agents)
        obs = env.get_observation()
        
        # Convert to torch and query policy for actions
        device = rollout_policy.device
        obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
        # Ensure history_availabilities is boolean type
        action = rollout_policy.get_action(obs_torch, step_index=counter)
        
        # Step environment forward
        env.step(action, num_steps_to_take=n_step_action, render=False)
        counter += n_step_action
        
        # Log trajectory
        timestamp = counter * 0.1  # ORCA is at 0.1s timestep
        env_info = env.get_info()
        trajectories.append({
            "timestamp": timestamp,
            "step": counter,
            "positions": env_info["buffer"][0]["centroid"][:, -1, :],  # (N, 1, 2)
            "yaws": obs.get("yaw", None),  # (N, 1)
            "speeds": obs.get("curr_speed", None),  # (N,)
            "full_observation": obs,
        })
        
        done = env.is_done()
    
    print(f"✓ Rollout complete: {len(trajectories)} timesteps")
    
    return trajectories, obs


def extract_trajectory_array(trajectories: List[Dict]) -> np.ndarray:
    """
    Convert trajectory list to clean numpy array.
    
    Returns:
        positions: (T, N, 2) - T timesteps, N agents, 2D positions
    """
    if not trajectories:
        return None
    
    positions = []
    for time_step in trajectories:
        pos = time_step["positions"]  # (N, 2)
        positions.append(pos)
    return np.stack(positions, axis=0)


def save_results(trajectories: List[Dict], output_dir: str = "./orca_results"):
    """Save trajectory results to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract positions array
    positions = extract_trajectory_array(trajectories)
    
    # Save as numpy
    np.save(f"{output_dir}/trajectories.npy", positions)
    
    # Save as JSON (for easy inspection)
    json_data = {
        "num_timesteps": len(trajectories),
        "num_agents": positions.shape[1],
        "timesteps": [traj["timestamp"] for traj in trajectories],
        "sample_positions": positions[:3].tolist(),  # First 3 timesteps
    }
    with open(f"{output_dir}/trajectory_info.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - trajectories.npy: shape {positions.shape}")
    print(f"  - trajectory_info.json: metadata")
    
    return positions


def visualize_trajectories(trajectories: List[Dict], output_path: str = "./orca_results/trajectory.png"):
    """Create a simple visualization of predicted trajectories."""
    print("\nCreating visualization...")
    
    positions = extract_trajectory_array(trajectories)
    if positions is None:
        print("No trajectories to visualize")
        return
    
    T, N, _ = positions.shape
    
    # Create figure
    fig, ax = plt.subplots(figsize=(40, 40))
    
    # Plot each agent's trajectory
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    
    for agent_idx in range(N):
        agent_traj = positions[:, agent_idx, :]  # (T, 2)
        
        # Plot trajectory line
        ax.plot(
            agent_traj[:, 0],
            agent_traj[:, 1],
            color=colors[agent_idx],
            linewidth=2,
            label=f"Agent {agent_idx}",
            alpha=0.7,
        )
        
        # Plot start point
        ax.scatter(
            agent_traj[0, 0],
            agent_traj[0, 1],
            color=colors[agent_idx],
            s=100,
            marker="o",
            edgecolors="black",
            zorder=5,
        )
        
        # Plot end point
        ax.scatter(
            agent_traj[-1, 0],
            agent_traj[-1, 1],
            color=colors[agent_idx],
            s=100,
            marker="X",
            edgecolors="black",
            zorder=5,
        )
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("ORCA Map: Predicted Agent Trajectories")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def print_trajectory_summary(trajectories: List[Dict]):
    """Print human-readable summary of results."""
    print("\n" + "="*60)
    print("STEP 3: Trajectory Summary")
    print("="*60)
    
    positions = extract_trajectory_array(trajectories)
    if positions is None:
        return
    
    T, N, _ = positions.shape
    print(f"\nPredicted trajectories:")
    print(f"  Timeline: {trajectories[0]['timestamp']:.1f}s to {trajectories[-1]['timestamp']:.1f}s")
    print(f"  Timesteps: {T}")
    print(f"  Agents: {N}")
    
    print(f"\nPositions range:")
    print(f"  X: [{positions[:, :, 0].min():.2f}, {positions[:, :, 0].max():.2f}] meters")
    print(f"  Y: [{positions[:, :, 1].min():.2f}, {positions[:, :, 1].max():.2f}] meters")
    
    print(f"\nSample trajectory (Agent 0):")
    for t in [0, T//2, T-1]:
        pos = positions[t, 0, :]
        ts = trajectories[t]["timestamp"]
        print(f"  t={ts:.1f}s: position = ({pos[0]:.2f}, {pos[1]:.2f})")


def main(config_path: str, scene_index: int, horizon: int):
    """Main inference pipeline."""
    print("\n" + "="*80)
    print(" "*20 + "ORCA MAP INFERENCE PIPELINE")
    print("="*80)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load config and create environment
    eval_cfg = load_orca_config(config_path)
    env, rollout_policy, exp_config = create_inference_env(eval_cfg, device)
    
    # Run inference
    trajectories, final_obs = rollout_scene(
        env=env,
        rollout_policy=rollout_policy,
        scene_index=scene_index,
        horizon=horizon,
    )
    
    if trajectories is None:
        print("❌ Inference failed!")
        return
    
    # Process results
    print("\n" + "="*60)
    print("STEP 3: Process and Save Results")
    print("="*60)
    
    # Save
    positions = save_results(trajectories)
    
    # Visualize
    visualize_trajectories(trajectories)
    
    # Summary
    print_trajectory_summary(trajectories)
    
    # Access results programmatically
    print("\n" + "="*60)
    print("Accessing Results Programmatically")
    print("="*60)
    print(f"""
# Load saved trajectories
positions = np.load('orca_results/trajectories.npy')  # (T, N, 2)

# Iterate through timesteps
for t in range(positions.shape[0]):
    timestep = trajectories[t]['timestamp']
    agents_pos = positions[t]  # (N, 2)
    print(f"At t={timestep:.1f}s, agents at: {{agents_pos}}")

# Access single agent trajectory
agent_0_traj = positions[:, 0, :]  # All timesteps, first agent
print(f"Agent 0 trajectory shape: {{agent_0_traj.shape}}")
""")
    
    print("\n✅ Pipeline complete!")


def main_with_arena_world(
    config_path: str,
    arena_world_path: str,
    num_agents: int,
    initial_positions: Optional[np.ndarray] = None,
    horizon: int = 50,
    output_dir: str = "./orca_arena_results",
):
    """
    Main inference pipeline with Arena world conversion.
    
    This pipeline:
    1. Loads an Arena simulation world
    2. Converts it to ORCA map format (without saving)
    3. Creates predefined agents with initial positions
    4. Runs inference using the TRACE model
    
    Args:
        config_path: Path to SceneEditingConfig JSON
        arena_world_path: Path to Arena world directory
        num_agents: Number of agents to create
        initial_positions: (num_agents, 2) array of initial positions. If None, randomly sampled.
        horizon: Number of timesteps to predict
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print(" "*15 + "ARENA WORLD → ORCA MAP INFERENCE PIPELINE")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # ========================================================================
    # STEP 1: Load Arena World and Convert to ORCA Map
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: Arena World Conversion")
    print("="*60)
    
    world = load_arena_world(arena_world_path)
    orca_map = arena_world_to_orca_map(world, wall_thickness=0.2)
    print(f"✓ Successfully converted Arena world to ORCA map with {len(orca_map)} obstacles")
    
    # ========================================================================
    # STEP 2: Create Predefined Agents
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: Predefined Agent Creation")
    print("="*60)
    
    agent_dict = create_predefined_agents(
        num_agents=num_agents,
        initial_positions=initial_positions,
        agent_radius=0.4,
        max_speed=2.0,
        preferred_velocity_range=(1.0, 2.0),
    )
    
    # ========================================================================
    # STEP 3: Load Model and Create Environment
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: Initialize TRACE Model and Environment")
    print("="*60)
    
    eval_cfg = load_orca_config(config_path)
    
    # Temporarily store arena data in eval_cfg for reference
    eval_cfg.arena_world_path = arena_world_path
    eval_cfg.orca_map = orca_map
    eval_cfg.predefined_agents = agent_dict
    
    env, rollout_policy, exp_config = create_inference_env(eval_cfg, device)
    
    # ========================================================================
    # STEP 4: Run Inference with Custom Map and Agents
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: Run Inference")
    print("="*60)
    
    trajectories, final_obs = rollout_scene(
        env=env,
        rollout_policy=rollout_policy,
        scene_index=0,  # Use first scene from dataset as template
        horizon=horizon,
    )
    
    if trajectories is None:
        print("❌ Inference failed!")
        return
    
    # ========================================================================
    # STEP 5: Process and Visualize Results
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 5: Results Processing")
    print("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract positions
    positions = extract_trajectory_array(trajectories)
    
    # Save trajectories
    np.save(f"{output_dir}/trajectories.npy", positions)
    
    # Save metadata
    metadata = {
        "arena_world_path": arena_world_path,
        "num_agents": num_agents,
        "num_timesteps": len(trajectories),
        "num_obstacles": len(orca_map),
        "horizon": horizon,
        "timesteps": [traj["timestamp"] for traj in trajectories],
        "initial_positions": initial_positions.tolist() if initial_positions is not None else None,
        "sample_positions": positions[:min(3, len(positions))].tolist(),
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save ORCA map visualization
    try:
        visualize_orca_map_with_trajectories(orca_map, positions, output_dir)
    except Exception as e:
        print(f"Warning: Could not visualize map with trajectories: {e}")
    
    # Visualize trajectories
    visualize_trajectories(trajectories, f"{output_dir}/trajectory.png")
    
    # Print summary
    print_trajectory_summary(trajectories)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"✓ Trajectories saved to: {output_dir}/trajectories.npy")
    print(f"✓ Metadata saved to: {output_dir}/metadata.json")
    print(f"✓ Shape: {positions.shape} (timesteps, agents, coordinates)")
    print(f"✓ Obstacles from Arena world: {len(orca_map)}")
    
    print("\n✅ Arena-based inference pipeline complete!")
    
    return positions, orca_map, agent_dict


def visualize_orca_map_with_trajectories(
    obstacles: List[List[Tuple[float, float]]],
    trajectories: np.ndarray,
    output_dir: str = "./orca_arena_results",
):
    """
    Visualize ORCA map obstacles with predicted trajectories.
    
    Args:
        obstacles: List of polygons representing ORCA map
        trajectories: (T, N, 2) array of positions
        output_dir: Directory to save visualization
    """
    print("Creating map visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot obstacles
    from matplotlib.patches import Polygon as mplPolygon
    for obs_idx, obs_poly in enumerate(obstacles):
        poly_patch = mplPolygon(
            obs_poly,
            color="gray",
            fill=True,
            alpha=0.5,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(poly_patch)
    
    # Plot trajectories
    T, N, _ = trajectories.shape
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    
    for agent_idx in range(N):
        agent_traj = trajectories[:, agent_idx, :]
        
        # Trajectory line
        ax.plot(
            agent_traj[:, 0],
            agent_traj[:, 1],
            color=colors[agent_idx],
            linewidth=2.5,
            label=f"Agent {agent_idx}",
            alpha=0.8,
            zorder=3,
        )
        
        # Start point (circle)
        ax.scatter(
            agent_traj[0, 0],
            agent_traj[0, 1],
            color=colors[agent_idx],
            s=150,
            marker="o",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
        
        # End point (star)
        ax.scatter(
            agent_traj[-1, 0],
            agent_traj[-1, 1],
            color=colors[agent_idx],
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )
    
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title("Arena World → ORCA Map with TRACE's Predicted Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    
    output_path = f"{output_dir}/arena_map_trajectories.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Map visualization saved: {output_path}")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ORCA Map Inference Example (with optional Arena world conversion)"
    )
    
    # Core arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to SceneEditingConfig JSON (if None, uses defaults)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Number of timesteps to predict",
    )
    
    # Standard dataset mode
    parser.add_argument(
        "--scene",
        type=int,
        default=0,
        help="Scene index to use from ORCA dataset (standard mode only)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="Number of agents to create (Arena world mode only)",
    )
    parser.add_argument(
        "--agent-positions",
        type=str,
        default=None,
        help="Path to JSON file with initial agent positions array (Arena world mode). Format: [[x1, y1], [x2, y2], ...]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./orca_arena_results",
        help="Directory to save results (Arena world mode only)",
    )
    
    args = parser.parse_args()
    
    # Load initial positions from JSON if provided
    initial_positions = None
    if args.agent_positions:
        try:
            with open(args.agent_positions, "r") as f:
                initial_positions = np.array(json.load(f))
                print(f"Loaded initial positions from {args.agent_positions}")
        except Exception as e:
            print(f"Warning: Could not load agent positions from {args.agent_positions}: {e}")
    
    world_path = "/home/linh/ductai_nguyen_ws/Arena_ws/install/arena_simulation_setup/share/arena_simulation_setup/worlds/hospital_1"

    # Arena world conversion mode
    main_with_arena_world(
        config_path=args.config,
        arena_world_path=world_path,
        num_agents=args.num_agents,
        initial_positions=initial_positions,
        horizon=args.horizon,
        output_dir=args.output_dir,
    )