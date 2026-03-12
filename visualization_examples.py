"""
Visualization Guide: How to use draw_scene_data() to visualize TRACE results

This script demonstrates:
1. How to properly structure scene_data for visualization
2. How to use draw_scene_data() from viz_utils.py
3. How to visualize guidance trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from tbsim.utils.viz_utils import draw_scene_data


def create_synthetic_scene_data(
    num_agents: int = 3,
    num_timesteps: int = 100,
    waypoint: tuple = (10.5, 0.0),
) -> dict:
    """
    Create synthetic scene data for visualization demonstration.
    
    This shows the structure needed for draw_scene_data().
    In real usage, this comes from env.get_info() during inference.
    """
    
    # Example: 3 agents following a waypoint
    scene_data = {
        # Centroid trajectories (agent positions over time) - REQUIRED
        'centroid': np.random.randn(num_agents, num_timesteps, 2) * 0.5 + np.array([5.0, 5.0]),
        
        # Agent headings/yaw over time - OPTIONAL but recommended
        'heading': np.zeros((num_agents, num_timesteps)),
        
        # Ground truth trajectory (if available) - OPTIONAL
        'target_positions': np.random.randn(num_agents, num_timesteps, 2) * 0.3 + np.array([5.0, 5.0]),
        'target_availabilities': np.ones((num_agents, num_timesteps)),
        
        # Action trajectories (predicted next step) - OPTIONAL
        'action_traj_positions': np.random.randn(num_agents, num_timesteps, 2) * 0.2 + np.array([6.0, 5.0]),
        
        # Sample trajectories from diffusion - OPTIONAL
        'action_sample_positions': np.random.randn(num_agents, 5, num_timesteps, 2) * 0.3 + np.array([6.0, 5.0]),
    }
    
    return scene_data


def create_dummy_rasterizer():
    """
    Create a dummy rasterizer for visualization when you don't have
    map/scene data. This is useful for testing visualization code.
    """
    
    class DummyRasterizer:
        def render(self, ras_pos, ras_yaw, scene_name):
            """
            Return a dummy image and identity transform.
            """
            # Create empty image (500x500 pixels, 3 channels)
            img = np.ones((500, 500, 3)) * 0.9  # Light gray background
            
            # Identity transform (no scaling/rotation)
            raster_from_world = np.eye(3)
            
            return img, raster_from_world
    
    return DummyRasterizer()


def visualize_example_1_basic():
    """
    Example 1: Basic visualization with just trajectories
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Trajectory Visualization")
    print("=" * 80)
    
    # Create synthetic data
    scene_data = create_synthetic_scene_data(num_agents=3, num_timesteps=100)
    rasterizer = create_dummy_rasterizer()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw scene data
    # Key parameters:
    # - draw_trajectory=True: Show agent paths
    # - draw_action=False: Don't show action samples
    # - draw_diffusion_step=None: Don't show diffusion stages
    draw_scene_data(
        ax=ax,
        scene_name="example_scene_1",
        scene_data=scene_data,
        starting_frame=0,
        rasterizer=rasterizer,
        guidance_config=None,  # No guidance to visualize
        draw_trajectory=True,
        draw_action=False,
        draw_diffusion_step=None,
        n_step_action=1,
        draw_action_sample=False,
        traj_len=200,
        ras_pos=np.array([250, 250]),  # Center of raster
        linewidth=2.0,
    )
    
    ax.set_title("Example 1: Basic Trajectories")
    plt.tight_layout()
    plt.savefig("/tmp/example_1_trajectories.png", dpi=100)
    print("✓ Saved to /tmp/example_1_trajectories.png")
    plt.close()


def visualize_example_2_with_guidance():
    """
    Example 2: Visualization showing guidance waypoints
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Visualization with Guidance Waypoints")
    print("=" * 80)
    
    # Create synthetic data
    scene_data = create_synthetic_scene_data(num_agents=2, num_timesteps=100)
    rasterizer = create_dummy_rasterizer()
    
    # Create guidance config showing waypoint
    waypoint = (10.5, 0.0)
    guidance_config = {
        'name': 'global_target_pos',
        'params': {
            'target_pos': [waypoint, waypoint],  # For 2 agents
            'urgency': [0.8, 0.8],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0, 1]
    }
    guidance_config_list = [guidance_config]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw scene data WITH guidance
    draw_scene_data(
        ax=ax,
        scene_name="example_scene_2",
        scene_data=scene_data,
        starting_frame=0,
        rasterizer=rasterizer,
        guidance_config=guidance_config_list,  # Show guidance info
        draw_trajectory=True,
        draw_action=True,  # Show action samples
        draw_diffusion_step=None,
        n_step_action=5,
        draw_action_sample=False,
        traj_len=200,
        ras_pos=np.array([250, 250]),
        linewidth=2.0,
    )
    
    ax.set_title(f"Example 2: Guidance to Waypoint {waypoint}")
    plt.tight_layout()
    plt.savefig("/tmp/example_2_with_guidance.png", dpi=100)
    print("✓ Saved to /tmp/example_2_with_guidance.png")
    plt.close()


def visualize_example_3_detailed():
    """
    Example 3: Detailed visualization with all available information
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Detailed Visualization with All Features")
    print("=" * 80)
    
    # Create synthetic data with more samples
    scene_data = create_synthetic_scene_data(num_agents=4, num_timesteps=150)
    rasterizer = create_dummy_rasterizer()
    
    # Multiple guidance types for different agents
    guidance_config = {
        'name': 'global_target_pos',
        'params': {
            'target_pos': [[10.5, 0.0], [10.5, 0.0], [8.0, 5.0], [8.0, 5.0]],
            'urgency': [0.9, 0.9, 0.6, 0.6],
            'pref_speed': 1.42,
            'dt': 0.1,
            'min_progress_dist': 0.5,
        },
        'agents': [0, 1, 2, 3]
    }
    guidance_config_list = [guidance_config]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw with all features
    draw_scene_data(
        ax=ax,
        scene_name="example_scene_3",
        scene_data=scene_data,
        starting_frame=0,
        rasterizer=rasterizer,
        guidance_config=guidance_config_list,
        draw_trajectory=True,       # Show paths
        draw_action=True,           # Show predicted next actions
        draw_diffusion_step=None,   # Can set to specific step to debug
        n_step_action=10,           # Action frequency
        draw_action_sample=True,    # Show sample trajectories from diffusion
        traj_len=250,
        ras_pos=np.array([250, 250]),
        linewidth=2.5,
    )
    
    ax.set_title("Example 3: Detailed with Samples and Guidance")
    plt.tight_layout()
    plt.savefig("/tmp/example_3_detailed.png", dpi=100)
    print("✓ Saved to /tmp/example_3_detailed.png")
    plt.close()


def visualize_arena_style():
    """
    Example 4: How to visualize TRACE results for Arena scenarios
    
    This shows the pattern you'd use with arena_trace_pipeline_v2.py
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Arena-Style Visualization")
    print("=" * 80)
    
    # Simulate Arena scenario: 2 pedestrians navigating around obstacles
    num_agents = 2
    num_timesteps = 120
    
    # Create more realistic trajectories (curves instead of random)
    scene_data = {}
    
    # Agent 1: goes from (0, 0) to (10.5, 0.0)
    t = np.linspace(0, 1, num_timesteps)
    agent1_traj = np.stack([
        np.ones(num_timesteps) * 0.0 + t * 10.5,  # x: 0 -> 10.5
        np.ones(num_timesteps) * 0.0 + 0.5 * np.sin(t * 4 * np.pi),  # y: slight curve
    ], axis=1)
    
    # Agent 2: goes from (0, 2) to (10.5, 2.0)
    agent2_traj = np.stack([
        np.ones(num_timesteps) * 0.0 + t * 10.5,  # x: 0 -> 10.5
        np.ones(num_timesteps) * 2.0 - 0.5 * np.sin(t * 4 * np.pi),  # y: 2 with opposite curve
    ], axis=1)
    
    scene_data['centroid'] = np.stack([agent1_traj, agent2_traj], axis=0)
    scene_data['heading'] = np.zeros((num_agents, num_timesteps))
    
    rasterizer = create_dummy_rasterizer()
    
    # Guidance: both agents go to same waypoint
    waypoint = (10.5, 1.0)  # Between their paths
    guidance_config_list = [
        {
            'name': 'global_target_pos',
            'params': {
                'target_pos': [waypoint, waypoint],
                'urgency': [1.0, 1.0],  # High urgency for Arena
                'pref_speed': 1.42,
                'dt': 0.1,
                'min_progress_dist': 0.3,  # Tighter control for Arena
            },
            'agents': [0, 1]
        }
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    draw_scene_data(
        ax=ax,
        scene_name="arena_simulation",
        scene_data=scene_data,
        starting_frame=0,
        rasterizer=rasterizer,
        guidance_config=guidance_config_list,
        draw_trajectory=True,
        draw_action=True,
        draw_diffusion_step=None,
        n_step_action=15,
        draw_action_sample=False,  # Cleaner visualization for presentation
        traj_len=300,
        ras_pos=np.array([250, 250]),
        linewidth=3.0,
    )
    
    # Annotate waypoint
    ax.plot(10.5, 1.0, 'r*', markersize=20, label='Target Waypoint')
    ax.legend()
    
    ax.set_title("Arena Scenario: Pedestrians Following Waypoint", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    plt.tight_layout()
    plt.savefig("/tmp/example_4_arena_style.png", dpi=100)
    print("✓ Saved to /tmp/example_4_arena_style.png")
    plt.close()


def main():
    """Run all visualization examples"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "TRACE Visualization Examples with draw_scene_data()".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    print("These examples show how to visualize TRACE results using draw_scene_data()")
    print("from tbsim/utils/viz_utils.py")
    print()
    
    try:
        visualize_example_1_basic()
        visualize_example_2_with_guidance()
        visualize_example_3_detailed()
        visualize_arena_style()
        
        print("\n" + "=" * 80)
        print("✓ ALL EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nKey takeaways:")
        print("1. scene_data must contain 'centroid' key with shape (num_agents, time, 2)")
        print("2. Optional keys: 'heading', 'action_traj_positions', 'action_sample_positions'")
        print("3. guidance_config_list shows waypoints and visualization colors")
        print("4. draw_trajectory=True to show agent paths")
        print("5. draw_action=True to show predicted action samples")
        print("6. draw_action_sample=True to show diffusion samples (slower)")
        print()
        print("In real usage:")
        print("  env = create_environment(...)")
        print("  stats, info = guided_rollout(env, policy, guidance_config)")
        print("  draw_scene_data(ax, scene_data=info, guidance_config=guidance_config)")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
