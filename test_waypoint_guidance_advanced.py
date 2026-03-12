"""
Advanced test script that mirrors scene_editor.py pattern.
Tests GlobalTargetPosGuidance with synthetic waypoints and visualization.

This script addresses:
1. Valid guidance config creation 
2. Proper initialization matching scene_editor.py
3. Visualization of results
"""

import numpy as np
import torch
import json
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add repo to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import importlib
from tbsim.utils.trajdata_utils import (
    set_global_trajdata_batch_env,
    set_global_trajdata_batch_raster_cfg,
)
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.utils.scene_edit_utils import guided_rollout
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.tensor_utils import map_ndarray
from tbsim.utils.guidance_loss import verify_guidance_config_list
from tbsim.utils.viz_utils import visualize_guided_rollout, get_trajdata_renderer


def create_synthetic_waypoint_guidance(
    num_agents: int,
    waypoint: Tuple[float, float] = (10.5, 0.0),
    urgency: float = 0.8,
    pref_speed: float = 1.42,
    dt: float = 0.1,
    min_progress_dist: float = 0.5,
    weight: float = 1.0,
    agent_indices: List[int] = None,
) -> List[List[Dict]]:
    """
    Create valid guidance config matching scene_editor.py pattern.
    
    CRITICAL: Must have structure [[{configs}]] for DiffuserGuidance
    - Outer list: scenes
    - Middle list: configs per scene
    - Dict: individual config with 'name', 'weight', 'params', 'agents'
    """
    if agent_indices is None:
        agent_indices = list(range(num_agents))
    
    # Create guidance config with ALL required keys
    guide_config = {
        'name': 'global_target_pos',
        'weight': weight,  # CRITICAL: This was missing!
        'params': {
            'target_pos': [list(waypoint) for _ in agent_indices],
            'urgency': [urgency] * len(agent_indices),
            'pref_speed': pref_speed,
            'dt': dt,
            'min_progress_dist': min_progress_dist,
        },
        'agents': agent_indices
    }
    
    # CRITICAL: Return [[{config}]] not [config]
    # Structure: list of scenes, each containing list of configs
    return [[guide_config]]


def run_scene_editor_with_waypoints(
    eval_cfg: SceneEditingConfig,
    save_cfg: bool = True,
    data_to_disk: bool = False,
    render_to_video: bool = False,
    render_to_img: bool = True,
    render_cfg: Dict = None,
    waypoint: Tuple[float, float] = (10.5, 0.0),
    urgency: float = 0.8,
):
    """
    Run scene editor with synthetic waypoint guidance.
    Mirrors the run_scene_editor function from scripts/scene_editor.py
    """
    
    # Setup (following scene_editor.py)
    print('\n' + '=' * 80)
    print('STEP 1: INITIALIZATION')
    print('=' * 80)
    
    # Set global trajdata environment
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    
    # For reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(eval_cfg.seed)
    
    # Create output directories
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    if render_to_video:
        os.makedirs(os.path.join(eval_cfg.results_dir, "videos/"), exist_ok=True)
    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    
    if save_cfg:
        with open(os.path.join(eval_cfg.results_dir, "config.json"), "w") as f:
            json.dump(vars(eval_cfg), f, indent=2, default=str)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load policy (following scene_editor.py pattern)
    print('\n' + '=' * 80)
    print('STEP 2: LOAD POLICY AND MODEL')
    print('=' * 80)
    
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    policy_model = policy.model
    
    print(f"✓ Model loaded: {type(policy_model).__name__}")
    print(f"✓ Eval class: {eval_cfg.eval_class}")
    
    # Setup rasterization
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # Create rollout wrapper
    rollout_policy = RolloutWrapper(agents_policy=policy)
    
    # Create environment
    print('\n' + '=' * 80)
    print('STEP 3: CREATE ENVIRONMENT')
    print('=' * 80)
    
    env_builder = EnvUnifiedBuilder(
        eval_config=eval_cfg, exp_config=exp_config, device=device
    )
    env = env_builder.get_env()
    
    print(f"✓ Environment created")
    print(f"✓ Total scenes available: {env.total_num_scenes}")
    
    # Setup renderer if needed
    render_rasterizer = None
    if render_to_video or render_to_img:
        print('\n' + '=' * 80)
        print('STEP 4: SETUP RENDERER')
        print('=' * 80)
        
        render_rasterizer = get_trajdata_renderer(
            eval_cfg.trajdata_source_test,
            eval_cfg.trajdata_data_dirs,
            raster_size=render_cfg['size'],
            px_per_m=render_cfg['px_per_m'],
            rebuild_maps=False,
            cache_location=eval_cfg.trajdata_cache_location
        )
        print("✓ Renderer initialized")
    
    # Run scenes
    print('\n' + '=' * 80)
    print('STEP 5: RUN INFERENCE WITH SYNTHETIC WAYPOINT GUIDANCE')
    print('=' * 80)
    
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    
    while scene_i < min(eval_cfg.num_scenes_to_evaluate, len(eval_scenes)):
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        
        print(f"\nProcessing scenes: {scene_indices}")
        
        # Check scene validity
        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
        valid_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        
        if len(valid_scene_indices) == 0:
            print(f"  ✗ No valid scenes in {scene_indices}")
            continue
        
        print(f"  ✓ Valid scenes: {valid_scene_indices}")
        print(f"  ✓ Agents in scene: {env.current_num_agents}")
        
        # Create guidance config for this scene
        # IMPORTANT: The guidance config needs to match the actual scene
        num_agents = env.current_num_agents
        
        print(f"\n  Creating waypoint guidance config:")
        print(f"    Waypoint: {waypoint}")
        print(f"    Urgency: {urgency}")
        print(f"    Agents: {num_agents}")
        
        guidance_config = create_synthetic_waypoint_guidance(
            num_agents=num_agents,
            waypoint=waypoint,
            urgency=urgency,
            pref_speed=1.42,
            dt=exp_config.algo.dt,
            min_progress_dist=0.5,
        )
        
        # Verify config is valid
        try:
            is_valid = verify_guidance_config_list(guidance_config)
            print(f"  ✓ Guidance config is valid: {is_valid}")
        except Exception as e:
            print(f"  ✗ Guidance config validation failed: {e}")
            continue
        
        # Run guided rollout
        print(f"\n  Running inference...")
        start_time = time.time()
        
        stats, info = guided_rollout(
            env=env,
            policy=rollout_policy,
            policy_model=policy_model,
            n_step_action=1,
            guidance_config=guidance_config,
            scene_indices=valid_scene_indices,
            device=device,
            obs_to_torch=True,
            horizon=eval_cfg.num_simulation_steps,
            use_gt=False,
            start_frames=None,
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Inference completed in {elapsed:.2f}s")
        
        # Save scene data
        if stats and info:
            scene_name = f"scene_{valid_scene_indices[0]:06d}"
            scene_data_path = os.path.join(eval_cfg.results_dir, f"{scene_name}_data.json")
            
            # Extract serializable data
            scene_summary = {
                'scene_index': valid_scene_indices[0],
                'num_agents': num_agents,
                'waypoint': list(waypoint),
                'urgency': urgency,
                'num_steps': stats.get('num_steps', 'unknown'),
                'elapsed_time': elapsed,
            }
            
            with open(scene_data_path, 'w') as f:
                json.dump(scene_summary, f, indent=2)
            
            print(f"  ✓ Results saved to {scene_data_path}")
        
        # Visualization (if requested)
        if render_to_img and render_rasterizer and info:
            print(f"  Rendering visualization...")
            try:
                scene_name = f"scene_{valid_scene_indices[0]:06d}"
                
                visualize_guided_rollout(
                    output_dir=eval_cfg.results_dir,
                    rasterizer=render_rasterizer,
                    si=valid_scene_indices[0],
                    scene_data=info,
                    guidance_config=guidance_config,
                    filter_yaw=False,
                    fps=10,
                    n_step_action=1,
                    viz_diffusion_steps=False,
                    first_frame_only=True,
                    viz_traj=True,
                    sim_num=0,
                )
                print(f"  ✓ Visualization saved")
            except Exception as e:
                print(f"  ✗ Visualization failed: {e}")
    
    print('\n' + '=' * 80)
    print('INFERENCE COMPLETE')
    print('=' * 80)
    print(f"Results saved to: {eval_cfg.results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test TRACE model with synthetic waypoint guidance"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./waypoint_guidance_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--waypoint_x",
        type=float,
        default=10.5,
        help="X coordinate of waypoint"
    )
    parser.add_argument(
        "--waypoint_y",
        type=float,
        default=0.0,
        help="Y coordinate of waypoint"
    )
    parser.add_argument(
        "--urgency",
        type=float,
        default=0.8,
        help="Urgency factor (0-1)"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="Number of scenes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render to video"
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="Render image size"
    )
    
    args = parser.parse_args()
    
    # Create config
    cfg = SceneEditingConfig()
    cfg.eval_class = "Diffuser"
    cfg.trajdata_source_test = ["orca_maps"]
    cfg.trajdata_data_dirs = {"orca_maps": "./datasets/orca_sim"}
    cfg.num_scenes_per_batch = 1
    cfg.num_simulation_steps = 100
    cfg.num_scenes_to_evaluate = args.num_scenes
    cfg.results_dir = args.results_dir
    cfg.name = "waypoint_guidance_test"
    cfg.seed = 0
    
    # Policy settings with guidance enabled
    cfg.policy.num_action_samples = 20
    cfg.policy.class_free_guide_w = 1.0  # Enable guidance
    cfg.policy.guide_as_filter_only = False
    cfg.policy.guide_clean = True
    
    # Checkpoint settings
    cfg.ckpt.policy.ckpt_dir = "./ckpt/trace/orca_mixed"
    cfg.ckpt.policy.ckpt_key = "iter40000"
    cfg.trajdata_cache_location = "~/.unified_data_cache"
    
    # Determine eval scenes
    cfg.eval_scenes = list(range(min(10, args.num_scenes + 5)))
    
    render_cfg = {
        'size': args.render_size,
        'px_per_m': 12.0,
    }
    
    cfg.lock()
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "TRACE with Synthetic Waypoint Guidance - Advanced Test".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    waypoint = (args.waypoint_x, args.waypoint_y)
    
    try:
        run_scene_editor_with_waypoints(
            cfg,
            save_cfg=True,
            data_to_disk=False,
            render_to_video=args.render,
            render_to_img=True,
            render_cfg=render_cfg,
            waypoint=waypoint,
            urgency=args.urgency,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
