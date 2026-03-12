"""
COMPREHENSIVE GUIDE: TRACE Model with Synthetic Waypoint Guidance

This document answers all your questions about using GlobalTargetPosGuidance
with the TRACE model for Arena world inference.

================================================================================
QUESTION 1: Does TRACE need ground truth trajectories for GlobalTargetPosGuidance?
================================================================================

SHORT ANSWER: NO

DETAILED EXPLANATION:
- GlobalTargetPosGuidance (from tbsim/utils/guidance_loss.py, line 673+) only requires:
  1. Current agent state (positions, velocities) - from the scene
  2. Target waypoint positions - provided by you
  3. Guidance parameters (urgency, speed, etc.)

- It does NOT require ground truth future trajectories
- The TRACE diffusion model GENERATES trajectories that reach your waypoints
- Guidance works by computing a loss term during diffusion that guides samples
  toward the target, then choosing the best sample

CODE EVIDENCE (guidance_loss.py, GlobalTargetPosLoss.forward):
```python
def forward(self, x, data_batch, agt_mask=None):
    # x: trajectory samples from diffusion (B, N, T, 6)
    # data_batch: current scene state
    # Guidance computes distance to target and creates loss
    local_target_pos = transform_points_tensor(self.target_pos, agent_from_world)
    loss = compute_progress_loss(pos_pred, local_target_pos, urgency, ...)
    return loss  # Used to select best sample, no GT needed!
```

================================================================================
QUESTION 2: Can you synthesize trajectories by providing sparse waypoints?
================================================================================

SHORT ANSWER: YES, ABSOLUTELY

HOW IT WORKS:
1. You provide sparse waypoints (e.g., [(10.5, 0.0)])
2. You specify urgency factors (0-1) for each waypoint:
   - urgency = 1.0: agent should reach it ASAP (straight line path)
   - urgency = 0.5: agent should gradually move toward it
   - urgency = 0.0: minimal progress required

3. The model generates smooth, natural trajectories that:
   - Start from current agent position
   - Gradually approach the waypoint
   - Respect learned patterns from training data

4. For multiple waypoints:
   - Create separate guidance configs for different waypoints
   - Use GlobalTargetPosAtTimeLoss for time-constrained waypoints
   - Chain waypoints by updating target at inference time

EXAMPLE - Single waypoint:
```python
waypoint = (10.5, 0.0)
guidance_config = {
    'name': 'global_target_pos',
    'params': {
        'target_pos': [waypoint],  # Where to go
        'urgency': [0.8],          # How fast (0=slow, 1=fast)
        'pref_speed': 1.42,        # m/s, assumed walking speed
        'dt': 0.1,                 # timestep, same as model
        'min_progress_dist': 0.5,  # minimum progress per step
    },
    'agents': [0]  # which agents
}
```

================================================================================
QUESTION 3: Is the number of inferred frames dependent on ORCA dataset?
================================================================================

SHORT ANSWER: NO

DETAILED EXPLANATION:

The inference horizon is controlled by CONFIG, not the dataset:

1. SceneEditingConfig.num_simulation_steps:
   - This parameter controls how many steps to generate
   - Set in code: cfg.num_simulation_steps = 100
   - Each scene uses this horizon, regardless of dataset

2. What varies by scene/dataset:
   - Valid frame ranges: Different datasets have different temporal bounds
   - Start frame index: Can vary per scene
   - But the inference horizon is always the same (set by config)

3. Each scene can have different characteristics:
   - ORCA: Typically synthetic, controlled environments
   - nuScenes: Real driving data, different dynamics
   - Arena: Pedestrian simulation, similar to ORCA
   - But all use the SAME num_simulation_steps for inference

EVIDENCE (SceneEditingConfig in configs/eval):
```json
{
    "num_simulation_steps": 100,  // <-- This controls horizon
    "num_scenes_to_evaluate": 10,
    "num_scenes_per_batch": 1,
}
```

To change inference length, modify:
```python
cfg.num_simulation_steps = 200  # Generate 200 steps instead of 100
```

================================================================================
QUESTION 4: How to use with Arena world? (Your main goal)
================================================================================

RECOMMENDED WORKFLOW:

1. Load Arena world (see arena_trace_pipeline_v2.py):
   ```python
   from arena_simulation_setup.tree.World import World
   world = World(path=Path(arena_world_path))
   ```

2. Extract initial agent positions from Arena:
   ```python
   initial_positions = world.load_agent_positions()  # (N, 2)
   ```

3. Define waypoints for each agent:
   ```python
   waypoints = {
       agent_0: [(5.0, 5.0), (10.5, 0.0)],  # Multiple waypoints
       agent_1: [(8.0, 2.0)],
   }
   ```

4. Create guidance configs:
   ```python
   guidance_configs = []
   for agent_id, waypoints_list in waypoints.items():
       # For first waypoint
       config = {
           'name': 'global_target_pos',
           'weight': 1.0,  # REQUIRED! Guidance strength
           'params': {
               'target_pos': [waypoints_list[0]],
               'urgency': [0.8],
               'pref_speed': 1.42,
               'dt': 0.1,
               'min_progress_dist': 0.5,
           },
           'agents': [agent_id]
       }
       guidance_configs.append(config)
   
   # CRITICAL: [[config1, config2], [configs for scene 2], ...]
   guidance_config_list = [guidance_configs]  # List of scenes, each with list of configs
   ```

5. Run inference:
   ```python
   # Load TRACE model
   cfg = SceneEditingConfig()
   cfg.evalclass = "Diffuser"
   # ... other settings ...
   
   # Create environment with Arena data
   env = create_environment_for_arena(cfg, initial_positions)
   
   # Run with guidance
   results = guided_rollout(
       env=env,
       policy=rollout_policy,
       policy_model=policy_model,
       guidance_config=guidance_config_list,
       horizon=cfg.num_simulation_steps,
   )
   ```

6. Visualize results:
   ```python
   # Use draw_scene_data() from viz_utils.py
   from tbsim.utils.viz_utils import draw_scene_data
   
   fig, ax = plt.subplots()
   draw_scene_data(
       ax=ax,
       scene_name="arena_scene",
       scene_data=results['scene_data'],
       starting_frame=0,
       rasterizer=renderer,
       guidance_config=guidance_config_list,
       draw_trajectory=True,
       draw_action=True,
   )
   plt.savefig("arena_trace_result.png")
   ```

================================================================================
VALID GUIDANCE CONFIGURATIONS
================================================================================

The code in tbsim/utils/guidance_loss.py defines all valid guidance types:

1. GlobalTargetPosGuidance (recommended for waypoints):
   - Reaches target at some point in future
   - No specific time constraint
   - Use for: general waypoint following

2. GlobalTargetPosAtTimeLoss:
   - Reaches target at specific timestep
   - Use for: time-constrained navigation

3. TargetPosLoss (local frame):
   - Same as global but in agent's local coordinate frame
   - Use for: relative targets

4. AgentCollisionLoss:
   - Avoid collisions with other agents
   - Can combine with waypoint guidance

5. MapCollisionLoss:
   - Stay on roads/valid areas
   - Use for: Arena obstacle avoidance

6. TargetSpeedLoss:
   - Maintain specific speed
   - Use for: speed-controlled navigation

VALID CONFIG STRUCTURE:
```python
{
    'name': 'global_target_pos',  # Must match GUIDANCE_FUNC_MAP keys
    'params': {
        'target_pos': (N, 2) list,  # Target position per agent
        'urgency': (N,) list,       # [0, 1] for each agent
        'pref_speed': float,        # Default 1.42 m/s
        'dt': float,                # Timestep, match model!
        'min_progress_dist': float, # Default 0.5 m
    },
    'agents': list of agent indices  # Which agents to apply to
}
```

================================================================================
IMPLEMENTATION CHECKLIST
================================================================================

For your Arena + TRACE integration:

□ Load Arena world and extract agent positions
□ Define waypoints for each agent
□ Create valid guidance configs (verify with verify_guidance_config_list())
□ Load TRACE model with checkpoint
□ Create environment with proper config settings:
    □ Set num_simulation_steps to desired inference length
    □ Enable guidance in policy settings (class_free_guide_w > 0)
    □ Set guide_clean=True for better results
    □ num_action_samples=20 (more samples = better guidance)
□ Run guided_rollout with guidance configs
□ Extract trajectories from results
□ Visualize with draw_scene_data() or create custom visualization
□ Validate that guidance is working:
    □ Agents should move toward waypoints
    □ Trajectories should be smooth
    □ No unrealistic jumps

================================================================================
COMMON ISSUES & SOLUTIONS
================================================================================

ISSUE: "Guidance config validation failed"
SOLUTION: 
- Make sure 'name' matches a key in GUIDANCE_FUNC_MAP (guidance_loss.py)
- Verify all required params are present
- Check dimensions match (target_pos should be list of (x,y) tuples)

ISSUE: "Model not following waypoint"
SOLUTION:
- Increase urgency (try 1.0 instead of 0.5)
- Enable classifier-free guidance (class_free_guide_w = 1.0)
- Use guide_clean=True
- Increase num_action_samples

ISSUE: "Trajectories too short"
SOLUTION:
- Check num_simulation_steps in config
- Verify inference is completing without errors
- Check scene has valid agents

ISSUE: "Memory error during inference"
SOLUTION:
- Reduce num_action_samples
- Use smaller scene batches (num_scenes_per_batch=1)
- Reduce num_simulation_steps
- Clear CUDA cache between scenes

================================================================================
KEY FILES REFERENCE
================================================================================

1. tbsim/utils/guidance_loss.py:
   - GlobalTargetPosLoss class (line 673+)
   - GuidanceConfig definition (line 69+)
   - GUIDANCE_FUNC_MAP dictionary (line 865+)

2. scripts/scene_editor.py:
   - run_scene_editor() function - full inference pipeline
   - How to set up configs and run rollouts

3. tbsim/utils/scene_edit_utils.py:
   - guided_rollout() function - core inference logic
   - heuristic_global_target_pos() - GT-based config creation
   - compute_heuristic_guidance() - how to create configs

4. tbsim/utils/viz_utils.py:
   - draw_scene_data() - visualization function
   - visualize_guided_rollout() - full visualization pipeline

5. configs/eval/orca/:
   - Example evaluation configs
   - target_pos.json - waypoint guidance example

================================================================================
TESTING YOUR IMPLEMENTATION
================================================================================

Two test scripts are provided:

1. test_waypoint_guidance.py:
   - Analyzes requirements theoretically
   - Creates synthetic guidance configs
   - Verifies config validity
   - Provides detailed explanations

2. test_waypoint_guidance_advanced.py:
   - Full integration test matching scene_editor.py
   - Runs actual inference with waypoint guidance
   - Generates visualizations
   - Usage: python test_waypoint_guidance_advanced.py \
        --waypoint_x 10.5 --waypoint_y 0.0 \
        --urgency 0.8 --num_scenes 3

Both scripts are self-contained and can be run independently to verify
that waypoint guidance works correctly before integrating with Arena.

================================================================================
NEXT STEPS
================================================================================

1. Run the test scripts to verify understanding
2. Modify test_waypoint_guidance_advanced.py to load Arena world instead of ORCA
3. Create guidance configs for your Arena agents
4. Visualize results
5. Iterate on waypoints and urgency parameters

Good luck! The TRACE model is very flexible with guidance.
"""

print(__doc__)
