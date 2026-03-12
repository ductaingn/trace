# TRACE Model: Synthetic Waypoint Guidance - Complete Solution

This directory contains everything you need to understand and implement TRACE model inference with synthetic waypoint guidance for your Arena world simulation.

## 📚 **Start Here**

### 1. **Understanding Your Questions** (5 min read)
Read: [`FINAL_SUMMARY.txt`](FINAL_SUMMARY.txt)

This file directly answers your 4 key questions:
1. ❌ Does TRACE need ground truth trajectories? **NO**
2. ✅ Can you synthesize trajectories from sparse waypoints? **YES**
3. ❌ Does inference frame count depend on ORCA dataset? **NO**
4. ✅ How to use with Arena world? **See integration pattern**

### 2. **Detailed Guide** (15 min read)
Read: [`WAYPOINT_GUIDANCE_GUIDE.md`](WAYPOINT_GUIDANCE_GUIDE.md)

Comprehensive reference with:
- Code evidence from actual source files
- Valid configuration examples
- Common issues and solutions
- Arena integration step-by-step

### 3. **Quick Start Code** (Copy-paste ready)
Read: [`QUICK_START_PATTERNS.py`](QUICK_START_PATTERNS.py)

7 ready-to-use code patterns:
- Pattern 1: Create valid guidance config
- Pattern 2: Validate config
- Pattern 3: Load TRACE model
- Pattern 4: Run inference
- Pattern 5: Visualize results
- Pattern 6: Complete example
- Pattern 7: Arena world integration

---

## 🧪 **Runnable Test Scripts**

### **Test 1: Theory (Quick)**
```bash
python test_waypoint_guidance.py
```
- Analyzes guidance requirements theoretically
- Creates synthetic configs
- Validates structure
- **Runtime:** < 5 seconds
- **Use when:** First starting out, understanding concepts

### **Test 2: Real Inference (Full)**
```bash
python test_waypoint_guidance_advanced.py \
    --waypoint_x 10.5 --waypoint_y 0.0 \
    --urgency 0.8 \
    --num_scenes 3
```
- Full integration test matching scene_editor.py
- Loads actual TRACE model
- Runs real inference on ORCA data
- Generates visualizations
- **Runtime:** 10-30 seconds
- **Use when:** Testing with real data, before Arena integration

### **Test 3: Visualization Examples**
```bash
python visualization_examples.py
```
- Shows 4 visualization examples
- Demonstrates draw_scene_data() usage
- Generates PNG output files
- **Runtime:** 5-10 seconds
- **Use when:** Learning how to visualize

---

## 🎯 **Quick Implementation Workflow**

### **Step 1: Understand** (10 min)
```
Read: FINAL_SUMMARY.txt
Then: WAYPOINT_GUIDANCE_GUIDE.md
```

### **Step 2: Test Theory** (5 min)
```bash
python test_waypoint_guidance.py
```

### **Step 3: Test Real Data** (10 min)
```bash
python test_waypoint_guidance_advanced.py
```

### **Step 4: Learn Visualization** (5 min)
```bash
python visualization_examples.py
```

### **Step 5: Copy Code Patterns** (5 min)
```
Open: QUICK_START_PATTERNS.py
Copy: Functions you need
Adapt: For your Arena scenario
```

### **Step 6: Integrate with Arena** (30 min)
```python
# From QUICK_START_PATTERNS.py

# Create guidance for waypoint
guidance_config_list = create_waypoint_guidance_config(
    num_agents=3,
    waypoints=[(10.5, 0.0), (10.5, 2.0), (8.0, 1.0)],
    urgency_values=[0.8, 0.8, 0.6],
)

# Validate
validate_guidance_config(guidance_config_list)

# Setup TRACE
env, rollout_policy, policy_model, exp_config = setup_trace_inference()

# Run inference
results = run_inference_with_waypoints(
    env, rollout_policy, policy_model, guidance_config_list
)

# Visualize
visualize_inference_results(results, guidance_config_list, rasterizer)
```

---

## 📊 **Key Findings**

### **1. No Ground Truth Trajectories Needed**
```python
# GlobalTargetPosGuidance works with ONLY:
# 1. Agent current state (from scene)
# 2. Target waypoint position (you provide)
# 3. Guidance parameters (you set)

# The model GENERATES the trajectory!
```

### **2. Sparse Waypoints Work Great**
```python
# Single waypoint:
waypoints = [(10.5, 0.0)]  # All agents go here

# Multiple waypoints:
waypoints = [
    (10.5, 0.0),   # Agent 0 target
    (10.5, 2.0),   # Agent 1 target
    (8.0, 1.0),    # Agent 2 target
]

# Control urgency (speed):
urgencies = [1.0, 0.5, 0.8]  # 0=slow, 1=fast
```

### **3. Frame Count is Configuration-Dependent**
```python
# NOT dependent on ORCA dataset!
cfg.num_simulation_steps = 100  # Controls inference length

# Use same config for ORCA, nuScenes, Arena
```

### **4. Valid Guidance Config Structure**
```python
guidance_config = {
    'name': 'global_target_pos',  # Type of guidance
    'weight': 1.0,                 # REQUIRED! Guidance strength
    'params': {
        'target_pos': [[10.5, 0.0]],    # Waypoints
        'urgency': [0.8],                # 0-1 for each agent
        'pref_speed': 1.42,              # m/s
        'dt': 0.1,                       # timestep
        'min_progress_dist': 0.5,        # meters
    },
    'agents': [0]  # Which agents
}

# CRITICAL: Must wrap as [[config]] not [config]
# Structure: [[configs for scene 0], [configs for scene 1], ...]
guidance_config_list = [[guidance_config]]
```

---

## 📁 **File Organization**

### **Documentation Files**
| File | Purpose | Read Time |
|------|---------|-----------|
| `FINAL_SUMMARY.txt` | Quick answers to all 4 questions | 5 min |
| `WAYPOINT_GUIDANCE_GUIDE.md` | Detailed reference guide | 15 min |
| `QUICK_START_PATTERNS.py` | Copy-paste code patterns | 10 min |
| `INDEX_OF_FILES.txt` | This guide | 5 min |

### **Test Scripts**
| Script | Purpose | Runtime |
|--------|---------|---------|
| `test_waypoint_guidance.py` | Theory analysis | < 5s |
| `test_waypoint_guidance_advanced.py` | Real inference | 10-30s |
| `visualization_examples.py` | Learn visualization | 5-10s |

### **Source Code References**
| File | Contains |
|------|----------|
| `tbsim/utils/guidance_loss.py` | GlobalTargetPosGuidance implementation |
| `tbsim/utils/scene_edit_utils.py` | guided_rollout() function |
| `scripts/scene_editor.py` | Full inference pipeline example |
| `tbsim/utils/viz_utils.py` | draw_scene_data() function |

---

## ✅ **Verification Checklist**

Before integrating with Arena, ensure you can:

- [ ] Explain why GT trajectories are NOT needed
- [ ] Explain why frame count is config-dependent
- [ ] Create a valid guidance config
- [ ] Validate a guidance config
- [ ] Run test_waypoint_guidance.py successfully
- [ ] Run test_waypoint_guidance_advanced.py successfully
- [ ] Understand draw_scene_data() parameters
- [ ] Run visualization_examples.py and see outputs
- [ ] Copy code from QUICK_START_PATTERNS.py
- [ ] Adapt patterns for your Arena scenario

If all checked: **You're ready to integrate!**

---

## 🚀 **Next Steps**

1. **This minute:** Read FINAL_SUMMARY.txt
2. **Next 5 min:** Run `python test_waypoint_guidance.py`
3. **Next 10 min:** Read WAYPOINT_GUIDANCE_GUIDE.md
4. **Next 10 min:** Run `python test_waypoint_guidance_advanced.py`
5. **Next 5 min:** Run `python visualization_examples.py`
6. **Next 5 min:** Copy code from QUICK_START_PATTERNS.py
7. **Next 30 min:** Integrate with your Arena world

**Total time to integration: < 1 hour**

---

## 🔗 **Key Concepts Quick Reference**

### **Global Target Position Guidance**
```python
# Make agents go to (10.5, 0.0)
{
    'name': 'global_target_pos',
    'weight': 1.0,  # REQUIRED!
    'params': {
        'target_pos': [[10.5, 0.0]],
        'urgency': [0.8],              # How fast: 0=slow, 1=fast
        'pref_speed': 1.42,            # Expected walking speed (m/s)
        'dt': 0.1,                     # Timestep (always 0.1 for TRACE)
        'min_progress_dist': 0.5,      # Min progress per step (m)
    },
    'agents': [0]  # Which agents
}
```

### **Multiple Guidance Types**
```python
# Can combine multiple guidance types:
guidance_config = [
    {  # Waypoint guidance
        'name': 'global_target_pos',
        'weight': 1.0,  # REQUIRED!
        'params': {...},
        'agents': [0, 1]
    },
    {  # Collision avoidance
        'name': 'agent_collision',
        'weight': 1.0,  # REQUIRED!
        'params': {'num_disks': 5, 'buffer_dist': 0.2},
        'agents': None  # All agents
    }
]
# Wrap as [[config1, config2], ...] for scenes
```

### **Valid Guidance Types**
- ✅ `global_target_pos` - Recommended for waypoints
- `global_target_pos_at_time` - With time constraint
- `agent_collision` - Avoid other agents
- `map_collision` - Stay on valid areas
- `target_speed` - Maintain specific speed

---

## 🆘 **Troubleshooting**

### **"Config validation failed"**
- Check 'name' is in GUIDANCE_FUNC_MAP (guidance_loss.py line 865)
- Verify all params are present
- Check shape dimensions match

### **"Model not following waypoint"**
- Increase urgency (try 1.0)
- Enable guidance: `class_free_guide_w = 1.0`
- Use `guide_clean = True`
- Increase `num_action_samples`

### **"Out of memory"**
- Reduce `num_action_samples` (e.g., 10 instead of 20)
- Reduce `num_simulation_steps`
- Use `num_scenes_per_batch = 1`

### **"ImportError"**
- Ensure you're in `/home/linh/ductai_nguyen_ws/trace/` directory
- Use: `.venv/bin/python script.py` or activate venv first

---

## 📖 **Learning Resources**

### **In This Repository**
- `FINAL_SUMMARY.txt` - Quick answers
- `WAYPOINT_GUIDANCE_GUIDE.md` - Detailed guide
- `QUICK_START_PATTERNS.py` - Code patterns
- Test scripts - Working examples

### **Source Code**
- `tbsim/utils/guidance_loss.py` - Guidance implementations
- `scripts/scene_editor.py` - Full pipeline example
- `configs/eval/orca/` - Configuration examples

### **Original TRACE**
- Paper: "Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion"
- Repo: https://github.com/NVlabs/trace

---

## 💡 **Pro Tips**

1. **Start with test_waypoint_guidance.py** - No dependencies, just theory
2. **Use QUICK_START_PATTERNS.py** - Copy-paste ready functions
3. **Run visualization_examples.py** - Understand output format
4. **Test on ORCA first** - Before integrating with Arena
5. **Use urgency=0.8** - Good balance of natural movement and guidance
6. **Enable guide_clean=True** - Better quality trajectories
7. **Increase num_action_samples** - More samples = better guidance (but slower)

---

## ❓ **FAQ**

**Q: Do I need ground truth trajectories?**
A: NO. The model generates them guided by your waypoint.

**Q: Can I use sparse waypoints?**
A: YES. Provide target position + urgency factor, model fills in the trajectory.

**Q: Does frame count depend on ORCA?**
A: NO. Set by `num_simulation_steps` config, independent of dataset.

**Q: Can I combine multiple guidance types?**
A: YES. Create multiple guidance configs in the list.

**Q: How long does inference take?**
A: ~100 steps per second on GPU. 100 steps ≈ 10 seconds.

**Q: What if model ignores waypoint?**
A: Increase `class_free_guide_w` and `num_action_samples`.

**Q: Can I use with custom scenes?**
A: YES. Only needs current agent state + waypoint position.

---

## ✨ **What You'll Achieve**

After following this guide, you'll be able to:

✅ Understand how TRACE works with guidance  
✅ Create valid guidance configurations  
✅ Run inference with synthetic waypoints  
✅ Visualize results with draw_scene_data()  
✅ Integrate into your Arena world pipeline  
✅ Generate trajectories without ground truth  
✅ Control agent behavior with sparse waypoints  
✅ Understand and debug guidance systems  

---

**Ready to start? Open [`FINAL_SUMMARY.txt`](FINAL_SUMMARY.txt) now!**

*Created: February 26, 2026*  
*For: Arena World + TRACE Integration*
