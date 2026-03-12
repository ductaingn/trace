# Bug Fixes Applied - Guidance Config Structure

## Summary of Issues Fixed

### Issue 1: Missing 'weight' Parameter
**Error:** `AttributeError: 'str' object has no attribute 'keys'` in `GuidanceConfig.from_dict()`

**Root Cause:** GuidanceConfig requires exactly 4 keys: `{'name', 'weight', 'params', 'agents'}`
- Location: [guidance_loss.py](guidance_loss.py#L88)
- The assertion was: `assert config_dict.keys() == {'name', 'weight', 'params', 'agents'}`

**Fix:** Added `'weight': 1.0` to all guidance config examples throughout the codebase.

**Files Updated:**
- ✅ test_waypoint_guidance.py - `create_synthetic_guidance_config()` function
- ✅ test_waypoint_guidance_advanced.py - `create_synthetic_waypoint_guidance()` function
- ✅ QUICK_START_PATTERNS.py - All guidance config examples
- ✅ 00_START_HERE.txt - Example code
- ✅ FINAL_SUMMARY.txt - All config examples
- ✅ WAYPOINT_GUIDANCE_GUIDE.md - All config examples
- ✅ TRACE_WAYPOINT_GUIDANCE_README.md - All config examples

### Issue 2: Wrong Structure (Single List vs List-of-Lists)
**Error:** When using `[{config}]` instead of `[[{config}]]`
- DiffuserGuidance expects: `guidance_config_list[scene_index][config_index]`
- With single list structure, it tries to iterate over string keys

**Root Cause:** Incorrect structure mismatch
- Location: [guidance_loss.py](guidance_loss.py#L881) in DiffuserGuidance.__init__()
- Code: `for cur_cfg in guidance_config_list[si]` where `si` is scene index
- Expects: `[[{config_dict}, ...], [{...}], ...]` (list of scenes, each with list of configs)

**Fix:** Changed all guidance config returns to use correct structure:
```python
# WRONG: return [guidance_config]
# CORRECT:
return [[guidance_config]]
```

**Files Updated:**
- ✅ test_waypoint_guidance.py - Returns `[[guidance_config]]`
- ✅ test_waypoint_guidance_advanced.py - Returns `[[guide_config]]`
- ✅ QUICK_START_PATTERNS.py - All patterns use `[[config]]`
- ✅ All documentation files - Show correct structure

## How to Verify Fixes

### Test 1: Basic Guidance Config (5 min)
```bash
cd /home/linh/ductai_nguyen_ws/trace
python3 test_waypoint_guidance.py
```
**Expected:** Should pass all analysis and validation phases

### Test 2: Full Integration with TRACE (15 min)
```bash
python3 test_waypoint_guidance_advanced.py
```
**Expected:** Should run TRACE inference with actual model

### Test 3: Visualization (10 min)
```bash
python3 visualization_examples.py
```
**Expected:** Should generate PNG visualizations

## Key Changes Summary

| File | Change |
|------|--------|
| All config dicts | Added `'weight': 1.0` field |
| create_*_guidance_config() functions | Now return `[[config]]` not `[config]` |
| Documentation examples | Updated to show correct structure |
| Comments | Added "CRITICAL" notes about structure |

## Valid Guidance Config Structure (AFTER FIXES)

```python
guidance_config = {
    'name': 'global_target_pos',       # Guidance type
    'weight': 1.0,                      # REQUIRED! Was missing
    'params': {
        'target_pos': [[10.5, 0.0]],   # Target positions
        'urgency': [0.8],               # Speed factors
        'pref_speed': 1.42,             # m/s
        'dt': 0.1,                      # timestep
        'min_progress_dist': 0.5,       # meters
    },
    'agents': [0]                       # Which agents
}

# CRITICAL: List-of-lists structure (was wrong before)
guidance_config_list = [[guidance_config]]  # [[configs for scene 0], [configs for scene 1], ...]
```

## Next Steps

1. Run the test scripts to verify they now work correctly
2. Copy patterns from QUICK_START_PATTERNS.py for your Arena integration
3. All documentation is now consistent and correct
