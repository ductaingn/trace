import numpy as np
import matplotlib.pyplot as plt
from tbsim.utils.viz_utils import get_trajdata_renderer
import os

# point this to the parent of "maps" (the same path you passed to load_orca_config)
data_root = "./orca_arena_results"
data_dirs = {"orca_maps": data_root}

# create renderer (will use cached maps if available)
renderer = get_trajdata_renderer(["orca_maps"], data_dirs, raster_size=224, px_per_m=5, rebuild_maps=False)

# show available scene keys (use one of these)
print("scenes:", list(renderer.scene_info.keys()))
scene_key = list(renderer.scene_info.keys())[0]  # usually 'scene_000000_orca_maps'

# load sim data to choose a position to center the raster on
sim = np.load(os.path.join(data_root, "maps", "scene_000000", "sim.npz"))
# choose mean agent position at t=0 (or any frame)
pos_world = sim["trackPos"][0, 0]  # [x, y]

# render raster (state_im is RGB image in [0,1])
state_im, raster_from_world = renderer.render(pos_world, 0.0, scene_key)
plt.imsave("orca_parsed_map.png", state_im)

# optional: overlay agent positions (first frame) onto the raster and save
pts_world = sim["trackPos"][:, 0, :]  # (N,2)
hom = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)  # (N,3)
pts_raster = (raster_from_world @ hom.T).T  # (N,3)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(state_im)
ax.scatter(pts_raster[:, 0], pts_raster[:, 1], c="red", s=30)
ax.set_axis_off()
plt.savefig("orca_map_with_agents.png", bbox_inches="tight", pad_inches=0)
print("Saved orca_parsed_map.png and orca_map_with_agents.png")