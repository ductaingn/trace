import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from tbsim.utils.viz_utils import draw_scene_data

# Dummy rasterizer that returns a blank image and a simple raster_from_world transform
class DummyRasterizer:
    def __init__(self, img_size=600, px_per_m=20):
        self.img_size = img_size
        self.px_per_m = px_per_m

    def render(self, ras_pos, ras_yaw, scene_name=None):
        # blank white image
        state_im = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        # raster_from_world: maps world coords to raster pixels
        # scale, flip y, translate center
        s = self.px_per_m
        tx = self.img_size / 2.0
        ty = self.img_size / 2.0
        raster_from_world = np.array([[s, 0.0, tx], [0.0, -s, ty], [0.0, 0.0, 1.0]])
        return state_im, raster_from_world

    def do_render_map(self, scene_name):
        return False


def interpolate_waypoints(waypoints, times, T):
    # waypoints: (K,2), times: (K,) with values in [0, T-1]
    traj = np.zeros((T, 2), dtype=float)
    for i in range(len(times) - 1):
        t0, t1 = int(times[i]), int(times[i + 1])
        p0, p1 = waypoints[i], waypoints[i + 1]
        if t1 == t0:
            traj[t0] = p0
            continue
        for t in range(t0, t1 + 1):
            alpha = (t - t0) / max(1, (t1 - t0))
            traj[t] = (1 - alpha) * p0 + alpha * p1
    # fill leading/trailing values
    if times[0] > 0:
        traj[: int(times[0])] = waypoints[0]
    if times[-1] < T - 1:
        traj[int(times[-1]) : ] = waypoints[-1]
    return traj


def make_scene_data(num_agents=2, T=50):
    # create simple waypoints per agent (sparse fewer than T)
    agents_waypoints = [
        np.array([[ -4.0, -2.0], [0.0, 2.0], [6.0, 2.0]]),
        np.array([[ -2.0, 4.0], [2.0, 0.0]])
    ]
    agents_times = [np.array([0, T//2, T-1]), np.array([0, T-1])]

    centroids = np.zeros((num_agents, T, 2), dtype=float)
    yaws = np.zeros((num_agents, T, 1), dtype=float)
    extent = np.zeros((num_agents, T, 2), dtype=float)
    for i in range(num_agents):
        traj = interpolate_waypoints(agents_waypoints[i], agents_times[i], T)
        centroids[i] = traj
        # simple yaw pointing along velocity (approx)
        vel = np.vstack([traj[1] - traj[0], traj[1:] - traj[:-1]])
        yaw = np.arctan2(vel[:, 1], vel[:, 0])
        yaws[i, :, 0] = yaw
        extent[i, :, :] = 0.4

    scene_data = {
        "centroid": centroids,
        "yaw": yaws,
        "extent": extent,
    }
    return scene_data


def main():
    out_dir = os.path.join(".", "demo_out")
    os.makedirs(out_dir, exist_ok=True)

    scene_data = make_scene_data(num_agents=2, T=60)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # Simple plot without draw_scene_data (just plot trajectories directly)
    centroids = scene_data["centroid"]  # (N, T, 2)
    yaws = scene_data["yaw"]  # (N, T, 1)
    
    colors = ["blue", "red"]
    for agent_idx in range(centroids.shape[0]):
        traj = centroids[agent_idx]  # (T, 2)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[agent_idx], linewidth=2, label=f"Agent {agent_idx}")
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[agent_idx], s=100, marker="o", edgecolors="black", label=f"Start Agent {agent_idx}")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[agent_idx], s=150, marker="*", edgecolors="black", label=f"End Agent {agent_idx}")
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Demo: Waypoint Interpolation for Agents")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    out_path = os.path.join(out_dir, "demo_waypoints.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved demo visualization to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
