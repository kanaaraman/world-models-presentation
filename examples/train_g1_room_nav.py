"""Train the G1 to navigate the furnished room, inside MuJoCo, on a laptop CPU.

Hierarchical RL. Unitree's pretrained walking policy stays frozen as the
low-level controller (50 Hz). PPO learns a new high-level policy (5 Hz) that
picks walking commands (forward speed, turn rate) to reach a random goal
somewhere in the room, using goal direction and 8 "lidar" rays as observations.

    # train (about 30 minutes for 150k steps on 4 CPU cores), then evaluate
    uv run --python 3.12 --with mujoco --with numpy --with gymnasium --with stable-baselines3 python train_g1_room_nav.py --train 150000
    uv run --python 3.12 --with mujoco --with numpy --with gymnasium --with stable-baselines3 python train_g1_room_nav.py --eval
    uv run --python 3.12 --with mujoco --with numpy --with gymnasium --with stable-baselines3 python train_g1_room_nav.py --view

Training full-body skills (walking itself, or arms that grasp) needs far more
samples: use GPU simulators such as MuJoCo Playground / MJX or Isaac Lab, as
explained in the deck and the README.
"""

import argparse
import os
import time

import gymnasium as gym
import mujoco
import numpy as np

from g1_room_walk import G1, WalkPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "g1_room_nav_ppo.zip")
RAY_ANGLES = np.linspace(-np.pi, np.pi, 8, endpoint=False)


class G1RoomNav(gym.Env):
    """Reach a random goal without bumping into furniture. Action: [forward, turn] in [-1, 1]. One step = 0.2 s."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=150):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "g1_room.xml"))
        self.d = mujoco.MjData(self.m)
        self.g1 = G1(self.m, self.d, WalkPolicy(os.path.join(HERE, "g1", "g1_walk_policy.json")))
        self.pel = self.m.body("pelvis").id
        self.seed0 = seed
        self.max_steps = max_steps
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-5, 5, (14,), np.float32)
        mujoco.mj_forward(self.m, self.d)
        # obstacles used to reject goals that sit inside furniture (walls are handled by the sampling box)
        self.obstacles = [(self.d.geom_xpos[g][:2].copy(), np.hypot(*self.m.geom_size[g][:2]))
                          for g in range(self.m.ngeom)
                          if self.m.geom_bodyid[g] != 0 and self.m.body_rootid[self.m.geom_bodyid[g]] != self.pel]
        self.ray_group = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # furniture and walls are group 0; robot collision prims are group 3

    # --- helpers -----------------------------------------------------------
    def _yaw(self):
        w, x, y, z = self.d.xquat[self.pel]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _bumping(self):
        """True if any robot geom touches furniture (walls and floor don't count)."""
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = m.body_rootid[b1] == self.pel, m.body_rootid[b2] == self.pel
            if r1 != r2 and (b2 if r1 else b1) != 0:
                return True
        return False

    def _sample_goal(self):
        for _ in range(500):
            g = self.np_random.uniform([-2.6, -2.1], [2.6, 2.1])
            if all(np.linalg.norm(g - c) > r + 0.35 for c, r in self.obstacles) and np.linalg.norm(g - self.d.xpos[self.pel][:2]) > 1.0:
                return g
        return np.array([0.0, -1.6])

    def _obs(self):
        p, yaw = self.d.xpos[self.pel], self._yaw()
        dxy = self.goal - p[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        local = np.array([c * dxy[0] - s * dxy[1], s * dxy[0] + c * dxy[1]])
        rays = []
        geomid = np.zeros(1, np.int32)
        for a in RAY_ANGLES:
            v = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(self.m, self.d, np.array([p[0], p[1], 0.3]), v, self.ray_group, 1, -1, geomid)
            rays.append(2.0 if dist < 0 else min(dist, 2.0))
        return np.concatenate([local / 3, [np.linalg.norm(dxy) / 4], np.array(rays) / 2, [self.d.qvel[self.g1.fv + 5] / 2], self.last_action]).astype(np.float32)

    # --- gym API -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed0 if seed is None and self._np_random is None else seed)
        mujoco.mj_resetData(self.m, self.d)
        self.g1.reset()
        yaw = self.np_random.uniform(-np.pi, np.pi)  # random starting heading
        self.d.qpos[self.g1.fq + 3:self.g1.fq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.goal = self._sample_goal()
        self.steps, self.last_action = 0, np.zeros(2, np.float32)
        self.prev = np.linalg.norm(self.goal - self.d.xpos[self.pel][:2])
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1).astype(np.float32)
        self.g1.cmd = np.array([0.3 + 0.3 * a[0], 0.0, 0.7 * a[1]])
        bump = False
        for k in range(100):  # 0.2 s at 500 Hz; the Unitree policy runs every 10 steps
            self.g1.step()
            if k % 20 == 19:
                bump |= self._bumping()
        self.steps += 1
        self.last_action = a
        dist = np.linalg.norm(self.goal - self.d.xpos[self.pel][:2])
        reward = 2.0 * (self.prev - dist) - 0.005 - 0.3 * bump
        self.prev = dist
        fell, reached = self.g1.height() < 0.5, dist < 0.35
        if reached:
            reward += 5.0
        if fell:
            reward -= 5.0
        return self._obs(), float(reward), bool(reached or fell), self.steps >= self.max_steps, {"reached": reached, "fell": fell, "bump": bump}


def evaluate(policy_fn, episodes=30, seed=123):
    env, wins, falls, steps, bumps = G1RoomNav(seed=seed), 0, 0, [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            obs, _, term, trunc, info = env.step(policy_fn(obs))
            bumps += info["bump"]
            if term or trunc:
                wins += info["reached"]; falls += info["fell"]; steps.append(env.steps)
                break
    return wins / episodes, falls / episodes, float(np.mean(steps)) * 0.2, bumps * 0.2 / episodes


def head_for_goal(obs):
    """Hand-written baseline: turn toward the goal, walk when roughly facing it (ignores furniture)."""
    ang = np.arctan2(obs[1], obs[0])
    return np.array([1.0 if abs(ang) < 0.6 else -1.0, np.clip(2 * ang, -1, 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, metavar="STEPS", help="train PPO for this many environment steps")
    ap.add_argument("--envs", type=int, default=4, help="parallel environments (CPU processes)")
    ap.add_argument("--eval", action="store_true", help="compare random, hand-written and trained policies")
    ap.add_argument("--view", action="store_true", help="watch the trained policy in the MuJoCo viewer")
    args = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if args.train:
        venv = make_vec_env(G1RoomNav, n_envs=args.envs, vec_env_cls=SubprocVecEnv if args.envs > 1 else None, env_kwargs={})
        model = PPO.load(MODEL, env=venv) if os.path.exists(MODEL) else PPO(
            "MlpPolicy", venv, n_steps=256, batch_size=256, learning_rate=3e-4, gamma=0.98, ent_coef=0.003, verbose=0,
            policy_kwargs={"net_arch": [64, 64]})
        t0, chunk = time.time(), max(2048, args.train // 10)
        done = 0
        while done < args.train:
            model.learn(chunk, reset_num_timesteps=False)
            done += chunk
            ep = [e["r"] for e in model.ep_info_buffer]
            print(f"{done:>7} steps · {time.time() - t0:5.0f} s · mean episode reward {np.mean(ep) if ep else float('nan'):6.2f}", flush=True)
            model.save(MODEL)
        venv.close()

    if args.eval:
        rng = np.random.default_rng(0)
        for name, fn in [("random commands", lambda o: rng.uniform(-1, 1, 2)), ("head straight for goal", head_for_goal)]:
            print(f"{name:24s} success {{:.0%}} · falls {{:.0%}} · avg time {{:.1f}} s · touching furniture {{:.1f}} s/episode".format(*evaluate(fn)))
        if os.path.exists(MODEL):
            model = PPO.load(MODEL)
            print("{:24s} success {:.0%} · falls {:.0%} · avg time {:.1f} s · touching furniture {:.1f} s/episode".format("trained PPO", *evaluate(lambda o: model.predict(o, deterministic=True)[0])))

    if args.view:
        from mujoco import viewer as mjviewer
        model, env = PPO.load(MODEL), G1RoomNav()
        obs, _ = env.reset()
        with mjviewer.launch_passive(env.m, env.d) as v:
            while v.is_running():
                t = time.time()
                obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
                v.user_scn.ngeom = 1  # green goal marker
                mujoco.mjv_initGeom(v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_CYLINDER, [0.25, 0.25, 0.01], [*env.goal, 0.01], np.eye(3).ravel(), [0.3, 1, 0.5, 0.8])
                v.sync()
                if term or trunc:
                    obs, _ = env.reset()
                time.sleep(max(0.0, 0.2 - (time.time() - t)))


if __name__ == "__main__":
    main()
