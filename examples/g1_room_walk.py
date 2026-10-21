"""Walk a Unitree G1 humanoid through the furnished room in MuJoCo.

The same scene as the deck's "humanoid in a real-looking room": Unitree's
pretrained walking policy (LSTM, 50 Hz) drives the G1's 12 leg joints with PD
control, and you steer it with velocity commands.

    uv run --with mujoco --with numpy python g1_room_walk.py --view    # interactive
    uv run --with mujoco --with numpy python g1_room_walk.py           # scripted walk, prints a summary

Viewer keys: Up = walk, Down = back up, Left/Right = turn, Space = stop.

Robot model and policy: unitree_rl_gym (BSD-3-Clause, (c) Unitree Robotics),
see g1/LICENSE. The policy weights are stored as JSON so no PyTorch is needed.
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class WalkPolicy:
    """LSTM(47 -> 64) + MLP(64 -> 32 -> 12), a numpy port of Unitree's motion.pt."""

    def __init__(self, path):
        p = json.load(open(path))
        self.cfg = p["config"]
        self.w = {k: np.array(v, dtype=np.float32).reshape(p["shapes"][k]) for k, v in p.items() if k not in ("shapes", "config")}
        self.reset()

    def reset(self):
        self.h = np.zeros(64, np.float32)
        self.c = np.zeros(64, np.float32)

    def __call__(self, x):
        w = self.w
        g = w["memory.weight_ih_l0"] @ x + w["memory.bias_ih_l0"] + w["memory.weight_hh_l0"] @ self.h + w["memory.bias_hh_l0"]
        i, f, gg, o = np.split(g, 4)
        sig = lambda z: 1 / (1 + np.exp(-z))
        self.c = sig(f) * self.c + sig(i) * np.tanh(gg)
        self.h = sig(o) * np.tanh(self.c)
        z = w["actor.0.weight"] @ self.h + w["actor.0.bias"]
        z = np.where(z > 0, z, np.exp(z) - 1)  # ELU
        return w["actor.2.weight"] @ z + w["actor.2.bias"]


class G1:
    def __init__(self, m, d, policy):
        self.m, self.d, self.pi, c = m, d, policy, policy.cfg
        free = m.body("pelvis").jntadr[0]
        self.fq, self.fv = m.jnt_qposadr[free], m.jnt_dofadr[free]
        hip = m.joint("left_hip_pitch_joint").id
        self.jq, self.jv = m.jnt_qposadr[hip], m.jnt_dofadr[hip]
        self.kp, self.kd, self.q0 = np.array(c["kps"]), np.array(c["kds"]), np.array(c["default_angles"])
        self.cmd = np.zeros(3)
        self.reset()

    def reset(self):
        d = self.d
        d.qpos[self.fq:self.fq + 7] = [-2, -1.6, 0.793, 1, 0, 0, 0]
        d.qpos[self.jq:self.jq + 12] = self.q0
        d.qvel[self.fv:self.fv + 6] = 0
        d.qvel[self.jv:self.jv + 12] = 0
        mujoco.mj_forward(self.m, d)
        self.pi.reset()
        self.act = np.zeros(12, np.float32)
        self.tgt = self.q0.copy()
        self.n = 0

    def step(self):
        d, c = self.d, self.pi.cfg
        d.ctrl[:] = (self.tgt - d.qpos[self.jq:self.jq + 12]) * self.kp - d.qvel[self.jv:self.jv + 12] * self.kd
        mujoco.mj_step(self.m, d)
        self.n += 1
        if self.n % c["decimation"]:
            return
        w, x, y, z = d.qpos[self.fq + 3:self.fq + 7]
        phase = (self.n * c["sim_dt"]) % c["period"] / c["period"]
        obs = np.concatenate([
            d.qvel[self.fv + 3:self.fv + 6] * c["ang_vel_scale"],
            [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)],  # gravity in body frame
            self.cmd * c["cmd_scale"],
            (d.qpos[self.jq:self.jq + 12] - self.q0) * c["dof_pos_scale"],
            d.qvel[self.jv:self.jv + 12] * c["dof_vel_scale"],
            self.act,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]).astype(np.float32)
        self.act = self.pi(obs)
        self.tgt = self.act * c["action_scale"] + self.q0

    def height(self):
        return self.d.xpos[self.m.body("pelvis").id][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "g1_room.xml"))
    d = mujoco.MjData(m)
    g1 = G1(m, d, WalkPolicy(os.path.join(HERE, "g1", "g1_walk_policy.json")))
    furniture = [b for b in range(1, m.nbody) if m.body_rootid[b] != m.body("pelvis").id]
    start = d.xpos.copy()

    if args.view:
        from mujoco import viewer as mjviewer
        keys = {265: [0.6, 0, 0], 264: [-0.3, 0, 0], 263: [0.2, 0, 0.7], 262: [0.2, 0, -0.7], 32: [0, 0, 0]}  # GLFW codes
        def on_key(k):
            if k in keys:
                g1.cmd = np.array(keys[k], dtype=float)
        with mjviewer.launch_passive(m, d, key_callback=on_key) as v:
            while v.is_running():
                t = time.time()
                for _ in range(10):
                    g1.step()
                if g1.height() < 0.45:
                    g1.reset()
                v.sync()
                time.sleep(max(0.0, 10 * m.opt.timestep - (time.time() - t)))
        return

    plan = [("stand", [0, 0, 0], 2), ("walk", [0.5, 0, 0], 4), ("turn left", [0.2, 0, 0.6], 2), ("walk", [0.6, 0, 0], 3), ("stop", [0, 0, 0], 2)]
    t0 = time.perf_counter()
    for name, cmd, secs in plan:
        g1.cmd = np.array(cmd, dtype=float)
        zmin = min((g1.step(), g1.height())[1] for _ in range(int(secs / m.opt.timestep)))
        xy = d.xpos[m.body("pelvis").id][:2]
        print(f"{name:10s} cmd={cmd}  lowest pelvis height {zmin:.2f} m  now at ({xy[0]:.2f}, {xy[1]:.2f})")
    moved = [m.body(b).name for b in furniture if np.linalg.norm(d.xpos[b] - start[b]) > 0.05]
    print(f"simulated {d.time:.1f} s in {time.perf_counter() - t0:.1f} s; furniture moved: {moved or 'none'}")


if __name__ == "__main__":
    main()
