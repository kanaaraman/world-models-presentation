"""Learn a world model from MuJoCo, then plan inside it.

MuJoCo plays the role of reality. The agent never reads MuJoCo's equations:
it only sees (state, action, next state) samples. From those it fits a world
model  s' = s + f(s, a)  and uses it to imagine futures with MPPI, applying
only the first action of the best imagined plan at each step.

    pipeline:  explore (random actions)  ->  fit world model  ->  imagine & control
                                                  ^                     |
                                                  +---- new data -------+

Run:
    uv run --with mujoco --with numpy python mujoco_world_model.py
    uv run --with mujoco --with numpy python mujoco_world_model.py --view   # live 3D viewer

This mirrors the in-browser demo in index.html (scene "MuJoCo demo").
"""

import argparse
import time

import mujoco
import numpy as np

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01" integrator="RK4"/>
  <visual><global offwidth="960" offheight="540"/></visual>
  <worldbody>
    <light pos="0 -3 4" dir="0 1 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.2" size="4 2 .1" rgba=".15 .17 .22 1"/>
    <geom name="rail" type="capsule" fromto="-2.6 0 1 2.6 0 1" size=".02" rgba=".5 .5 .6 1"
          contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.6 2.6" damping="0.0005"/>
      <geom type="box" size=".2 .15 .1" mass="1" rgba=".9 .9 .95 1" contype="0" conaffinity="0"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.0002"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size=".045" mass=".1" rgba="1 .7 .28 1"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/></actuator>
</mujoco>
"""

SUBSTEPS = 4  # 4 x 10 ms physics steps per 40 ms control step


class Reality:
    """Thin wrapper around MuJoCo. The agent only uses state(), step() and reset_*()."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.data = mujoco.MjData(self.model)

    def state(self):
        d = self.data
        return np.array([d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1]])

    def step(self, u):
        self.data.ctrl[0] = u
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def reset_hanging(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = np.pi
        mujoco.mj_forward(self.model, self.data)

    def reset_random(self, rng):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi)]
        self.data.qvel[:] = [rng.normal(), 3 * rng.normal()]
        mujoco.mj_forward(self.model, self.data)


class WorldModel:
    """s' = s + f(s, a), with f = random Fourier features + ridge regression.

    That is a one-hidden-layer network whose hidden layer is random and fixed and
    whose output layer is solved in closed form. It is small, fast and trains in
    milliseconds, which makes it handy for teaching. Swap in an MLP or an
    ensemble (as in PETS) for harder tasks.
    """

    IN_SCALE = np.array([2.0, 1.0, 1.0, 4.0, 8.0, 1.0])
    OUT_SCALE = np.array([0.2, 0.3, 1.0, 2.0])

    def __init__(self, n_features=200, lengthscale=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(6, n_features)) / lengthscale
        self.b = rng.uniform(0, 2 * np.pi, size=n_features)
        self.k = np.sqrt(2.0 / n_features)
        f = n_features + 6 + 1
        self.A = np.zeros((f, f))
        self.B = np.zeros((f, 4))
        self.coef = np.zeros((f, 4))
        self.n = 0

    def _features(self, s, u):
        s = np.atleast_2d(s)
        u = np.atleast_1d(u)
        z = np.stack([s[:, 0], np.cos(s[:, 1]), np.sin(s[:, 1]), s[:, 2], s[:, 3], u], axis=1)
        z = z / self.IN_SCALE
        rff = self.k * np.cos(z @ self.W + self.b)
        return np.hstack([rff, z, np.ones((len(z), 1))])

    def add(self, s, u, s_next):
        phi = self._features(s, u)
        y = (np.atleast_2d(s_next) - np.atleast_2d(s)) / self.OUT_SCALE
        self.A += phi.T @ phi
        self.B += phi.T @ y
        self.n += len(phi)

    def fit(self, ridge=1e-3):
        lam = ridge * max(1.0, self.n * 1e-3)
        self.coef = np.linalg.solve(self.A + lam * np.eye(len(self.A)), self.B)

    def predict(self, s, u):
        return np.atleast_2d(s) + (self._features(s, u) @ self.coef) * self.OUT_SCALE


class MPPI:
    """Model-predictive path integral control that only queries the world model."""

    def __init__(self, model, horizon=40, samples=64, sigma=0.5, temperature=30.0, seed=0):
        self.model, self.H, self.K, self.sigma, self.temp = model, horizon, samples, sigma, temperature
        self.U = np.zeros(horizon)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def cost(s, u):
        x, th, _, thdot = s.T
        return -np.cos(th) + 0.25 * x**2 + 0.002 * thdot**2 + 0.01 * u**2 + 5.0 * (np.abs(x) > 2.2)

    def act(self, s0):
        U = np.clip(self.U + self.sigma * self.rng.normal(size=(self.K, self.H)), -1, 1)
        eps = U - self.U
        s = np.repeat(s0[None], self.K, axis=0)
        total = np.zeros(self.K)
        for h in range(self.H):  # imagined rollouts, all K at once
            s = self.model.predict(s, U[:, h])
            total += self.cost(s, U[:, h])
        lam = max(total.max() - total.min(), 1e-6) / self.temp
        w = np.exp(-(total - total.min()) / lam)
        w /= w.sum()
        self.U = np.clip(self.U + w @ eps, -1, 1)
        a = self.U[0]
        self.U = np.roll(self.U, -1)
        self.U[-1] = 0.0
        return a


def explore(env, model, episodes, rng):
    u = 0.0
    for _ in range(episodes):
        env.reset_random(rng)
        for _ in range(100):
            u = float(np.clip(0.7 * u + 0.6 * rng.normal(), -1, 1))
            s = env.state()
            s2 = env.step(u)
            if abs(s2[0]) > 2.4:
                break
            model.add(s, u, s2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=30, help="random exploration episodes (100 steps each)")
    p.add_argument("--steps", type=int, default=750, help="control steps (40 ms each)")
    p.add_argument("--sigma", type=float, default=0.5, help="MPPI exploration noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="store_true", help="open the MuJoCo viewer while controlling")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env, model = Reality(), WorldModel(seed=args.seed)

    t = time.perf_counter()
    explore(env, model, args.episodes, rng)
    model.fit()
    print(f"1. explored + 2. fitted: {model.n} transitions in {time.perf_counter() - t:.2f} s")

    # held-out check: one-step error on fresh random states
    errs = []
    for _ in range(300):
        env.reset_random(rng)
        s, u = env.state(), rng.uniform(-1, 1)
        errs.append(model.predict(s, u)[0] - env.step(u))
    rmse = np.sqrt(np.mean(np.square(errs), axis=0))
    print(f"   one-step RMSE  x={rmse[0]:.4f} m  theta={rmse[1]:.4f} rad  "
          f"xdot={rmse[2]:.4f} m/s  thetadot={rmse[3]:.4f} rad/s")

    planner = MPPI(model, sigma=args.sigma, seed=args.seed)
    env.reset_hanging()
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    upright, first_up, plan_ms = [], None, []
    for k in range(args.steps):
        s = env.state()
        t = time.perf_counter()
        a = planner.act(s)
        plan_ms.append((time.perf_counter() - t) * 1000)
        s2 = env.step(a)
        model.add(s, a, s2)  # keep learning online
        if k % 100 == 99:
            model.fit()
        up = np.cos(s2[1]) > 0.9
        upright.append(up)
        if up and first_up is None:
            first_up = k
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, SUBSTEPS * env.model.opt.timestep - plan_ms[-1] / 1000))

    tail = np.mean(upright[-400:])
    print(f"3. imagined & controlled: first upright after {first_up if first_up is not None else '-'} steps "
          f"({(first_up or 0) * SUBSTEPS * env.model.opt.timestep:.1f} s), "
          f"upright {tail:.0%} of the last {min(400, len(upright))} steps, "
          f"{np.mean(plan_ms):.1f} ms per plan, {model.n} transitions learned")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
