"""Use Isaac Lab as a data factory for a learned world model.

Isaac Lab is an engineered world model: exact, steppable GPU physics. This
script uses it the way world-model research does. It steps thousands of
parallel environments with exploratory actions, records (obs, action, next_obs)
transitions, and trains a neural world model  obs' = obs + f(obs, action)
that you could then plan or "dream" with (MPPI, Dreamer, TD-MPC...).

It's the GPU-scale version of mujoco_world_model.py (scene 10 of the deck).

Requirements: an Isaac Lab 3.0 checkout and an NVIDIA GPU. Run from the Isaac Lab root:

    uv run python /path/to/examples/isaaclab_world_model.py --task Isaac-Cartpole --num_envs 4096 --steps 300

This is written against the Isaac Lab 3.0 API, using the same calls as
isaaclab_rl/entrypoints/simple_agents.py (the random agent). It hasn't been
run on a GPU by the author of this deck, so treat it as a starting point.
"""

# Warp reads this at import time; Isaac Lab doesn't need Warp autodiff here.
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Collect Isaac Lab transitions and fit a world model.")
    p.add_argument("--task", default="Isaac-Cartpole")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=300, help="environment steps per env")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="isaaclab_transitions.pt")
    add_launcher_args(p)
    p.set_defaults(device=None)  # let the task config pick the simulation device
    args, hydra_args = setup_preset_cli(p, sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def policy_obs(obs):
    return obs["policy"] if isinstance(obs, dict) else obs


class WorldModel(torch.nn.Module):
    """Predicts the change in observation; inputs and outputs are normalized."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, obs_dim),
        )
        self.register_buffer("mu", torch.zeros(obs_dim + act_dim))
        self.register_buffer("sd", torch.ones(obs_dim + act_dim))
        self.register_buffer("dmu", torch.zeros(obs_dim))
        self.register_buffer("dsd", torch.ones(obs_dim))

    def set_stats(self, x, d):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.dmu, self.dsd = d.mean(0), d.std(0) + 1e-6

    def forward(self, obs, act):
        x = (torch.cat([obs, act], -1) - self.mu) / self.sd
        return obs + self.net(x) * self.dsd + self.dmu


def collect(env, steps):
    """Smoothed random actions; drop transitions that cross an auto-reset."""
    u = env.unwrapped
    obs = policy_obs(env.reset()[0])
    act_shape = env.action_space.shape
    a = torch.zeros(act_shape, device=u.device)
    O, A, O2 = [], [], []
    for _ in range(steps):
        a = (0.8 * a + 0.6 * torch.randn(act_shape, device=u.device)).clamp(-1, 1)
        with torch.no_grad():  # plain tensors, so they can feed training later
            nxt, _, term, trunc, _ = env.step(a)
        nxt = policy_obs(nxt)
        keep = ~(term | trunc)  # after a reset the next obs isn't the true successor
        O.append(obs[keep].clone()); A.append(a[keep].clone()); O2.append(nxt[keep].clone())
        obs = nxt
    return torch.cat(O), torch.cat(A), torch.cat(O2)


def train(O, A, O2, epochs, device):
    n = len(O); perm = torch.randperm(n, device=device); split = int(0.9 * n)
    tr, va = perm[:split], perm[split:]
    wm = WorldModel(O.shape[1], A.shape[1]).to(device)
    wm.set_stats(torch.cat([O[tr], A[tr]], -1), (O2 - O)[tr])
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    for ep in range(epochs):
        for idx in tr[torch.randperm(len(tr), device=device)].split(4096):
            loss = (((wm(O[idx], A[idx]) - O2[idx]) / wm.dsd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val = (((wm(O[va], A[va]) - O2[va]) / wm.dsd) ** 2).mean().item()
            copy = (((O[va] - O2[va]) / wm.dsd) ** 2).mean().item()  # "nothing changes" baseline
        print(f"epoch {ep + 1:2d}  train {loss.item():.4f}  val {val:.4f}  (copy-last baseline {copy:.4f})")
    return wm


def open_loop_error(env, wm, horizon=10):
    """Roll the learned model forward for `horizon` steps next to the simulator."""
    u = env.unwrapped
    obs = policy_obs(env.reset()[0]); pred = obs.clone()
    errs = []
    for _ in range(horizon):
        a = (2 * torch.rand(env.action_space.shape, device=u.device) - 1)
        with torch.no_grad():  # plain tensors, so they can feed training later
            obs, _, term, trunc, _ = env.step(a)
        obs = policy_obs(obs)
        with torch.no_grad():
            pred = wm(pred, a)
        alive = ~(term | trunc)
        errs.append(((pred - obs)[alive] / wm.dsd).pow(2).mean().sqrt().item())
    return errs


def main():
    args = parse_args()
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        device = env.unwrapped.device
        t = time.perf_counter()
        O, A, O2 = collect(env, args.steps)
        dt = time.perf_counter() - t
        print(f"collected {len(O):,} transitions from {args.num_envs} envs in {dt:.1f} s "
              f"({len(O) / dt:,.0f} transitions/s)")
        torch.save({"obs": O.cpu(), "act": A.cpu(), "next_obs": O2.cpu(), "task": args.task}, args.out)
        wm = train(O, A, O2, args.epochs, device)
        errs = open_loop_error(env, wm)
        print("open-loop RMSE (normalized) by horizon:", " ".join(f"{e:.3f}" for e in errs))
        print("Watch it grow with the horizon: that's the compounding error from scene 15.")
        env.close()


if __name__ == "__main__":
    main()
