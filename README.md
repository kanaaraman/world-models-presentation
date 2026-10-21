# World Models, Explored

An interactive, scroll-driven presentation about AI **world models**. It covers what they are, why we need them, the state of the art as of late 2026, use cases, robotics (MuJoCo, Isaac Lab, sim-to-real), the software landscape and open-source tools.

## ▶ Open the presentation

**Live, no install:** https://kanaaraman.github.io/world-models-presentation/

That's the easiest way to run it. It works in any modern desktop or mobile browser (Chrome, Edge, Firefox, Safari).

## How to run it locally

### 1. Get the code

```bash
git clone https://github.com/kanaaraman/world-models-presentation.git
cd world-models-presentation
```

Or download the ZIP from GitHub (**Code → Download ZIP**) and unzip it.

### 2. Serve the folder

Most of the deck works if you just double-click `index.html`. The **MuJoCo** and **furnished-room** scenes (WebAssembly) and the embedded **Ridge Q1** and **Isaac Lab** demos need the page to be served over `http://`, so a local web server is recommended. Pick whichever tool you already have:

| You have | Command | Then open |
|---|---|---|
| Python 3 | `python -m http.server 8000` | http://localhost:8000 |
| uv | `uv run python -m http.server 8000` | http://localhost:8000 |
| Node.js | `npx http-server -p 8000` | http://localhost:8000 |
| VS Code | Install **Live Server**, right-click `index.html` → *Open with Live Server* | (opens automatically) |

Run the command from inside the `world-models-presentation` folder.

### 3. Internet connection

The deck itself is a single self-contained `index.html` with no build step. A few parts load from the internet:

- Google Fonts (falls back to system fonts offline)
- MuJoCo 3.14 WebAssembly from `cdn.jsdelivr.net` (about 9 MB, loaded when you reach a MuJoCo scene)
- three.js 0.160 from `cdn.jsdelivr.net` (for the furnished-room 3D view)
- Isaac Lab's official interactive demos from `isaac-sim.github.io` (loaded when you click *Launch*)
- Cosmos and NuRec example GIFs from NVIDIA's download server

Everything else, including all the other demos, runs offline.

## Presenting

| Action | Key / gesture |
|---|---|
| Next / previous scene | <kbd>↓</kbd> <kbd>↑</kbd>, <kbd>Space</kbd> / <kbd>Shift</kbd>+<kbd>Space</kbd>, <kbd>PgDn</kbd> <kbd>PgUp</kbd>, or scroll |
| First / last scene | <kbd>Home</kbd> / <kbd>End</kbd> |
| Jump to a scene | Navigation dots on the right (desktop) |
| Dream mode (background) | <kbd>D</kbd> |
| Full screen | <kbd>F11</kbd> (Windows/Linux) or <kbd>Ctrl</kbd>+<kbd>Cmd</kbd>+<kbd>F</kbd> (macOS) |

Tips:
- The counter in the bottom-left shows the current scene (for example `15 / 21`).
- Heavier demos (MuJoCo, the furnished room, Isaac Lab) start only when you reach or launch them. Give them a few seconds on first load.
- For a talk, open the live link a minute before you start and click through once, so the browser caches the WebAssembly files.

## What's inside

| # | Scene | Interactive element |
|---|-------|---------------------|
| 1 | Intro | Morphing 3D particle field that reacts to your cursor |
| 2 | What is a world model | Animated perceive → predict → plan loop, with an awake/dream toggle |
| 3 | Live demo | An agent with a ballistics world model learns a hidden wind and plans its shots |
| 4 | Why we need them | LLM vs world model comparison and six motivations |
| 5 | Five schools | Generative, JEPA, model-based RL, 3D/spatial and simulators, each with a live visual |
| 6 | History | Draggable, filterable timeline from 1943 to 2026 |
| 7 | Current state | Animated KPIs and maturity by domain |
| 8 | Use cases | Orbit explorer across 10 industries |
| 9 | Robotics case: Ridge Q1 | Embedded live quadruped sim ([`examples/ridge-q1.html`](examples/ridge-q1.html)) plus a checklist verdict: a map-based planner with verbal memory, **not** a world model |
| 10 | Ridge W | The same robot with a learned world model: it imagines 16 futures per route, learns slip, speed and payload limits, declines tasks it predicts will fail, and is compared with classic A* |
| 11 | MuJoCo demo | **Real MuJoCo 3.14 (WebAssembly)** as "reality": the agent explores, fits a world model Δs = f(s, a), and swings up a cart-pole with MPPI planning entirely in its imagination |
| 12 | Isaac Lab | Isaac Lab 3.0's **official browser demos** embedded live (ANYmal-D, G1, Franka lift/drawer/cloth, cartpole, dominoes, MPM), plus throughput bars from Isaac Lab's published benchmarks (up to ~1M env-steps/s) |
| 13 | Isaac Lab × world models | Three pipelines: Isaac Lab as a data factory for a learned world model, Isaac Lab Mimic → Cosmos Transfer, and model-free RL as the baseline |
| 14 | Sim-to-real | Train → check in sim → export (LEAPP .onnx) → deploy (Isaac ROS / LeRobot) → test on hardware, plus a **consistency lab**: break input, response or output consistency and watch a "real" arm diverge from the simulated one |
| 15 | Robot in a furnished room | **Real MuJoCo physics + three.js 3D**: drive a robot (WASD / arrows / on-screen pad) through a living room with a table, chairs, sofa, bookshelf, lamp, boxes and ball. Twice a second it copies the world and imagines 1.2 s ahead (violet ghosts), then scores its prediction. Toggle an *imperfect world model* to see predictions drift. Model file: [`examples/furnished_room.xml`](examples/furnished_room.xml) |
| 16 | Real room → sim | Scan a real space with NVIDIA NuRec, train in its digital twin with Isaac Lab + COMPASS, deploy in the same room. NVIDIA's living-room sample scans, plus an Isaac Lab furniture-scene config sketch |
| 17 | Software map | Searchable, filterable catalog of 38 tools (open source, open weights, closed, simulators) |
| 18 | Get started | Six hands-on paths with copyable commands |
| 19 | Challenges | Compounding-error simulator plus open problems |
| 20–21 | Future and close | Trends and references |

## Python examples

### Furnished room in the MuJoCo viewer

```bash
cd examples
uv run --with mujoco python -m mujoco.viewer --mjcf=furnished_room.xml
```

The robot's actuators are `fx`, `fy` (forces) and `tz` (yaw torque). Use the viewer's control sliders to push it into the furniture.

### MuJoCo: learn a world model and plan with it (CPU, about 1 minute)

[`examples/mujoco_world_model.py`](examples/mujoco_world_model.py) runs the same pipeline as scene 11: explore with random actions, fit a random-Fourier-feature world model, then plan with MPPI using only the learned model. With [uv](https://docs.astral.sh/uv/) installed, no other setup is needed:

```bash
cd examples
uv run --with mujoco --with numpy python mujoco_world_model.py          # prints metrics
uv run --with mujoco --with numpy python mujoco_world_model.py --view   # live MuJoCo 3D viewer
```

Without uv: `pip install mujoco numpy`, then `python mujoco_world_model.py`.

Typical output: one-step θ̇ error ≈ 0.014 rad/s, swing-up after ~1.8 s, 100% upright afterwards, ~8 ms per plan. Options: `--episodes`, `--steps`, `--sigma`, `--seed`.

### Isaac Lab: collect data at GPU scale (NVIDIA GPU)

[`examples/isaaclab_world_model.py`](examples/isaaclab_world_model.py) steps thousands of parallel Isaac Lab environments with exploratory actions, saves `(obs, action, next_obs)` transitions, trains an MLP world model, and reports 1-step and 10-step open-loop error against the simulator. [Install Isaac Lab 3.0](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then run from the Isaac Lab root:

```bash
uv run python /path/to/world-models-presentation/examples/isaaclab_world_model.py \
  --task Isaac-Cartpole --num_envs 4096 --steps 300
```

It's written against the Isaac Lab 3.0 API (the same calls as Isaac Lab's own random agent) but hasn't been run here, because it needs an NVIDIA GPU.

## Project structure

```
index.html                        the whole presentation (HTML + CSS + JS, no build)
examples/ridge-q1.html            Ridge Q1 quadruped simulation (embedded in scene 9)
examples/furnished_room.xml       MuJoCo living room + mobile robot (scene 15)
examples/mujoco_world_model.py    MuJoCo world-model + MPPI example
examples/isaaclab_world_model.py  Isaac Lab data-collection + world-model example
```

## Troubleshooting

- **MuJoCo or furnished-room scene says it couldn't load:** you opened the file directly (`file://`) or you're offline. Serve the folder (step 2) and check your connection.
- **Isaac Lab demo stays blank:** it's downloading physics assets, so wait a few seconds. Or use the *open ↗* button to view it in its own tab.
- **Driving doesn't respond:** click inside the room first so it has keyboard focus. Arrow keys then drive the robot instead of changing scenes. Click outside the room to get scene navigation back.
- **Animations look slow:** background tabs and hidden windows are throttled by the browser. Keep the presentation tab in front.

Facts reflect public information as of October 2026. Check the linked sources for the latest details.
