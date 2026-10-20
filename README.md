# World Models, Explored

An interactive, scroll-driven presentation about AI **world models**. It covers what they are, why we need them, the state of the art as of late 2026, use cases, the software landscape and open-source tools.

**Live:** https://kanaaraman.github.io/world-models-presentation/

## What's inside

| # | Scene | Interactive element |
|---|-------|---------------------|
| 01 | Intro | Morphing 3D particle field that reacts to your cursor |
| 02 | What is a world model | Animated perceive → predict → plan loop, with an awake/dream toggle |
| 03 | Live demo | Agent with a ballistics world model learns a hidden wind and plans shots |
| 04 | Why we need them | LLM vs world model comparison and six motivations |
| 05 | Five schools | Generative, JEPA, model-based RL, 3D/spatial and simulators, each with a live visual |
| 06 | History | Draggable, filterable timeline from 1943 to 2026 |
| 07 | Current state | Animated KPIs and maturity by domain |
| 08 | Use cases | Orbit explorer across 10 industries |
| 08b | Robotics case: Ridge Q1 | Embedded live quadruped sim ([`examples/ridge-q1.html`](examples/ridge-q1.html)) plus a checklist verdict: a map-based planner with verbal memory, **not** a world model |
| 08c | Ridge W | The same robot with a learned world model: it imagines 16 futures per route, learns slip, speed and payload limits from experience, and declines tasks it predicts will fail. Compare it with classic A* |
| 08d | MuJoCo demo | **Real MuJoCo 3.14 (WebAssembly)** is the "reality". The agent explores, fits a world model Δs = f(s, a), and swings up a cart-pole with MPPI planning entirely in its imagination |
| 09 | Software map | Searchable, filterable catalog of 37 tools (open/closed/simulator) |
| 10 | Get started | Six hands-on paths with copyable commands |
| 11 | Challenges | Compounding-error simulator plus open problems |
| 12–13 | Future and close | Trends and references |

## Controls

- Scroll, or use <kbd>↑</kbd> <kbd>↓</kbd> / <kbd>Space</kbd> / <kbd>PgUp</kbd> <kbd>PgDn</kbd> to move between scenes
- <kbd>D</kbd> toggles "dream mode" for the background
- Navigation dots on the right (desktop)

## MuJoCo example (Python)

[`examples/mujoco_world_model.py`](examples/mujoco_world_model.py) runs the same pipeline as the browser demo: explore with random actions, fit a random-Fourier-feature world model, then plan with MPPI using only the learned model.

```bash
cd examples
uv run --with mujoco --with numpy python mujoco_world_model.py          # prints metrics
uv run --with mujoco --with numpy python mujoco_world_model.py --view   # live MuJoCo viewer
```

Typical output: one-step θ̇ error ≈ 0.014 rad/s, swing-up after ~1.8 s, 100% upright afterwards, ~8 ms per plan.

## Run locally

It's a single self-contained `index.html` with no build step and no dependencies apart from Google Fonts. Open it in a browser, or serve the folder with any static server.

Facts reflect public information as of October 2026. Check the linked sources for the latest details.
