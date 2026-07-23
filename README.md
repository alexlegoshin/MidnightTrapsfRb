# MidnightTrapsfRb

A real-time desktop **optical-trap "cell" simulator** for rubidium-87. It models
a magneto-optical trap (MOT) and its transfer into an optical dipole trap and
renders the atoms live, as if you were watching the cell through an IR
fluorescence camera. Switch the cooling beams, repumper and dipole trap on and
off and watch the cloud cool, glow, go dark or fall in real time.

*Midnight* (built during many late nights, à la Midnight Club); *Trapsf* =
**Trap** + **Transfer** (and it conveniently ends in **Rb**).

The physics comes from the [MOTorNOT](https://github.com/alexlegoshin/MOTorNOT)
library; this repository is the application on top of it.

## Install & run

```bash
# 1. install the physics engine (separate repo)
pip install -e ../MOTorNOT          # or: pip install git+https://github.com/alexlegoshin/MOTorNOT
#    optional GPU: pip install "cupy-cuda12x[ctk]"

# 2. install the app's own dependencies
pip install -r requirements.txt

# 3. run
python run.py                        # or:  python -m midnightrb
```

## What you can do

- **Configuration tab** — set the MOT (power, detuning, beam radius, B-gradient),
  the dipole trap (wavelength, power, waist, lattice on/off) and the initial
  cloud (atom number, temperature, size), then *Apply & reset cloud*.
- **Controls tab** — toggle the cooling beams, repumper and dipole trap; hit
  **Recapture into dipole** to cut the MOT and hold the dipole in one click;
  set the simulation **speed** and the fast/full model switch.
- **Camera tab** — field of view, optical blur, exposure, colormap and viewing
  axis of the simulated camera.

Try: cool with beams + repumper → turn the repumper off and watch the cloud pump
dark and fall → or hit *Recapture* to catch the coldest atoms in the dipole trap.

## How it stays real-time

The full 6-beam MOT force is accurate but expensive. For interactivity the app
runs a **linearised model** near the trap centre (a spring + damping fit,
`F ≈ F₀ − κ·(x − c) − β·v`), which is ~50× faster and, for the trapped cloud,
matches the full force to ~0.1%.

Those coefficients are not frozen. A background **CoefficientEstimator** keeps
its own full 6-beam model and continuously re-fits `κ, β, F₀` around the *live*
cloud centroid, also reporting the linear-vs-full **model error** shown in the
status line. Every coefficient set is tagged with a config generation, so
changing a MOT parameter can never apply stale coefficients — the fast model
self-corrects while staying cheap.

Three threads cooperate through short-held locks: a **physics** thread (owns the
simulation, advances it and publishes snapshots), the **estimator** thread, and
the **GUI** thread (renders the camera and handles input) — so the window stays
responsive no matter how heavy the physics is. A **speed** control scales
sim-time per frame for a clear, watchable evolution.

Photon-recoil heating and gravity are included so the MOT settles at a realistic
finite size (~Doppler-limit temperature) and an untrapped cloud falls.

## Layout

```
midnightrb/
  config.py    dataclasses: MOT / dipole / cloud / camera / sim settings
  engine.py    RealTimeSimulation + CoefficientEstimator (physics controller)
  camera.py    Camera: atom positions -> simulated IR frame
  app.py       Dear PyGui interface and threading
run.py         entry point
```

Requires MOTorNOT and `dearpygui`, `numpy`, `scipy`, `matplotlib`.
