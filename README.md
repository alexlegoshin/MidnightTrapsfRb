# MidnightTrapsfRb

A real-time desktop **optical-trap "cell" simulator** for rubidium-87. It models
a magneto-optical trap (MOT) and its transfer into an optical dipole trap and
renders the atoms live, as if you were watching the cell through an IR
fluorescence camera. Switch the cooling beams, repumper and dipole trap on and
off and watch the cloud cool, glow, go dark or fall in real time.

*Midnight* (built during many late nights, à la Midnight Club); *Trapsf* =
**Trap** + **Transfer** (and it conveniently ends in **Rb**).

## History & design

MidnightTrapsfRb began as a **from-scratch simulation** of Rb-87 MOT cooling and
recapture into a dipole trap, with its own hand-written physics: 1D Doppler
cooling, hyperfine level populations with a repumper, Gaussian and optical-lattice
dipole potentials, and even a quantum (Schrödinger + tunnelling) treatment of the
recapture.

On careful review that bespoke physics turned out to be broken in most of its
core blocks: the MOT force had the **wrong sign** (it heated rather than cooled)
and produced **no spatial restoring force** (so it was not actually a trap); the
dipole potential came out **repulsive** instead of attractive; the level-transition
model was numerically **inert**; and the quantum recapture was **dimensionally
inconsistent**. Rather than patch a pile of subtle sign and unit bugs, the
project switched to a dedicated engine —
[**MOTorNOT**](https://github.com/alexlegoshin/MOTorNOT) — and became the
application layer on top of it.

**What that trade cost and bought:**

- **Better:** MOTorNOT's MOT force is a correct semiclassical model (σ± / Zeeman
  sublevels, Doppler shift, saturation), with real magnetic-field geometry and a
  correct AC-Stark dipole potential. All of it is tested and physical.
- **Worse:** MOTorNOT is **purely semiclassical** — it does *not* attempt the
  quantum bound-state / tunnelling recapture the original aimed at. For the
  many-µK trap depths and thermal atoms simulated here the classical
  energy criterion dominates, so this is an acceptable simplification.
- MOTorNOT originally had **no** dipole trap, level dynamics, ensemble
  diagnostics or recapture — these were added *to the library* (on correct
  foundations), so the ambition of the original project lives on.

The result is a responsive, visual **simulator** of the trap cell with
configurable lasers, repumper, a choice of dipole potential, a live camera and a
recapture/level report.

## Install & run

> **Important:** install MOTorNOT from **this fork**
> (`github.com/alexlegoshin/MOTorNOT`), *not* from PyPI. The published
> `pip install MOTorNOT` is an old version that lacks the dipole traps, level
> dynamics, diagnostics, recapture and GPU backend this app relies on.

```bash
# 1. install the physics engine from the alexlegoshin fork
pip install git+https://github.com/alexlegoshin/MOTorNOT
#    ...or, if you have a local checkout next to this repo:
#    pip install -e ../MOTorNOT
#    optional GPU: pip install "cupy-cuda12x[ctk]"

# 2. install the app's own dependencies
pip install -r requirements.txt

# 3. run
python run.py                        # or:  python -m midnightrb
```

## Using it

- **Configuration** — set the MOT (power, detuning, beam radius, B-gradient), the
  dipole trap (**potential**: Gaussian well / optical lattice / crossed beams,
  plus **depth**, waist and wavelength) and the initial cloud (atom number,
  temperature, size); *Apply & reset cloud*.
- **Controls** — toggle the cooling beams, repumper, dipole trap and the
  **imaging laser** (illumination that makes atoms glow even with the cooling
  beams off); hit **Recapture into dipole** to cut the MOT, hold the dipole and
  switch the imaging laser on in one click; set the simulation **speed** and the
  fast/full model switch.
- **Report** (updates every ~2 s) — how many atoms were recaptured (and the
  fraction), the trapped vs. whole-cloud temperature, and the internal-state
  level distribution (F=2 / F′=3 / dark F=1), plus the active trap parameters.
  The trapped temperature here is the meaningful one (typically well below the
  trap depth); the status line shows `T(view)`, the temperature of atoms still in
  frame, because after a transfer the untrapped atoms fly off and would otherwise
  dominate a whole-cloud average.
- **Spectroscopy** — one button sweeps the cooling detuning over a range you
  choose and saves a numbered camera frame per point plus a fluorescence-vs-
  detuning spectrum into `spectroscopy_output/`. The MOT fluorescence peaks just
  red of resonance and collapses on the blue side (where the trap turns
  anti-trapping) — a good check that the model behaves like a real MOT.
- **Camera** — field of view, optical blur, exposure, colormap and viewing axis.

The window is fully resizable; the camera view stays square and grows to fill the
space, and the controls wrap to fit.

Try: cool with beams + repumper → open the Report tab → hit *Recapture* and watch
the recaptured fraction settle → turn the repumper off and watch the atoms pump
dark and fall.

## The physics

The engine composes MOTorNOT's building blocks and adds what a *live* cloud
needs. See the MOTorNOT README for the full equations; in brief:

- **MOT force** — the semiclassical scattering force `F = Σ_b ħ k_b R_b(x,v)`
  with per-beam rates that carry the Doppler shift `k·v`, the Zeeman shift
  `m·gF·μ_B·|B|/ħ` and σ± polarisation weights. This gives both the velocity
  damping and the spatial restoring force.
- **Dipole potential** — the far-detuned AC-Stark well
  `U = −(3πc²/2ω₀³)·Γ·(1/(ω₀−ω)+1/(ω₀+ω))·I(r)`, red-detuned so `U < 0`.
  Selectable as a single Gaussian well, a retro-reflected 1D lattice (wells every
  `λ/2`), or two crossed beams (a tight 3D trap). The laser power is solved from
  the requested trap depth.
- **Level dynamics & fluorescence** — a 3-level rate model (F=2 ↔ excited,
  spontaneous decay with a dark-state leak, repumper). The bright fraction scales
  the MOT force (dark atoms feel no light) and the excited-state population sets
  the fluorescence the camera sees, so turning the repumper off makes the cloud
  go dark.
- **Recapture** — an atom is bound if `E = ½mv² + U(x) < 0`; the Report tab
  counts those. The transferred fraction is small (a few percent) because the
  MOT settles at a size much larger than a tight dipole trap — this is real, not
  a bug (no sub-Doppler compression is modelled). A single beam is oriented
  **horizontally** by default so gravity acts along a tightly confined direction;
  a vertical beam would let atoms leak out along its weak axis. The `crossed`
  geometry holds atoms best.
- **Imaging** — atoms are only visible while they scatter light. The cooling
  beams provide that in the MOT, but the far-detuned dipole trap is dark (its
  1064 nm light barely scatters, so it does not touch the internal state), so a
  separate near-resonant **imaging laser** illuminates the trapped atoms. It
  drives the same cycling transition as the cooling light, so it feeds the level
  model properly: it produces fluorescence *and* pumps atoms into the dark F=1
  state — turn the repumper off while imaging and the glow fades, exactly as in a
  real fluorescence image (Recapture switches the repumper on for this reason).
  Imaging recoil heating is off by default (`imaging_heats`) so the view is
  non-destructive; enable it for realistic, destructive imaging.
- **Recoil heating & gravity** — photon-recoil momentum diffusion lets the MOT
  settle at a realistic finite size (~Doppler-limit temperature) instead of
  collapsing to a point, and gravity makes an untrapped cloud fall.
- **Cell & recapture** — the atoms live in a vacuum cell with reflecting walls,
  so an untrapped cloud stays localised (and its velocity stays bounded) instead
  of accelerating away for ever. The MOT force is tapered to the beam radius and
  rolls off past the capture velocity, so switching the cooling beams off and
  back on disperses and then reassembles the cloud, just as a real MOT recaptures
  atoms from the cell.

### Staying real-time

The full 6-beam MOT force is accurate but expensive. For interactivity the app
runs a **linearised model** near the trap centre (`F ≈ F₀ − κ·(x−c) − β·v`),
~50× faster and, for the trapped cloud, matching the full force to ~0.1%.

Those coefficients are not frozen: a background **CoefficientEstimator** keeps
its own full 6-beam model and continuously re-fits `κ, β, F₀` around the *live*
cloud centroid, reporting the linear-vs-full **model error** in the status line.
Each coefficient set is tagged with a config generation, so changing a MOT
parameter can never apply stale coefficients — the fast model self-corrects while
staying cheap. The integrator timestep is auto-capped to the tightest active trap
frequency so even the MHz optical lattice stays stable under the speed
multiplier.

The dipole trap needs no such estimator: unlike the 6-beam MOT force, its
gradient is a simple analytic expression, so the dipole force is computed
**exactly** (and ~4× faster than by finite differences). The MOT model-error
indicator therefore applies only while the cooling beams are on.

Three threads cooperate through short-held locks — a **physics** thread (owns the
simulation), the **estimator**, and the **GUI** (renders the camera and handles
input) — so the window stays responsive no matter how heavy the physics is.

## Layout

```
midnightrb/
  config.py    dataclasses: MOT / dipole / cloud / camera / sim settings
  engine.py    RealTimeSimulation, DipoleTrapModel, CoefficientEstimator
  camera.py    Camera: atom positions -> simulated IR frame
  app.py       Dear PyGui interface and threading
run.py         entry point
```

Requires MOTorNOT and `dearpygui`, `numpy`, `scipy`, `matplotlib`.
