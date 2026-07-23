''' Real-time simulation engine.

    A thin controller around MOTorNOT that advances a rubidium ensemble one small
    timestep at a time so a GUI can drive it live: switch the cooling beams,
    repumper and dipole trap on and off and watch the cloud respond.

    Physics assembled from MOTorNOT plus two effects needed for a believable
    *live* cloud that the library leaves out on purpose:

      * photon-recoil heating -- random momentum kicks proportional to the
        scattering rate, so the MOT settles at a finite size instead of
        collapsing to a point;
      * gravity -- so an untrapped cloud falls out of view.

    Internal-state shelving is modelled with MOTorNOT.LevelDynamics: the global
    bright fraction scales the MOT force (dark atoms feel no light) and the
    excited-state population sets the fluorescence the camera sees. Turn the
    repumper off and the atoms pump dark -- the glow fades and the cloud drops.
'''
import threading
import time
import numpy as np
from scipy.constants import k as kB, hbar, physical_constants

import MOTorNOT as mn
from MOTorNOT import backend
from MOTorNOT.dipole import DipoleTrap, OpticalLattice
from MOTorNOT.levels import LevelDynamics
from MOTorNOT.integration import integrate
from MOTorNOT import diagnostics as dg
from MOTorNOT import recapture as rc

amu = physical_constants['atomic mass constant'][0]
G_ACCEL = 9.80665


def analytic_trap_force(trap, X):
    ''' Exact analytic force -grad U of a MOTorNOT DipoleTrap / OpticalLattice,
        computed directly from the Gaussian-beam gradient instead of by finite
        differences. Same result, ~6x cheaper, and array-agnostic (CPU/GPU).

        For a beam of waist w0 along `axis` with U = C/w2 * exp(-2 r^2/w2),
        w2 = w0^2 (1 + (z/zR)^2):
            F_transverse_i = U * 4 x_i / w2
            F_axial        = U * (dw2/w2) * (1 - 2 r^2/w2),  dw2 = 2 w0^2 z/zR^2
        An optical lattice multiplies U by 4 cos^2(kz); its gradient adds the
        standing-wave term 4k U sin(2kz) along the axis.
    '''
    xp = backend.get_array_module(X)
    X = xp.atleast_2d(X)
    a = trap.axis
    others = [i for i in range(3) if i != a]
    dX = X - xp.asarray(trap.center)
    z = dX[:, a]
    r2 = dX[:, others[0]] ** 2 + dX[:, others[1]] ** 2
    w0, zR = trap.waist, trap.zR
    w2 = w0 ** 2 * (1 + (z / zR) ** 2)
    U = trap._coeff * (2 * trap.power / np.pi) * xp.exp(-2 * r2 / w2) / w2  # envelope
    dw2 = 2 * w0 ** 2 * z / zR ** 2

    F = xp.zeros(X.shape)
    F[:, others[0]] = U * 4 * dX[:, others[0]] / w2
    F[:, others[1]] = U * 4 * dX[:, others[1]] / w2
    F[:, a] = U * (dw2 / w2) * (1 - 2 * r2 / w2)

    if isinstance(trap, OpticalLattice):
        k = 2 * np.pi / trap.wavelength
        cos2 = xp.cos(k * z) ** 2
        F[:, others[0]] = cos2 * 4 * F[:, others[0]]
        F[:, others[1]] = cos2 * 4 * F[:, others[1]]
        F[:, a] = cos2 * 4 * F[:, a] + 4 * k * U * xp.sin(2 * k * z)
    return F


class DipoleTrapModel:
    ''' A selectable optical dipole potential built to a requested depth.

        Kinds:
            'gaussian' -- one focused red-detuned beam (a Gaussian well)
            'lattice'  -- retro-reflected 1D lattice (wells every lambda/2)
            'crossed'  -- two perpendicular Gaussian beams (a tight 3D trap)

        The laser power is solved from the requested depth (U is linear in
        power). Exposes exactly the interface the engine and MOTorNOT.recapture
        use: potential/force/acceleration, mass, center, waist, depth_uK.
    '''

    def __init__(self, atom, potential, wavelength, waist, depth_uK, axis=2):
        self.mass = atom['mass'] * amu
        self.waist = waist
        self.center = np.zeros(3)
        self.kind = potential
        target_J = depth_uK * 1e-6 * kB

        def gaussian(power, ax):
            return DipoleTrap(atom=atom, wavelength=wavelength, power=power,
                              waist=waist, axis=ax)

        if potential == 'lattice':
            probe = OpticalLattice(atom=atom, wavelength=wavelength, power=1.0,
                                   waist=waist, axis=axis)
            power = target_J / probe.depth()
            self.traps = [OpticalLattice(atom=atom, wavelength=wavelength,
                                         power=power, waist=waist, axis=axis)]
        elif potential == 'crossed':
            probe = gaussian(1.0, 0)
            power = (target_J / 2) / probe.depth()   # two beams add at centre
            self.traps = [gaussian(power, 0), gaussian(power, 2)]
        else:  # 'gaussian'
            probe = gaussian(1.0, axis)
            power = target_J / probe.depth()
            self.traps = [gaussian(power, axis)]

        # tightest oscillation frequency (used to cap the integrator timestep)
        self.omega_max = max(max(abs(np.asarray(t.trap_frequencies())))
                             for t in self.traps)

    def potential(self, X):
        return sum(t.potential(X) for t in self.traps)

    def force(self, X):
        return sum(analytic_trap_force(t, X) for t in self.traps)

    def acceleration(self, X, V=None):
        return self.force(X) / self.mass

    def depth_uK(self):
        U0 = backend.asnumpy(self.potential(self.center.reshape(1, 3)))[0]
        return float(abs(U0)) / kB * 1e6

    def trap_frequencies(self):
        return self.traps[0].trap_frequencies()


class RealTimeSimulation:
    def __init__(self, config):
        self.cfg = config
        self.atom = mn.atom(config.atom)
        self.mass = self.atom['mass'] * amu
        self.linewidth = 2 * np.pi * self.atom['gamma']
        self.wavenumber = 2 * np.pi / (self.atom['wavelength'] * 1e-9)
        self.v_recoil = hbar * self.wavenumber / self.mass

        # laser / field switches
        self.cooling_on = True
        self.repumper_on = True
        self.dipole_on = False
        self.imaging_on = False        # dedicated illumination laser (see below)

        # internal-state populations [F=2, excited, F=1(dark)]
        self.populations = np.array([1.0, 0.0, 0.0])
        self.dark_fraction = 0.0
        self.bright_fraction = 1.0
        self.scattering_rate = 0.0

        self.speed = self.cfg.speed

        # linearised-MOT model, refreshed asynchronously by CoefficientEstimator.
        # config_version is bumped whenever the MOT config changes, so stale
        # coefficients (fitted under a previous config) are never applied.
        self.config_version = 0
        self.center = np.zeros(3)      # linearisation centre (cloud centroid)
        self.F0 = np.zeros(3)          # full force at the centre (offset)
        self.kappa = np.zeros(3)       # spring constants per axis
        self.beta = np.zeros(3)        # damping constants per axis
        self.model_error = float('nan')  # linear-vs-full force discrepancy

        self.time = 0.0
        self._build_mot()
        self._build_dipole()
        self.reset_cloud()

    # ------------------------------------------------------------ build ----
    def _build_mot(self):
        m = self.cfg.mot
        self.mot = mn.six_beam_mot(self.atom, power=m.power, radius=m.radius,
                                   detuning=m.detuning, B_gradient=m.B_gradient,
                                   handedness=m.handedness)
        self._linearise_mot()

    def _linearise_mot(self):
        ''' Synchronous one-shot linearisation about the origin. Gives the fast
            model valid coefficients immediately on start-up and whenever the
            config changes; the CoefficientEstimator then refines them around
            the live cloud centroid. '''
        self.kappa, self.beta, self.F0 = fit_linear_coeffs(self.mot, np.zeros(3))
        self.center = np.zeros(3)

    def _build_dipole(self):
        d = self.cfg.dipole
        self.trap = DipoleTrapModel(self.atom, d.potential, d.wavelength,
                                    d.waist, d.depth_uK, d.axis)

    def reset_cloud(self):
        c = self.cfg.cloud
        X, V = mn.thermal_cloud(self.atom, c.N, c.temperature, c.sigma_r,
                                seed=None)
        self.X = backend.asarray(X)
        self.V = backend.asarray(V)
        self.populations = np.array([1.0, 0.0, 0.0])
        self.dark_fraction = 0.0
        self.bright_fraction = 1.0
        self.scattering_rate = 0.0
        self.time = 0.0

    # --------------------------------------------------------- controls ----
    def set_cooling(self, on):
        self.cooling_on = bool(on)

    def set_repumper(self, on):
        self.repumper_on = bool(on)

    def set_dipole(self, on):
        self.dipole_on = bool(on)

    def set_imaging(self, on):
        ''' The imaging laser is a separate, near-resonant illumination beam:
            it makes atoms fluoresce (so they are visible on the camera) even
            when the cooling beams are off, exactly as a real experiment images
            atoms held in a dark dipole trap. Modelled as pure illumination
            here (adds brightness, applies no force). '''
        self.imaging_on = bool(on)

    def recapture(self):
        ''' One-click MOT -> dipole transfer: cut the MOT, hold the dipole, and
            switch on the imaging laser so the transferred atoms stay visible. '''
        self.cooling_on = False
        self.dipole_on = True
        self.imaging_on = True

    def update_mot(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.cfg.mot, key, value)
        self.config_version += 1        # invalidate stale linear coefficients
        self._build_mot()               # immediate synchronous refit at origin

    # -------------------------------------------- linear-model plumbing ----
    def operating_point(self, sample_n=200):
        ''' A consistent snapshot for the estimator, taken under the caller's
            lock: cloud centroid and RMS spreads, a random atom subsample for
            the error metric, and the MOT parameters + config generation. '''
        X = backend.asnumpy(self.X)
        V = backend.asnumpy(self.V)
        n = X.shape[0]
        idx = np.random.choice(n, size=min(sample_n, n), replace=False)
        m = self.cfg.mot
        return {
            'center': X.mean(axis=0),
            'rms_x': X.std(axis=0),
            'rms_v': V.std(axis=0),
            'sample_X': X[idx].copy(),
            'sample_V': V[idx].copy(),
            'mot_params': dict(power=m.power, radius=m.radius, detuning=m.detuning,
                               B_gradient=m.B_gradient, handedness=m.handedness),
            'version': self.config_version,
            'sim_time': self.time,
        }

    def set_linear_coeffs(self, kappa, beta, F0, center, version, error):
        ''' Adopt refreshed coefficients only if they were fitted under the
            current config generation (otherwise they are stale and ignored). '''
        if version != self.config_version:
            return False
        self.kappa, self.beta, self.F0, self.center = kappa, beta, F0, center
        self.model_error = error
        return True

    def update_dipole(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.cfg.dipole, key, value)
        self._build_dipole()

    # ------------------------------------------------------------- step ----
    def _update_populations(self, dt):
        ''' Advance the internal-state populations by dt.

            The cooling cycle (F=2 <-> excited) equilibrates on ~1/Gamma (tens of
            ns), far faster than a motional timestep, so it is adiabatically
            eliminated: within the bright manifold the excited fraction is the
            quasi-steady rho = W/(2W+Gamma). The only slow variable is the dark
            (F=1) fraction, pumped by the off-resonant leak and emptied by the
            repumper; it is integrated with RK4 at the motional dt.
        '''
        G = self.linewidth
        if self.cooling_on:
            s = self.cfg.mot.Isat_saturation
            detuning = self.cfg.mot.detuning * G
            W = (G / 2) * s / (1 + (2 * detuning / G) ** 2)
            rho = W / (2 * W + G)                 # excited fraction (bright atoms)
        else:
            rho = 0.0                             # no light -> no scattering
        repump = 1e6 if self.repumper_on else 0.0
        b = 1e-3                                   # excited-state leak to dark

        def dDdt(D):
            return G * b * rho * (1 - D) - repump * D
        D = self.dark_fraction
        k1 = dDdt(D); k2 = dDdt(D + 0.5 * dt * k1)
        k3 = dDdt(D + 0.5 * dt * k2); k4 = dDdt(D + dt * k3)
        D = min(max(D + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0), 1.0)
        self.dark_fraction = D

        Ne = rho * (1 - D)
        self.populations = np.array([(1 - rho) * (1 - D), Ne, D])
        self.scattering_rate = G * Ne              # photons/s per bright atom
        self.bright_fraction = 1 - D               # atoms that feel the light

    def _acceleration_fn(self):
        xp = backend.get_array_module(self.X)
        grav = xp.asarray([0.0, 0.0, -G_ACCEL]) if self.cfg.gravity \
            else xp.zeros(3)
        force_scale = self.bright_fraction if self.cooling_on else 0.0
        cooling_on = self.cooling_on
        dipole_on = self.dipole_on
        fast = self.cfg.fast_mode
        kappa = xp.asarray(self.kappa)
        beta = xp.asarray(self.beta)
        center = xp.asarray(self.center)
        F0 = xp.asarray(self.F0)

        def accel(Xq, Vq):
            a = grav + xp.zeros_like(Xq)
            if cooling_on and force_scale > 1e-6:
                if fast:  # linearised force about the live cloud centroid
                    a = a + force_scale * (F0 - kappa * (Xq - center)
                                           - beta * Vq) / self.mass
                else:     # full 6-beam scattering force
                    a = a + force_scale * self.mot.acceleration(Xq, Vq)
            if dipole_on:
                a = a + self.trap.acceleration(Xq)
            return a
        return accel

    def _stable_dt(self, dt):
        ''' Cap the timestep so the tightest active trap stays well-resolved by
            RK4 (dt*omega ~ 0.15). Without this a fast trap -- especially an
            optical lattice with MHz oscillations -- would blow up when the
            speed multiplier makes dt large. '''
        omega = 0.0
        if self.cooling_on:
            omega = max(omega, np.sqrt(max(self.kappa.max(), 0.0) / self.mass))
        if self.dipole_on:
            omega = max(omega, self.trap.omega_max)
        if omega > 0:
            dt = min(dt, 0.15 / omega)
        return dt

    def step(self, dt=None):
        ''' Advance the simulation by one timestep (scaled by self.speed, and
            capped for numerical stability of the active traps). '''
        dt = self.cfg.dt * self.speed if dt is None else dt
        dt = self._stable_dt(dt)
        self._update_populations(dt)

        accel = self._acceleration_fn()
        _, X, V = integrate(accel, self.X, self.V, dt, dt, record=False)
        self.X, self.V = X, V

        if self.cfg.recoil_heating and self.scattering_rate > 0:
            # random-walk momentum diffusion from absorption + spontaneous
            # emission; heating_factor calibrates the steady-state temperature.
            xp = backend.get_array_module(self.V)
            std = self.cfg.heating_factor * self.v_recoil \
                * np.sqrt(2 * self.scattering_rate * dt)
            self.V = self.V + std * xp.random.standard_normal(self.V.shape)

        self.time += dt

    def advance(self, n=None):
        ''' Advance `n` steps (default: cfg.substeps) -- one rendered frame. '''
        n = self.cfg.substeps if n is None else n
        for _ in range(n):
            self.step()

    # -------------------------------------------------------- readouts ----
    def snapshot(self):
        ''' Current atom positions (NumPy, N x 3) and the per-atom fluorescence
            weight for the camera: scattering from the cooling light plus, if the
            imaging laser is on, its illumination -- so atoms remain visible in a
            dark dipole trap. '''
        X = backend.asnumpy(self.X)
        weight = float(self.populations[1])
        if self.imaging_on:
            weight += self.cfg.imaging_scatter
        return X, weight

    def temperature(self):
        ''' Temperature of the whole ensemble. '''
        return dg.temperature(self.V, self.mass)

    def display_temperature(self):
        ''' Temperature of atoms still within the camera's field of view. After a
            transfer the untrapped atoms accelerate away and would otherwise
            inflate the whole-cloud temperature far above the trap depth; the
            atoms you actually see are the ones near the trap. '''
        X = backend.asnumpy(self.X)
        V = backend.asnumpy(self.V)
        inside = np.linalg.norm(X - self.trap.center, axis=1) < self.cfg.camera.fov
        if inside.sum() < 5:
            return self.temperature()
        return dg.temperature(V[inside], self.mass)

    def rms_size(self):
        return dg.rms_radius(self.X)[0]

    def n_atoms(self):
        return self.X.shape[0]

    def status(self):
        ''' Human-readable one-line status for the UI. '''
        return {
            'time_ms': self.time * 1e3,
            'T_uK': self.display_temperature() * 1e6,
            'rms_mm': self.rms_size() * 1e3,
            'bright': self.bright_fraction,
            'fluorescence': self.populations[1],
            'cooling': self.cooling_on,
            'repumper': self.repumper_on,
            'dipole': self.dipole_on,
            'imaging': self.imaging_on,
            'model_error': self.model_error,
        }

    def report(self):
        ''' Slow-cadence diagnostics of the transfer: how many atoms are bound
            in the current dipole potential (energy criterion E < 0), their
            temperature, the internal-state (level) distribution and the trap
            parameters. Meant to be polled every few seconds, not every frame. '''
        frac, mask = rc.capture_fraction(self.trap, self.X, self.V)
        n_total = self.n_atoms()
        T_trapped = dg.temperature(self.V[mask], self.mass) \
            if bool(mask.any()) else 0.0
        return {
            'n_total': n_total,
            'n_trapped': int(round(frac * n_total)),
            'trapped_fraction': frac,
            'T_all': self.temperature(),
            'T_trapped': T_trapped,
            'pop_F2': float(self.populations[0]),
            'pop_excited': float(self.populations[1]),
            'pop_dark': float(self.populations[2]),
            'bright_fraction': self.bright_fraction,
            'dipole_type': self.trap.kind,
            'dipole_depth_uK': self.trap.depth_uK(),
            'dipole_on': self.dipole_on,
        }


# ============================ linear-model fitting ==========================
def fit_linear_coeffs(mot, center, rms_x=None, rms_v=None):
    ''' Fit F(x, v) ~ F0 - kappa*(x - center) - beta*v by sampling the full MOT
        force around `center`. Finite-difference steps default to the cloud RMS
        (so the linear model is tangent over the cloud's actual extent) with a
        floor to stay well-conditioned. Returns (kappa, beta, F0), each (3,). '''
    center = np.asarray(center, dtype=float)
    c = center.reshape(1, 3)
    zero = np.zeros((1, 3))
    dx_vec = np.maximum(rms_x if rms_x is not None else np.full(3, 1e-4), 5e-5)
    dv_vec = np.maximum(rms_v if rms_v is not None else np.full(3, 0.1), 1e-2)

    F0 = backend.asnumpy(mot.force(c, zero))[0]
    kappa = np.zeros(3)
    beta = np.zeros(3)
    for i in range(3):
        ex = np.zeros((1, 3)); ex[0, i] = dx_vec[i]
        Fp = backend.asnumpy(mot.force(c + ex, zero))[0, i]
        Fm = backend.asnumpy(mot.force(c - ex, zero))[0, i]
        kappa[i] = -(Fp - Fm) / (2 * dx_vec[i])
        ev = np.zeros((1, 3)); ev[0, i] = dv_vec[i]
        Fvp = backend.asnumpy(mot.force(c, ev))[0, i]
        Fvm = backend.asnumpy(mot.force(c, -ev))[0, i]
        beta[i] = -(Fvp - Fvm) / (2 * dv_vec[i])
    return kappa, beta, F0


def linear_model_error(mot, kappa, beta, F0, center, sample_X, sample_V):
    ''' Relative RMS discrepancy between the linear force and the full 6-beam
        force over a cloud subsample -- the validation metric. '''
    if len(sample_X) == 0:
        return float('nan')
    F_full = backend.asnumpy(mot.force(sample_X, sample_V))
    F_lin = F0 - kappa * (sample_X - center) - beta * sample_V
    denom = np.sqrt(np.mean(np.sum(F_full ** 2, axis=1))) + 1e-30
    return float(np.sqrt(np.mean(np.sum((F_full - F_lin) ** 2, axis=1))) / denom)


class CoefficientEstimator(threading.Thread):
    ''' Background supervisor: keeps its own full 6-beam MOT model and, in
        parallel with the fast simulation, refits the linear coefficients around
        the live cloud centroid and measures the linear-vs-full error.

        It never touches the running simulation directly. Instead it reads an
        "operating point" and writes coefficient sets through two callables
        supplied by the caller (which handle locking):

            get_op()      -> latest operating_point() dict, or None
            put_coeffs(c) -> hand a coefficient dict back to the sim

        Every coefficient set carries the config generation it was fitted under;
        the engine's set_linear_coeffs ignores any that no longer match, so a
        config change can never apply stale coefficients.
    '''

    def __init__(self, atom, get_op, put_coeffs, cadence=0.05):
        super().__init__(daemon=True)
        self.atom = atom
        self.get_op = get_op
        self.put_coeffs = put_coeffs
        self.cadence = cadence
        self.running = False
        self._mot = None
        self._built_version = -1

    def _ensure_mot(self, params, version):
        if version != self._built_version:
            self._mot = mn.six_beam_mot(
                self.atom, power=params['power'], radius=params['radius'],
                detuning=params['detuning'], B_gradient=params['B_gradient'],
                handedness=params['handedness'])
            self._built_version = version

    def run(self):
        self.running = True
        while self.running:
            op = self.get_op()
            if op is not None:
                self._ensure_mot(op['mot_params'], op['version'])
                kappa, beta, F0 = fit_linear_coeffs(
                    self._mot, op['center'], op['rms_x'], op['rms_v'])
                error = linear_model_error(
                    self._mot, kappa, beta, F0, op['center'],
                    op['sample_X'], op['sample_V'])
                self.put_coeffs({'kappa': kappa, 'beta': beta, 'F0': F0,
                                 'center': op['center'], 'version': op['version'],
                                 'error': error})
            time.sleep(self.cadence)

    def stop(self):
        self.running = False
