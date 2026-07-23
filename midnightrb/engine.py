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
import numpy as np
from scipy.constants import k as kB, hbar, physical_constants

import MOTorNOT as mn
from MOTorNOT import backend
from MOTorNOT.dipole import DipoleTrap, OpticalLattice
from MOTorNOT.levels import LevelDynamics
from MOTorNOT.integration import integrate
from MOTorNOT import diagnostics as dg

amu = physical_constants['atomic mass constant'][0]
G_ACCEL = 9.80665


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

        # internal-state populations [F=2, excited, F=1(dark)]
        self.populations = np.array([1.0, 0.0, 0.0])
        self.dark_fraction = 0.0
        self.bright_fraction = 1.0
        self.scattering_rate = 0.0

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

    def _build_dipole(self):
        d = self.cfg.dipole
        cls = OpticalLattice if d.lattice else DipoleTrap
        self.trap = cls(atom=self.atom, wavelength=d.wavelength, power=d.power,
                        waist=d.waist, axis=d.axis)

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

    def recapture(self):
        ''' One-click MOT -> dipole transfer: cut the MOT, hold the dipole. '''
        self.cooling_on = False
        self.dipole_on = True

    def update_mot(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.cfg.mot, key, value)
        self._build_mot()

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

        def accel(Xq, Vq):
            a = grav + xp.zeros_like(Xq)
            if cooling_on and force_scale > 1e-6:
                a = a + force_scale * self.mot.acceleration(Xq, Vq)
            if dipole_on:
                a = a + self.trap.acceleration(Xq)
            return a
        return accel

    def step(self, dt=None):
        ''' Advance the simulation by one timestep. '''
        dt = self.cfg.dt if dt is None else dt
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
            weight (excited-state population) for the camera. '''
        X = backend.asnumpy(self.X)
        weight = float(self.populations[1])
        return X, weight

    def temperature(self):
        return dg.temperature(self.V, self.mass)

    def rms_size(self):
        return dg.rms_radius(self.X)[0]

    def n_atoms(self):
        return self.X.shape[0]

    def status(self):
        ''' Human-readable one-line status for the UI. '''
        return {
            'time_ms': self.time * 1e3,
            'T_uK': self.temperature() * 1e6,
            'rms_mm': self.rms_size() * 1e3,
            'bright': self.bright_fraction,
            'fluorescence': self.populations[1],
            'cooling': self.cooling_on,
            'repumper': self.repumper_on,
            'dipole': self.dipole_on,
        }
