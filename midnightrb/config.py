''' Configuration dataclasses for the simulator.

    All quantities are SI unless noted. Detunings are given in units of the
    natural linewidth (negative = red). These objects are plain data; the
    engine reads them and (re)builds the MOTorNOT objects when they change.
'''
from dataclasses import dataclass, field


@dataclass
class MOTConfig:
    ''' Six-beam MOT parameters. '''
    power: float = 15e-3          # power per beam, W
    radius: float = 10e-3         # beam radius, m
    detuning: float = -1.0        # laser detuning in linewidths (negative = red)
    B_gradient: float = 0.1       # magnetic field gradient B0, T/m
    handedness: int = -1          # polarization handedness (with B0>0 => 3D trap)
    Isat_saturation: float = 3.0  # on-resonance saturation parameter s0 = I/Isat


@dataclass
class DipoleConfig:
    ''' Optical dipole trap parameters. '''
    wavelength: float = 1064e-9   # trap laser wavelength, m
    power: float = 3.0            # trap power, W
    waist: float = 25e-6          # 1/e^2 beam waist, m
    axis: int = 2                 # propagation axis (0=x,1=y,2=z)
    lattice: bool = False         # retro-reflected 1D lattice instead of a well


@dataclass
class CloudConfig:
    ''' Initial atomic ensemble. '''
    N: int = 800                  # number of atoms
    temperature: float = 300e-6   # initial temperature, K
    sigma_r: float = 0.5e-3       # initial RMS cloud radius, m


@dataclass
class CameraConfig:
    ''' Simulated IR camera looking at the trap. '''
    resolution: tuple = (256, 256)  # pixels (width, height)
    fov: float = 4e-3               # field of view across the frame, m
    view_axis: int = 1              # line of sight (0=x,1=y,2=z); image spans the other two
    psf_sigma_px: float = 1.5       # optical blur (point-spread) in pixels
    exposure: float = 1.0           # brightness gain
    read_noise: float = 0.01        # additive Gaussian read noise (0..1 scale)
    colormap: str = 'inferno'       # matplotlib colormap name for false colour


@dataclass
class SimConfig:
    ''' Top-level simulation settings, bundling the sub-configs. '''
    atom: str = 'Rb87'
    dt: float = 2e-6                # physics timestep, s
    substeps: int = 4               # physics steps per published buffer
    speed: float = 20.0             # sim-time multiplier (dt_eff = dt * speed)
    fast_mode: bool = True          # linearised MOT force near centre (fast)
    gravity: bool = True            # let atoms fall when untrapped
    recoil_heating: bool = True     # photon-recoil momentum diffusion
    heating_factor: float = 0.4     # scales recoil heating (=> Doppler-limit T)
    mot: MOTConfig = field(default_factory=MOTConfig)
    dipole: DipoleConfig = field(default_factory=DipoleConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
