''' MidnightTrapsfRb -- a real-time optical-trap "cell" simulator.

    A desktop application that models a rubidium magneto-optical trap and its
    transfer ("Trapsf" = Trap + Transfer) into an optical dipole trap, and
    renders it live as if watched through an IR camera. The physics engine is
    MOTorNOT (installed as a separate library); this package is the application
    layer on top of it.
'''
__version__ = '1.0.0'

from .config import (MOTConfig, DipoleConfig, CloudConfig, CameraConfig,
                     SimConfig)
from .engine import RealTimeSimulation, CoefficientEstimator
from .camera import Camera
# NB: the GUI (`.app`) is imported lazily (see __main__) so the engine and
# camera can be used headless without requiring Dear PyGui.
