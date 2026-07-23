''' Simulated IR camera.

    Turns a 3D atom distribution into a 2D "camera frame" the way a real
    fluorescence/absorption image is formed: project the atoms onto the sensor
    plane, bin them into pixels weighted by how brightly they scatter, blur with
    the optical point-spread function, add sensor noise and false-colour it.

    render() returns an (H, W, 4) float32 RGBA image in [0, 1], ready to be
    pushed straight into a Dear PyGui texture.
'''
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl


class Camera:
    def __init__(self, config):
        self.cfg = config
        self._peak = 1.0  # running normalisation so brightness auto-scales

    def _image_axes(self):
        ''' The two spatial axes spanning the image (horizontal, vertical),
            given the line-of-sight axis. Vertical is chosen so gravity (-z)
            points down when looking along y. '''
        va = self.cfg.view_axis
        axes = [a for a in (0, 1, 2) if a != va]
        horizontal, vertical = axes[0], axes[1]
        return horizontal, vertical

    def render(self, X, weight=1.0, center=(0.0, 0.0, 0.0)):
        ''' Build an RGBA frame from atom positions X (N,3) in metres.

            weight: scalar or per-atom fluorescence brightness.
            center: trap centre placed at the middle of the frame.
        '''
        cfg = self.cfg
        W, H = cfg.resolution
        ax_h, ax_v = self._image_axes()
        half = cfg.fov / 2.0
        c = np.asarray(center, dtype=float)

        u = X[:, ax_h] - c[ax_h]
        v = X[:, ax_v] - c[ax_v]
        w = weight if np.ndim(weight) else float(weight)

        # bin atoms into pixels (v is the row index -> vertical)
        frame, _, _ = np.histogram2d(
            v, u, bins=[H, W],
            range=[[-half, half], [-half, half]],
            weights=(w * np.ones(len(X)) if np.ndim(w) == 0 else w))
        frame = np.flipud(frame)  # image row 0 at the top

        # optical blur
        if cfg.psf_sigma_px > 0:
            frame = gaussian_filter(frame, cfg.psf_sigma_px)

        # auto-scaling exposure: track the peak so a bright cloud fills the range
        peak = frame.max()
        if peak > 0:
            self._peak = 0.9 * self._peak + 0.1 * peak
        norm = frame / (self._peak + 1e-12) * cfg.exposure
        norm = np.clip(norm, 0.0, 1.0)

        # sensor read noise
        if cfg.read_noise > 0:
            norm = np.clip(norm + cfg.read_noise * np.random.standard_normal(norm.shape),
                           0.0, 1.0)

        rgba = mpl.colormaps[cfg.colormap](norm).astype(np.float32)
        return rgba

    def render_flat(self, X, weight=1.0, center=(0.0, 0.0, 0.0)):
        ''' Same as render() but flattened to a 1D float list for a DPG texture. '''
        return self.render(X, weight, center).ravel()
