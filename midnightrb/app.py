''' Dear PyGui front-end: a live "optical trap cell" viewer.

    Three cooperating parts share the simulation through short-held locks:

      * a physics thread that solely owns the RealTimeSimulation, advances it in
        small batches and publishes snapshots (for the camera) and an operating
        point (for the estimator);
      * a background CoefficientEstimator that refits the fast linear MOT model
        against the full 6-beam physics and reports the model error;
      * the GUI thread, which renders the simulated camera and drives the
        controls, never blocking on the physics.
'''
import threading
import time
import numpy as np
import dearpygui.dearpygui as dpg

from .config import SimConfig
from .engine import RealTimeSimulation, CoefficientEstimator
from .camera import Camera

COLORMAPS = ['inferno', 'magma', 'plasma', 'viridis', 'hot', 'gray']


class TrapSimulatorApp:
    def __init__(self, config=None):
        self.cfg = config or SimConfig()
        self.sim = RealTimeSimulation(self.cfg)
        self.cam = Camera(self.cfg.camera)

        self.lock = threading.Lock()
        self.running = False
        self.batch = 6
        self._speed = self.cfg.speed

        # cross-thread buffers (guarded by self.lock)
        self._pub = None       # (X, weight, status) for the camera
        self._op = None        # operating_point() for the estimator
        self._coeff = None      # coefficients handed back by the estimator

        self.estimator = CoefficientEstimator(self.sim.atom, self._get_op,
                                              self._put_coeffs, cadence=0.05)
        self._frame_times = []
        W, H = self.cfg.camera.resolution
        self._tex_w, self._tex_h = W, H
        self._blank = np.zeros(W * H * 4, dtype=np.float32)

    # -------------------------------------------- thread communication ----
    def _get_op(self):
        with self.lock:
            return None if self._op is None else dict(self._op)

    def _put_coeffs(self, c):
        with self.lock:
            self._coeff = c

    def _physics_loop(self):
        while self.running:
            with self.lock:
                if self._coeff is not None:
                    c = self._coeff; self._coeff = None
                    self.sim.set_linear_coeffs(c['kappa'], c['beta'], c['F0'],
                                               c['center'], c['version'], c['error'])
                self.sim.speed = self._speed
                for _ in range(self.batch):
                    self.sim.step()
                X, w = self.sim.snapshot()
                self._pub = (X, w, self.sim.status())
                self._op = self.sim.operating_point()
            time.sleep(0.0008)  # yield so the GUI/estimator can take the lock

    # -------------------------------------------------------- callbacks ----
    def _set_cooling(self, s, a):
        with self.lock: self.sim.set_cooling(a)

    def _set_repumper(self, s, a):
        with self.lock: self.sim.set_repumper(a)

    def _set_dipole(self, s, a):
        with self.lock: self.sim.set_dipole(a)

    def _set_fast(self, s, a):
        with self.lock: self.sim.cfg.fast_mode = a

    def _set_speed(self, s, a):
        self._speed = a

    def _do_recapture(self, s, a):
        with self.lock:
            self.sim.recapture()
        dpg.set_value('chk_cooling', self.sim.cooling_on)
        dpg.set_value('chk_dipole', self.sim.dipole_on)

    def _reset_cloud(self, s, a):
        with self.lock: self.sim.reset_cloud()

    def _apply_config(self, s, a):
        with self.lock:
            self.sim.update_mot(power=dpg.get_value('mot_power'),
                                detuning=dpg.get_value('mot_detuning'),
                                radius=dpg.get_value('mot_radius'),
                                B_gradient=dpg.get_value('mot_grad'))
            self.sim.update_dipole(wavelength=dpg.get_value('dip_wl') * 1e-9,
                                   power=dpg.get_value('dip_power'),
                                   waist=dpg.get_value('dip_waist') * 1e-6,
                                   lattice=dpg.get_value('dip_lattice'))
            self.cfg.cloud.N = int(dpg.get_value('cloud_N'))
            self.cfg.cloud.temperature = dpg.get_value('cloud_T') * 1e-6
            self.cfg.cloud.sigma_r = dpg.get_value('cloud_sigma') * 1e-3
            self.sim.reset_cloud()

    def _apply_camera(self, s, a):
        c = self.cfg.camera
        c.fov = dpg.get_value('cam_fov') * 1e-3
        c.psf_sigma_px = dpg.get_value('cam_psf')
        c.exposure = dpg.get_value('cam_exposure')
        c.colormap = dpg.get_value('cam_cmap')
        c.view_axis = {'x': 0, 'y': 1, 'z': 2}[dpg.get_value('cam_axis')]

    # ------------------------------------------------------------- build ---
    def _build_ui(self):
        c = self.cfg
        with dpg.texture_registry():
            dpg.add_raw_texture(self._tex_w, self._tex_h, self._blank,
                                format=dpg.mvFormat_Float_rgba, tag='cam_tex')

        with dpg.window(tag='primary'):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=360, autosize_y=True):
                    with dpg.tab_bar():
                        self._tab_configuration(c)
                        self._tab_controls()
                        self._tab_camera(c)
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_text('', tag='status')
                    dpg.add_image('cam_tex', width=512, height=512)
        dpg.set_primary_window('primary', True)

    def _tab_configuration(self, c):
        with dpg.tab(label='Configuration'):
            dpg.add_text('MOT beams')
            dpg.add_input_float(label='power / beam (W)', tag='mot_power',
                                default_value=c.mot.power, step=1e-3, format='%.4f')
            dpg.add_input_float(label='detuning (linewidths)', tag='mot_detuning',
                                default_value=c.mot.detuning, step=0.1, format='%.2f')
            dpg.add_input_float(label='beam radius (m)', tag='mot_radius',
                                default_value=c.mot.radius, step=1e-3, format='%.4f')
            dpg.add_input_float(label='B gradient (T/m)', tag='mot_grad',
                                default_value=c.mot.B_gradient, step=0.01, format='%.3f')
            dpg.add_separator()
            dpg.add_text('Dipole trap')
            dpg.add_input_float(label='wavelength (nm)', tag='dip_wl',
                                default_value=c.dipole.wavelength * 1e9, step=1, format='%.1f')
            dpg.add_input_float(label='power (W)', tag='dip_power',
                                default_value=c.dipole.power, step=0.5, format='%.2f')
            dpg.add_input_float(label='waist (um)', tag='dip_waist',
                                default_value=c.dipole.waist * 1e6, step=1, format='%.1f')
            dpg.add_checkbox(label='optical lattice', tag='dip_lattice',
                             default_value=c.dipole.lattice)
            dpg.add_separator()
            dpg.add_text('Cloud')
            dpg.add_input_int(label='atoms N', tag='cloud_N', default_value=c.cloud.N, step=100)
            dpg.add_input_float(label='temperature (uK)', tag='cloud_T',
                                default_value=c.cloud.temperature * 1e6, step=10, format='%.0f')
            dpg.add_input_float(label='cloud size (mm)', tag='cloud_sigma',
                                default_value=c.cloud.sigma_r * 1e3, step=0.1, format='%.2f')
            dpg.add_spacer(height=6)
            dpg.add_button(label='Apply & reset cloud', callback=self._apply_config, width=-1)

    def _tab_controls(self):
        with dpg.tab(label='Controls'):
            dpg.add_checkbox(label='Cooling beams', tag='chk_cooling',
                             default_value=True, callback=self._set_cooling)
            dpg.add_checkbox(label='Repumper', tag='chk_repumper',
                             default_value=True, callback=self._set_repumper)
            dpg.add_checkbox(label='Dipole trap', tag='chk_dipole',
                             default_value=False, callback=self._set_dipole)
            dpg.add_spacer(height=10)
            dpg.add_button(label='>> Recapture into dipole <<',
                           callback=self._do_recapture, width=-1, height=40)
            dpg.add_spacer(height=6)
            dpg.add_button(label='Reset cloud', callback=self._reset_cloud, width=-1)
            dpg.add_separator()
            dpg.add_text('Simulation')
            dpg.add_slider_float(label='speed', tag='sim_speed', default_value=self._speed,
                                 min_value=1.0, max_value=200.0, callback=self._set_speed)
            dpg.add_checkbox(label='fast (linearised) model', tag='chk_fast',
                             default_value=self.cfg.fast_mode, callback=self._set_fast)
            dpg.add_separator()
            dpg.add_text('Sequence:', color=(150, 150, 150))
            dpg.add_text('1. Cool with beams + repumper on\n'
                         '2. Turn cooling off\n'
                         '3. Turn dipole on (or hit Recapture)\n'
                         '4. Repumper off => atoms go dark',
                         wrap=330, color=(150, 150, 150))

    def _tab_camera(self, c):
        with dpg.tab(label='Camera'):
            dpg.add_slider_float(label='FOV (mm)', tag='cam_fov',
                                 default_value=c.camera.fov * 1e3, min_value=0.5,
                                 max_value=10.0, callback=self._apply_camera)
            dpg.add_slider_float(label='PSF blur (px)', tag='cam_psf',
                                 default_value=c.camera.psf_sigma_px, min_value=0.0,
                                 max_value=5.0, callback=self._apply_camera)
            dpg.add_slider_float(label='exposure', tag='cam_exposure',
                                 default_value=c.camera.exposure, min_value=0.1,
                                 max_value=5.0, callback=self._apply_camera)
            dpg.add_combo(COLORMAPS, label='colormap', tag='cam_cmap',
                          default_value=c.camera.colormap, callback=self._apply_camera)
            dpg.add_combo(['x', 'y', 'z'], label='view axis', tag='cam_axis',
                          default_value='xyz'[c.camera.view_axis], callback=self._apply_camera)

    # -------------------------------------------------------- per-frame ---
    def _update_frame(self):
        t0 = time.perf_counter()
        with self.lock:
            pub = self._pub
        if pub is not None:
            X, weight, st = pub
            dpg.set_value('cam_tex', self.cam.render_flat(X, weight))
            self._frame_times.append(t0)
            self._frame_times = self._frame_times[-30:]
            fps = 0.0
            if len(self._frame_times) > 1:
                fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0] + 1e-9)
            err = st['model_error']
            err_txt = 'n/a' if err != err else '%.1f%%' % (err * 100)
            dpg.set_value('status',
                          'sim t=%7.1f ms   T=%6.1f uK   rms=%.2f mm   fluor=%.3f\n'
                          'cooling=%s  repumper=%s  dipole=%s    model err=%s    %.0f fps'
                          % (st['time_ms'], st['T_uK'], st['rms_mm'], st['fluorescence'],
                             st['cooling'], st['repumper'], st['dipole'], err_txt, fps))

    # ------------------------------------------------------------- run ----
    def run(self, smoke_frames=0, screenshot=None):
        dpg.create_context()
        self._build_ui()
        dpg.create_viewport(title='MidnightTrapsfRb - optical trap cell', width=920, height=620)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        self.running = True
        physics = threading.Thread(target=self._physics_loop, daemon=True)
        physics.start()
        self.estimator.running = True
        self.estimator.start()

        frame_i = 0
        while dpg.is_dearpygui_running():
            self._update_frame()
            dpg.render_dearpygui_frame()
            frame_i += 1
            if smoke_frames and frame_i >= smoke_frames:
                if screenshot:
                    dpg.output_frame_buffer(screenshot)
                break

        self.running = False
        self.estimator.stop()
        time.sleep(0.05)
        dpg.destroy_context()


def main():
    TrapSimulatorApp().run()


if __name__ == '__main__':
    main()
