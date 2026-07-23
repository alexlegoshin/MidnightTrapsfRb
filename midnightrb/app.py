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
        self._panel_w = 360        # left control-panel width

        # cross-thread buffers (guarded by self.lock)
        self._pub = None       # (X, weight, status) for the camera
        self._op = None        # operating_point() for the estimator
        self._coeff = None      # coefficients handed back by the estimator
        self._report = None     # slow-cadence transfer/level report
        self._last_report = 0.0
        self.report_interval = 2.0  # seconds between reports
        self._spec_request = None   # pending spectroscopy sweep parameters
        self._spec_status = ''      # spectroscopy progress text

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
            if self._spec_request is not None:
                params = self._spec_request
                self._spec_request = None
                try:
                    self._run_spectroscopy(params)
                except Exception as exc:          # keep the loop alive on error
                    self._spec_status = 'spectroscopy error: %s' % exc
                continue
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
                now = time.time()
                if now - self._last_report > self.report_interval:
                    self._report = self.sim.report()
                    self._last_report = now
            time.sleep(0.0008)  # yield so the GUI/estimator can take the lock

    # -------------------------------------------------------- callbacks ----
    def _set_cooling(self, s, a):
        with self.lock: self.sim.set_cooling(a)

    def _set_repumper(self, s, a):
        with self.lock: self.sim.set_repumper(a)

    def _set_dipole(self, s, a):
        with self.lock: self.sim.set_dipole(a)

    def _set_imaging(self, s, a):
        with self.lock: self.sim.set_imaging(a)

    def _set_fast(self, s, a):
        with self.lock: self.sim.cfg.fast_mode = a

    def _set_speed(self, s, a):
        self._speed = a

    def _do_recapture(self, s, a):
        with self.lock:
            self.sim.recapture()
            self._last_report = 0.0   # refresh the report promptly
        dpg.set_value('chk_cooling', self.sim.cooling_on)
        dpg.set_value('chk_dipole', self.sim.dipole_on)
        dpg.set_value('chk_imaging', self.sim.imaging_on)

    def _reset_cloud(self, s, a):
        with self.lock: self.sim.reset_cloud()

    def _run_spec_clicked(self, s, a):
        ''' Queue a spectroscopy sweep (the physics thread runs it). '''
        if self._spec_request is None:
            self._spec_request = dict(
                start=dpg.get_value('spec_start'),
                stop=dpg.get_value('spec_stop'),
                n=max(2, int(dpg.get_value('spec_n'))),
                settle=max(1, int(dpg.get_value('spec_settle'))))

    def _run_spectroscopy(self, p):
        ''' Sweep the cooling detuning, settle at each point, save a numbered
            camera frame, and record the total fluorescence vs detuning (the
            spectrum). Runs in the physics thread, publishing as it goes so the
            live view shows the sweep. Output goes to spectroscopy_output/. '''
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = os.path.join(root, 'spectroscopy_output')
        os.makedirs(out, exist_ok=True)

        dets = np.linspace(p['start'], p['stop'], p['n'])
        with self.lock:
            orig = self.sim.cfg.mot.detuning
        frames, signal, temps = [], [], []
        for i, d in enumerate(dets):
            if not self.running:
                break
            with self.lock:
                self.sim.update_mot(detuning=float(d))
            for _ in range(p['settle']):
                if not self.running:
                    break
                with self.lock:
                    for _ in range(self.batch):
                        self.sim.step()
                    X, w = self.sim.snapshot()
                    self._pub = (X, w, self.sim.status())
                    self._op = self.sim.operating_point()
                time.sleep(0.0005)
            with self.lock:
                X, w = self.sim.snapshot()
                n_atoms = self.sim.n_atoms()
                temp = self.sim.display_temperature()
                frame = self.cam.intensity_frame(X, w, center=self.sim.trap.center)
            frames.append(frame)
            # collected fluorescence = light within the camera frame, so atoms
            # that disperse out of view (e.g. at blue detuning) drop the signal
            signal.append(float(frame.sum()))
            temps.append(temp)
            _ = n_atoms
            self._spec_status = 'running: %d/%d   detuning=%+.2f G' % (i + 1, len(dets), d)

        with self.lock:                          # restore the original detuning
            self.sim.update_mot(detuning=orig)

        # save frames on a common brightness scale so the resonance is visible
        scale = max((f.max() for f in frames), default=1.0) + 1e-12
        cmap = mpl.colormaps[self.cfg.camera.colormap]
        for i, (d, f) in enumerate(zip(dets, frames)):
            rgba = cmap(np.clip(f / scale, 0.0, 1.0))
            plt.imsave(os.path.join(out, 'spec_%03d_det%+.2fG.png' % (i, d)), rgba)
        dets = dets[:len(signal)]
        np.savetxt(os.path.join(out, 'spectrum.txt'),
                   np.column_stack([dets, signal, temps]),
                   header='detuning[linewidths]  fluorescence_signal[arb]  T[K]')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(dets, signal, 'o-', color='crimson')
        ax.set_xlabel('cooling detuning (linewidths)')
        ax.set_ylabel('fluorescence signal (arb.)')
        ax.set_title('Rb-87 MOT fluorescence spectrum')
        ax.grid(True, ls='--', lw=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'spectrum.png'), dpi=140)
        plt.close('all')
        self._spec_status = 'done: %d frames + spectrum -> %s' % (len(frames), out)

    def _apply_config(self, s, a):
        with self.lock:
            self.sim.update_mot(power=dpg.get_value('mot_power'),
                                detuning=dpg.get_value('mot_detuning'),
                                radius=dpg.get_value('mot_radius'),
                                B_gradient=dpg.get_value('mot_grad'))
            self.sim.update_dipole(potential=dpg.get_value('dip_type'),
                                   wavelength=dpg.get_value('dip_wl') * 1e-9,
                                   waist=dpg.get_value('dip_waist') * 1e-6,
                                   depth_uK=dpg.get_value('dip_depth'))
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
                with dpg.child_window(width=self._panel_w, autosize_y=True):
                    with dpg.tab_bar():
                        self._tab_configuration(c)
                        self._tab_controls()
                        self._tab_report()
                        self._tab_spectroscopy()
                        self._tab_camera(c)
                with dpg.child_window(autosize_x=True, autosize_y=True,
                                      tag='right_panel'):
                    dpg.add_text('', tag='status', wrap=520)
                    dpg.add_image('cam_tex', width=512, height=512, tag='cam_image')
        dpg.set_primary_window('primary', True)

    def _on_resize(self):
        ''' Keep the camera image square and as large as fits, and wrap the
            status text to the panel width, whatever the window size. '''
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        right_w = max(140, vw - self._panel_w - 28)
        right_h = max(140, vh - 24)
        size = int(max(140, min(right_w - 8, right_h - 64)))
        dpg.configure_item('cam_image', width=size, height=size)
        dpg.configure_item('status', wrap=right_w - 8)

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
            dpg.add_combo(['gaussian', 'lattice', 'crossed'], label='potential',
                          tag='dip_type', default_value=c.dipole.potential)
            dpg.add_input_float(label='depth (uK)', tag='dip_depth',
                                default_value=c.dipole.depth_uK, step=50, format='%.0f')
            dpg.add_input_float(label='waist (um)', tag='dip_waist',
                                default_value=c.dipole.waist * 1e6, step=1, format='%.1f')
            dpg.add_input_float(label='wavelength (nm)', tag='dip_wl',
                                default_value=c.dipole.wavelength * 1e9, step=1, format='%.1f')
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
            dpg.add_checkbox(label='Imaging laser (illumination)', tag='chk_imaging',
                             default_value=False, callback=self._set_imaging)
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

    def _tab_report(self):
        with dpg.tab(label='Report'):
            dpg.add_text('Recapture & level report', color=(180, 180, 180))
            dpg.add_text('updates every %.0f s' % self.report_interval,
                         color=(120, 120, 120))
            dpg.add_separator()
            dpg.add_text('waiting for data...', tag='report_text', wrap=330)

    def _tab_spectroscopy(self):
        with dpg.tab(label='Spectroscopy'):
            dpg.add_text('Sweep the cooling detuning; save a frame per point',
                         color=(150, 150, 150), wrap=330)
            dpg.add_input_float(label='start (linewidths)', tag='spec_start',
                                default_value=-4.0, step=0.5, format='%.2f')
            dpg.add_input_float(label='stop (linewidths)', tag='spec_stop',
                                default_value=1.0, step=0.5, format='%.2f')
            dpg.add_input_int(label='points', tag='spec_n', default_value=21, step=1)
            dpg.add_input_int(label='settle (frames/point)', tag='spec_settle',
                              default_value=40, step=5)
            dpg.add_spacer(height=6)
            dpg.add_button(label='Run spectroscopy', callback=self._run_spec_clicked,
                           width=-1, height=34)
            dpg.add_text('saved to spectroscopy_output/', color=(120, 120, 120), wrap=330)
            dpg.add_text('', tag='spec_status', wrap=330)

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
            report = self._report
        if report is not None:
            self._render_report(report)
        dpg.set_value('spec_status', self._spec_status)
        if pub is not None:
            X, weight, st = pub
            dpg.set_value('cam_tex', self.cam.render_flat(X, weight))
            self._frame_times.append(t0)
            self._frame_times = self._frame_times[-30:]
            fps = 0.0
            if len(self._frame_times) > 1:
                fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0] + 1e-9)
            # the model-error indicator is the MOT linearisation error; it is
            # only meaningful while the cooling beams are on. The dipole force is
            # computed exactly, so it has no such error.
            if not st['cooling']:
                model_txt = 'exact (dipole)' if st['dipole'] else '--'
            else:
                err = st['model_error']
                model_txt = 'n/a' if err != err else '%.1f%%' % (err * 100)
            dpg.set_value('status',
                          'sim t=%7.1f ms   T(view)=%6.1f uK   rms=%.2f mm   fluor=%.3f\n'
                          'cooling=%s  repumper=%s  dipole=%s  imaging=%s   MOT model=%s   %.0f fps'
                          % (st['time_ms'], st['T_uK'], st['rms_mm'], st['fluorescence'],
                             st['cooling'], st['repumper'], st['dipole'], st['imaging'],
                             model_txt, fps))

    def _render_report(self, r):
        txt = (
            'Dipole trap: %s,  depth = %.0f uK  [%s]\n'
            '\n'
            'Recaptured: %d / %d atoms  (%.1f%%)\n'
            '  T (all cloud)   = %.1f uK\n'
            '  T (trapped)     = %.1f uK\n'
            '\n'
            'Level populations:\n'
            '  F=2  (bright)   = %5.1f %%\n'
            "  F'=3 (excited)  = %5.1f %%\n"
            '  F=1  (dark)     = %5.1f %%\n'
            '  bright fraction = %5.1f %%'
            % (r['dipole_type'], r['dipole_depth_uK'],
               'ON' if r['dipole_on'] else 'off',
               r['n_trapped'], r['n_total'], 100 * r['trapped_fraction'],
               r['T_all'] * 1e6, r['T_trapped'] * 1e6,
               100 * r['pop_F2'], 100 * r['pop_excited'], 100 * r['pop_dark'],
               100 * r['bright_fraction']))
        dpg.set_value('report_text', txt)

    # ------------------------------------------------------------- run ----
    def run(self, smoke_frames=0, screenshot=None):
        dpg.create_context()
        self._build_ui()
        dpg.create_viewport(title='MidnightTrapsfRb - optical trap cell', width=920, height=620)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(lambda *a: self._on_resize())
        self._on_resize()

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
