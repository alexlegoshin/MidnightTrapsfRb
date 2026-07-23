# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build spec -- produces ONE self-contained MidnightTrapsfRb.exe.
# Build with:
#
#     pyinstaller MidnightTrapsfRb.spec
#
# CuPy/GPU support is excluded from the frozen build: MOTorNOT's backend
# falls back to NumPy cleanly when CuPy isn't importable (it's wrapped in a
# broad try/except), the real-time engine already runs on the CPU via its
# linearised MOT model, and bundling CuPy would make the exe both huge and
# tied to the exact CUDA/driver stack of the build machine.

excludes = [
    'cupy', 'cupyx', 'cupy_backends', 'fastrlock',
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'notebook', 'pytest',
]

# MOTorNOT ships parameters.yml as package data; the app doesn't call
# load_parameters() on its own atom path, but bundle it anyway so anything
# that does still finds it in the frozen build.
motornot_data = [('../MOTorNOT/MOTorNOT/parameters.yml', 'MOTorNOT')]

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=motornot_data,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MidnightTrapsfRb',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # GUI app -- no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
