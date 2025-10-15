#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trame>=3.10",
#     "trame-components>=2.5",
#     "trame-vtklocal",
#     "trame-vuetify",
#     "vtk==9.5.0rc2",
#     "xrayutilities",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vtk.org"
# ///
import vtk

from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vtklocal, trame as tw, vuetify3 as v3, html as h
from trame.decorators import change
from trame.assets.remote import HttpFile  # optional utility; kept for parity with example
from trame.assets.local import to_url     # optional utility; kept for parity with example

# -----------------------------------------------------------------------------
# Domain imports
# -----------------------------------------------------------------------------
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import xrayutilities as xu

# Your RSMBuilder (must be importable on PYTHONPATH)
from rsm3d.rsm3d import RSMBuilder


# -----------------------------------------------------------------------------
# App constants / initial state
# -----------------------------------------------------------------------------
INITIAL_STATE = {
    "trame__title": "RSM Stream (Vue3)",
    # Add your own favicon if desired: "trame__favicon": to_url("/path/to/icon.png"),
    "space": "hkl",
    "normalize": "mean",
    "fuzzy": False,
    "width": 0.0,
    "nx": 128,
    "ny": 128,
    "nz": 128,
    "update_every": 5,
    "running": False,
    "frame": 0,
    "elapsed": 0.0,
    "clim": (0.0, 1.0),
    "status": "Idle",
    "error": "",
    # UI-driven paths (no CLI)
    "spec_path": "",
    "tiff_dir": "",
}


# -----------------------------------------------------------------------------
# VTK helpers
# -----------------------------------------------------------------------------
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkRenderWindow, vtkVolume, vtkVolumeProperty,
)
try:
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
    HAVE_SMART = True
except Exception:
    HAVE_SMART = False
try:
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkGPUVolumeRayCastMapper
    HAVE_GPU = True
except Exception:
    HAVE_GPU = False
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper as vtkCPUVolumeMapper
from vtkmodules.util.numpy_support import numpy_to_vtk


def make_volume_mapper():
    if HAVE_SMART:
        return vtkSmartVolumeMapper()
    if HAVE_GPU:
        return vtkGPUVolumeRayCastMapper()
    return vtkCPUVolumeMapper()


def vtk_set_image_from_grid(image: vtkImageData, grid: np.ndarray, axes):
    nx, ny, nz = [int(len(a)) for a in axes]
    if grid.shape != (nx, ny, nz):
        raise ValueError(f"grid shape {grid.shape} != ({nx},{ny},{nz})")
    xax, yax, zax = axes
    dx = float(xax[1] - xax[0]) if nx > 1 else 1.0
    dy = float(yax[1] - yax[0]) if ny > 1 else 1.0
    dz = float(zax[1] - zax[0]) if nz > 1 else 1.0
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(dx, dy, dz)
    image.SetOrigin(float(xax[0]), float(yax[0]), float(zax[0]))
    scalars = numpy_to_vtk(grid.ravel(order="F"), deep=False)
    scalars.SetName("Intensity")
    image.GetPointData().SetScalars(scalars)
    image.Modified()


def robust_percentiles(a: np.ndarray, lo=1.0, hi=99.5):
    a = np.asarray(a, dtype=np.float32)
    m = np.isfinite(a)
    if not m.any():
        return 0.0, 1.0
    p0, p1 = np.percentile(a[m], [lo, hi])
    if p0 == p1:
        p1 = p0 + 1e-3
    return float(p0), float(p1)


# -----------------------------------------------------------------------------
# Build VTK scene
# -----------------------------------------------------------------------------

def create_vtk_volume_scene():
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.OffScreenRenderingOn()

    image = vtkImageData()
    mapper = make_volume_mapper()
    mapper.SetInputData(image)

    prop = vtkVolumeProperty()
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOff()
    prop.SetScalarOpacityUnitDistance(0.5)

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer.AddVolume(volume)
    renderer.SetBackground(0.10, 0.10, 0.12)

    # Seed tiny volume so canvas isn’t blank
    init_n = 8
    init_grid = np.zeros((init_n, init_n, init_n), dtype=np.float32)
    axes = (np.arange(init_n, dtype=float),) * 3
    vtk_set_image_from_grid(image, init_grid, axes)

    render_window.Render()
    renderer.ResetCamera()

    return render_window, renderer, image, mapper, volume


# -----------------------------------------------------------------------------
# Worker: stream frames → gridder → VTK
# -----------------------------------------------------------------------------

def stream_process_and_update(
    app: TrameApp,
    backend: SimpleNamespace,
    builder: RSMBuilder,
    *,
    space="hkl",
    grid_shape=(128, 128, 128),
    normalize="mean",
    fuzzy=False,
    width=None,
    update_every=5,
):
    state = app.state
    image = backend.image
    ren   = backend.ren
    running_flag = backend.running_flag

    try:
        nx, ny, nz = map(int, grid_shape)
        G = (xu.FuzzyGridder3D if fuzzy else xu.Gridder3D)(nx, ny, nz)
        G.KeepData(True)

        # fix range to stabilize axes
        arr = builder.hkl if space == "hkl" else builder.Q_samp
        mins = arr.reshape(-1, 3).min(0)
        maxs = arr.reshape(-1, 3).max(0)
        try:
            G.dataRange(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2], fixed=True)
        except TypeError:
            G.dataRange(mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])

        n_frames = builder.intensity.shape[0]
        last_update = -1
        t0 = time.time()
        G.Normalize(False)

        state.status = f"Streaming {n_frames} frames…"
        for i in range(n_frames):
            if not running_flag["value"]:
                break

            coords = (builder.hkl if space == "hkl" else builder.Q_samp)[i]
            Xi = coords[..., 0].ravel()
            Yi = coords[..., 1].ravel()
            Zi = coords[..., 2].ravel()
            Wi = builder.intensity[i].ravel()

            if fuzzy and (width is not None):
                G(Xi, Yi, Zi, Wi, width=width)
            else:
                G(Xi, Yi, Zi, Wi)

            if (i - last_update) >= int(update_every) or (i == n_frames - 1):
                last_update = i
                grid_sum = np.array(G.data, copy=True, dtype=np.float32)
                xax, yax, zax = G.xaxis, G.yaxis, G.zaxis

                def _push():
                    grid_view = grid_sum / max(1, i + 1) if normalize.lower() == "mean" else grid_sum
                    vtk_set_image_from_grid(image, grid_view, (xax, yax, zax))
                    lo, hi = robust_percentiles(grid_view)
                    state.clim = (lo, hi)
                    state.frame = i + 1
                    state.elapsed = time.time() - t0
                    state.status = f"Updated frame {i+1}/{n_frames}"
                    backend.view_update()
                    if i == 0:
                        ren.ResetCamera()
                        backend.view_update()

                app.call_in_idle(_push)
            time.sleep(0)

        if running_flag["value"] and normalize.lower() == "mean":
            G.Normalize(True)
            grid = np.array(G.data, copy=True, dtype=np.float32)
            xax, yax, zax = G.xaxis, G.yaxis, G.zaxis

            def _final():
                vtk_set_image_from_grid(image, grid, (xax, yax, zax))
                lo, hi = robust_percentiles(grid)
                state.clim = (lo, hi)
                state.frame = n_frames
                state.elapsed = time.time() - t0
                state.status = f"Done. Frames: {n_frames}"
                backend.view_update()

            app.call_in_idle(_final)

    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        state.status = "Error"
    finally:
        running_flag["value"] = False
        state.running = False


# -----------------------------------------------------------------------------
# Trame app (Vue3 style)
# -----------------------------------------------------------------------------
class App(TrameApp):
    def __init__(self, server=None):
        super().__init__(server)

        # VTK setup
        self.rw, self.ren, self.image, self.mapper, self.volume = create_vtk_volume_scene()
        self.backend = SimpleNamespace(
            ren=self.ren,
            rw=self.rw,
            image=self.image,
            mapper=self.mapper,
            volume=self.volume,
            running_flag={"value": False},
            view_update=lambda: None,
        )

        # GUI + state
        self._build_ui()
        self.state.update(INITIAL_STATE)

    # --- Reactive helpers (optional) -----------------------------------------
    @change("fuzzy")
    def _on_fuzzy(self, fuzzy, **_):
        if not fuzzy:
            self.state.width = 0.0

    # --- Controls -------------------------------------------------------------
    def _ensure_builder(self):
        s = self.state
        spec_file = (s.spec_path or "").strip()
        tiff_dir  = (s.tiff_dir or "").strip()
        if not spec_file:
            s.error = "Please provide a SPEC file path."
            s.status = "Missing path"
            raise RuntimeError(s.error)
        if not tiff_dir:
            s.error = "Please provide a TIFF directory path."
            s.status = "Missing path"
            raise RuntimeError(s.error)
        if not os.path.exists(spec_file):
            s.error = f"SPEC file not found: {spec_file}"
            s.status = "Path error"
            raise FileNotFoundError(s.error)
        if not os.path.isdir(tiff_dir):
            s.error = f"TIFF directory not found: {tiff_dir}"
            s.status = "Path error"
            raise NotADirectoryError(s.error)

        if not hasattr(self, "_builder") or (self._builder is None):
            s.status = "Loading data…"
            b = RSMBuilder(
                spec_file, tiff_dir,
                ub_includes_2pi=True,
                center_is_one_based=False,
                motor_map={"omega": "th", "chi": "chi", "phi": "phi", "theta": "tth"},
                dtype=np.float32,
            )
            b.compute_full(verbose=True)
            self._builder = b
            s.status = "Data ready"

    def start_stream(self, *_):
        s = self.state
        if s.running:
            return
        try:
            self._ensure_builder()
        except Exception:
            return

        s.running = True
        self.backend.running_flag["value"] = True
        s.frame = 0
        s.elapsed = 0.0
        s.error = ""

        grid_shape = (int(s.nx), int(s.ny), int(s.nz))
        fuzzy = bool(s.fuzzy)
        width = None if (not fuzzy or float(s.width) <= 0) else float(s.width)

        t = threading.Thread(
            target=stream_process_and_update,
            args=(self, self.backend, self._builder),
            kwargs=dict(
                space=s.space,
                grid_shape=grid_shape,
                normalize=s.normalize,
                fuzzy=fuzzy,
                width=width,
                update_every=int(s.update_every),
            ),
            daemon=True,
        )
        t.start()

    def stop_stream(self, *_):
        self.backend.running_flag["value"] = False
        self.state.running = False

    def clear_builder(self, *_):
        self._builder = None
        self.state.status = "Idle"
        self.state.error = ""

    # --- UI ------------------------------------------------------------------
    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server, full_height=True) as layout:
            self.ui = layout

            # Toolbar
            with layout.toolbar as toolbar:
                toolbar.density = "compact"
                layout.title.set_text("RSM Stream (VTK)")
                v3.VTextField(v_model=("spec_path",), density="compact", label="SPEC file path", hide_details=True, style="max-width: 360px")
                v3.VTextField(v_model=("tiff_dir",),  density="compact", label="TIFF directory path", hide_details=True, style="max-width: 360px")
                v3.VBtn("Clear", density="comfortable", variant="tonal", class_="ml-2", click=self.clear_builder)
                v3.VDivider(vertical=True, class_="mx-2")

                v3.VSelect(v_model=("space",), items=["hkl", "q"], label="Space", density="compact", hide_details=True, style="max-width: 140px")
                v3.VSelect(v_model=("normalize",), items=["mean", "sum"], label="Normalize", density="compact", hide_details=True, style="max-width: 160px")
                v3.VSwitch(v_model=("fuzzy",), label="Fuzzy", density="compact", hide_details=True)
                v3.VTextField(v_model=("width",), type="number", label="Width (fuzzy)", density="compact", hide_details=True, style="max-width: 160px")
                v3.VDivider(vertical=True, class_="mx-2")
                v3.VTextField(v_model=("nx",), type="number", label="Nx", density="compact", hide_details=True, style="max-width: 90px")
                v3.VTextField(v_model=("ny",), type="number", label="Ny", density="compact", hide_details=True, style="max-width: 90px")
                v3.VTextField(v_model=("nz",), type="number", label="Nz", density="compact", hide_details=True, style="max-width: 90px")
                v3.VTextField(v_model=("update_every",), type="number", label="Update every N frames", density="compact", hide_details=True, style="max-width: 220px")

                v3.VDivider(vertical=True, class_="mx-2")
                v3.VBtn("Start", variant="flat", class_="ml-1", disabled=("running",), click=self.start_stream)
                v3.VBtn("Stop",  variant="tonal", class_="ml-1", disabled=("not running",), click=self.stop_stream)
                v3.VBtn("Render", variant="text", class_="ml-2", click=lambda *_: self.backend.view_update())
                v3.VSpacer()
                h.Div("Status: {{ status }} — Frame: {{ frame }} — Elapsed: {{ elapsed.toFixed(1) }} s", class_="font-weight-medium mr-3")
                h.Div("{{ error }}", style="color: #ff6b6b;")

            # Drawer (left empty for now; you can add advanced controls here)
            with layout.drawer:
                h.Div("Ready", class_="pa-2 text-medium-emphasis")

            # Content
            with layout.content:
                with vtklocal.LocalView(self.rw, throttle_rate=20) as view:
                    self.ctrl.view_update = view.update_throttle
                    self.ctrl.view_reset_camera = view.reset_camera

        # End of UI build


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    app = App()
    # Default host/port; open browser automatically
    app.server.start(address="127.0.0.1", port=1234, open_browser=True)


if __name__ == "__main__":
    main()
