# app_trame_rsm.py
import os
import argparse
import threading
import time
from types import SimpleNamespace

import numpy as np
import xrayutilities as xu

# ── trame (force Vue2 only if your plugins are v1) ────────────────────────────
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify as v
from trame.widgets import html as h
from trame.widgets import vtk as wvtk
from trame.app import get_server

# If your stack is Vue2, keep client_type="vue2". If you upgraded to Vue3 plugins,
# switch to: server = get_server()
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# Vuetify aliases to support both naming schemes
VSelect     = getattr(v, "VSelect",     getattr(v, "Select", None))
VSwitch     = getattr(v, "VSwitch",     getattr(v, "Switch", None))
VTextField  = getattr(v, "VTextField",  getattr(v, "TextField", None))
VDivider    = getattr(v, "VDivider",    getattr(v, "Divider", None))
VBtn        = getattr(v, "VBtn",        getattr(v, "Btn", None))
VSpacer     = getattr(v, "VSpacer",     getattr(v, "Spacer", None))
VContainer  = getattr(v, "VContainer",  getattr(v, "Container", None))
VRow        = getattr(v, "VRow",        getattr(v, "Row", None))
VCol        = getattr(v, "VCol",        getattr(v, "Col", None))

# Bail early if aliases missing (prevents white page)
for _name, _w in dict(
    VSelect=VSelect, VSwitch=VSwitch, VTextField=VTextField,
    VDivider=VDivider, VBtn=VBtn, VSpacer=VSpacer,
    VContainer=VContainer, VRow=VRow, VCol=VCol,
).items():
    if _w is None:
        raise RuntimeError(f"Vuetify widget '{_name}' not available in this stack.")

# ── VTK imports / fallbacks ───────────────────────────────────────────────────
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkVolume, vtkVolumeProperty
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

# ── Your RSMBuilder (must be importable) ──────────────────────────────────────
from rsm3d.rsm3d import RSMBuilder

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Worker: stream frames → gridder → VTK ─────────────────────────────────────
def stream_process_and_update(
    server,
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
    state = server.state
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
                server.call_in_idle(_push)
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
            server.call_in_idle(_final)

    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        state.status = "Error"
    finally:
        running_flag["value"] = False
        state.running = False

# ── App builder ───────────────────────────────────────────────────────────────
def build_app(spec_file, tiff_dir):
    # VTK scene (raw objects kept in backend)
    ren = vtkRenderer()
    rw  = vtkRenderWindow()
    rw.AddRenderer(ren)

    image  = vtkImageData()
    mapper = make_volume_mapper()
    mapper.SetInputData(image)

    prop = vtkVolumeProperty()
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOff()
    prop.SetScalarOpacityUnitDistance(0.5)

    volume = vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    ren.AddVolume(volume)
    ren.SetBackground(0.10, 0.10, 0.12)

    # Seed tiny volume so canvas isn’t blank
    init_n = 8
    init_grid = np.zeros((init_n, init_n, init_n), dtype=np.float32)
    axes = (np.arange(init_n, dtype=float),) * 3
    vtk_set_image_from_grid(image, init_grid, axes)
    ren.ResetCamera()

    backend = SimpleNamespace(
        ren=ren, rw=rw, image=image, mapper=mapper, volume=volume,
        running_flag={"value": False}, view_update=lambda: None,
    )

    # UI state
    state.space = "hkl"
    state.normalize = "mean"
    state.fuzzy = False
    state.width = 0.0
    state.nx = 128
    state.ny = 128
    state.nz = 128
    state.update_every = 5
    state.running = False
    state.frame = 0
    state.elapsed = 0.0
    state.clim = (0.0, 1.0)
    state.status = "Idle"
    state.error = ""

    # Build data lazily when user clicks Start (prevents blocking/white page)
    builder_holder = {"obj": None}

    def ensure_builder():
        if builder_holder["obj"] is None:
            state.status = "Loading data…"
            b = RSMBuilder(
                spec_file, tiff_dir,
                ub_includes_2pi=True,
                center_is_one_based=False,
                motor_map={"omega": "th", "chi": "chi", "phi": "phi", "theta": "tth"},
                dtype=np.float32,
            )
            b.compute_full(verbose=True)
            builder_holder["obj"] = b
            state.status = "Data ready"

    def start_stream(*_):
        if state.running:
            return
        try:
            ensure_builder()
        except Exception as e:
            state.error = f"Builder error: {e}"
            state.status = "Error"
            return

        state.running = True
        backend.running_flag["value"] = True
        state.frame = 0
        state.elapsed = 0.0
        state.error = ""

        grid_shape = (int(state.nx), int(state.ny), int(state.nz))
        fuzzy = bool(state.fuzzy)
        width = None if (not fuzzy or float(state.width) <= 0) else float(state.width)

        t = threading.Thread(
            target=stream_process_and_update,
            args=(server, backend, builder_holder["obj"]),
            kwargs=dict(
                space=state.space,
                grid_shape=grid_shape,
                normalize=state.normalize,
                fuzzy=fuzzy,
                width=width,
                update_every=int(state.update_every),
            ),
            daemon=True,
        )
        t.start()

    def stop_stream(*_):
        backend.running_flag["value"] = False
        state.running = False

    ctrl.start_stream = start_stream
    ctrl.stop_stream  = stop_stream

    # ── UI layout ─────────────────────────────────────────────────────────────
    with SinglePageLayout(server) as layout:
        layout.title.set_text("RSM Stream (trame + VTK) — FourC")

        with layout.toolbar:
            VSelect(v_model=("space",), items=(["hkl", "q"]), label="Space",
                    dense=True, hide_details=True, style="max-width: 120px")
            VSelect(v_model=("normalize",), items=(["mean", "sum"]), label="Normalize",
                    dense=True, hide_details=True, style="max-width: 140px")
            VSwitch(v_model=("fuzzy",), label="Fuzzy", hide_details=True, dense=True)
            VTextField(v_model=("width",), type="number", label="Width (fuzzy)",
                       dense=True, hide_details=True, style="max-width: 140px")
            VDivider(vertical=True, classes="mx-2")
            VTextField(v_model=("nx",), type="number", label="Nx", dense=True, hide_details=True, style="max-width: 90px")
            VTextField(v_model=("ny",), type="number", label="Ny", dense=True, hide_details=True, style="max-width: 90px")
            VTextField(v_model=("nz",), type="number", label="Nz", dense=True, hide_details=True, style="max-width: 90px")
            VTextField(v_model=("update_every",), type="number", label="Update every N frames",
                       dense=True, hide_details=True, style="max-width: 200px")
            VDivider(vertical=True, classes="mx-2")
            VBtn("Start", click=ctrl.start_stream, disabled=("running",))
            VBtn("Stop",  click=ctrl.stop_stream,  disabled=("not running",))
            VBtn("Render", click=lambda *_: backend.view_update(), classes="ml-2")
            VSpacer()
            h.Div("Status: {{ status }} — Frame: {{ frame }} — Elapsed: {{ elapsed.toFixed(1) }} s",
                  style="font-weight: 500;")
            h.Div("{{ error }}", style="color: #ff6b6b;")

        with layout.content:
            # Give the container height; create view and expose only a callable
            with VContainer(fluid=True, classes="pa-0", style="height: calc(100vh - 120px);"):
                with VRow(classes="fill-height", style="height: 100%;"):
                    with VCol(classes="fill-height"):
                        view = wvtk.VtkRemoteView(backend.rw, interactive_ratio=1, ref="view")
                        backend.view_update = view.update
                        view.update()   # push initial empty frame
                        view

        with layout.footer:
            h.Div("CLim: {{ clim[0].toFixed(3) }} … {{ clim[1].toFixed(3) }}")

    return server

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to SPEC file")
    ap.add_argument("--tiff", required=True, help="Directory of TIFF frames")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 1234)))
    ap.add_argument("--server", action="store_true", help="Don't open a browser.")
    args = ap.parse_args()

    app = build_app(args.spec, args.tiff)
    app.start(address=args.host, port=args.port, open_browser=not args.server)

if __name__ == "__main__":
    main()