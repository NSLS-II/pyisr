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

import os
import numpy as np
import vtk

from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vtklocal, vuetify3 as v3, html as h
from trame.decorators import change

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_mapper():
    if HAVE_SMART:
        return vtkSmartVolumeMapper()
    if HAVE_GPU:
        return vtkGPUVolumeRayCastMapper()
    return vtkCPUVolumeMapper()


def _robust_percentiles(a: np.ndarray, lo=1.0, hi=99.8):
    a = np.asarray(a, dtype=np.float32)
    m = np.isfinite(a)
    if not m.any():
        return 0.0, 1.0
    p0, p1 = np.percentile(a[m], [lo, hi])
    if p0 == p1:
        p1 = p0 + 1e-6
    return float(p0), float(p1)


def _to_image(image: vtkImageData, grid: np.ndarray, axes):
    """
    Load a (nx, ny, nz) cube into vtkImageData with name 'Intensity'.
    Use SetExtent + SetDimensions + SetSpacing and deep-copy scalars so VTK owns them.
    """
    nx, ny, nz = [int(len(a)) for a in axes]
    if grid.shape != (nx, ny, nz):
        raise ValueError(f"grid shape {grid.shape} != ({nx},{ny},{nz})")

    xax, yax, zax = axes

    def _step(ax):
        if len(ax) < 2:
            return 1.0
        d = np.diff(ax)
        return float(np.mean(d))

    dx, dy, dz = _step(xax), _step(yax), _step(zax)

    # Define extent explicitly, then spacing/origin
    image.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(dx, dy, dz)
    image.SetOrigin(float(xax[0]), float(yax[0]), float(zax[0]))

    # VTK likes X as the fastest axis; use Fortran order
    flat = np.asfortranarray(grid, dtype=np.float32).ravel(order="F")
    scalars = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    scalars.SetName("Intensity")

    pd = image.GetPointData()
    pd.SetScalars(scalars)            # becomes active automatically
    pd.SetActiveScalars("Intensity")
    image.Modified()


def _viridis_points():
    # A few key points from the Viridis colormap (0..1)
    return [
        (0.00, 0.267004, 0.004874, 0.329415),
        (0.10, 0.282327, 0.094955, 0.417331),
        (0.25, 0.253935, 0.265254, 0.529983),
        (0.50, 0.163625, 0.471133, 0.558148),
        (0.70, 0.134692, 0.658636, 0.517649),
        (0.85, 0.477504, 0.821444, 0.318195),
        (1.00, 0.993248, 0.906157, 0.143936),
    ]


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------
class RSMTrameViewer(TrameApp):
    def __init__(
        self,
        grid: np.ndarray,
        axes,
        *,
        space: str = "hkl",
        name: str = "RSM",
        log_view: bool = True,
        contrast_percentiles=(1.0, 99.8),
        cmap: str = "viridis",
        rendering: str = "attenuated_mip",
        server=None,
    ):
        super().__init__(server)

        self._raw_grid = np.asarray(grid, dtype=np.float32)
        self._axes = axes
        self._space = space
        self._name = name
        self._log_view = bool(log_view)
        self._p_lo, self._p_hi = map(float, contrast_percentiles)
        self._cmap = cmap
        self._rendering = rendering

        # VTK objects
        self.ren = vtkRenderer()
        self.rw = vtkRenderWindow()
        self.rw.AddRenderer(self.ren)
        self.rw.OffScreenRenderingOn()

        self.image = vtkImageData()
        self._init_empty_image()  # allocate dummy scalar BEFORE mapper connection

        self.mapper = _make_mapper()
        # Ensure we use point data
        if hasattr(self.mapper, "SetScalarModeToUsePointData"):
            self.mapper.SetScalarModeToUsePointData()
        else:
            try:
                self.mapper.SetScalarMode(vtk.VTK_SCALAR_MODE_USE_POINT_DATA)
            except Exception:
                pass
        self.mapper.SetInputData(self.image)  # connect immediately

        self.prop = vtkVolumeProperty()
        self.prop.SetInterpolationTypeToLinear()
        self.prop.ShadeOff()
        self.prop.SetScalarOpacityUnitDistance(0.5)

        self.volume = vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.prop)
        self.ren.AddVolume(self.volume)
        self.ren.SetBackground(0.10, 0.10, 0.12)

        # UI state
        s = self.state
        s.trame__title = name
        s.status = "Initializing…"
        s.error = ""
        s.space = space
        s.log_view = self._log_view
        s.p_lo = self._p_lo
        s.p_hi = self._p_hi
        s.rendering = self._rendering

        # UI
        self._build_ui()

        # Populate real data AFTER mapper is wired
        self._update_image_from_grid()
        self._apply_rendering_mode(s.rendering)
        self._apply_color_and_opacity()
        self.ren.ResetCamera()
        try:
            self.mapper.Update()
            self.rw.Render()
        except Exception:
            pass
        s.status = "Ready"

    def _init_empty_image(self):
        """Allocate a 1-voxel image so mapper has a valid input immediately."""
        self.image.SetExtent(0, 0, 0, 0, 0, 0)
        self.image.SetDimensions(1, 1, 1)
        self.image.SetSpacing(1.0, 1.0, 1.0)
        self.image.SetOrigin(0.0, 0.0, 0.0)
        self.image.AllocateScalars(vtk.VTK_FLOAT, 1)
        arr = self.image.GetPointData().GetScalars()
        if arr:
            arr.FillComponent(0, 0.0)
            arr.SetName("Intensity")

    # -------------------- data mapping ---------------------------------------
    def _data_view(self):
        a = self._raw_grid
        if self.state.log_view:
            a = np.log1p(np.maximum(0.0, a))
        return a

    def _update_image_from_grid(self):
        a = self._data_view()
        _to_image(self.image, a, self._axes)
        # Explicitly select our scalar array
        try:
            self.mapper.SelectScalarArray("Intensity")
        except Exception:
            try:
                self.mapper.SetRequestedArrayName("Intensity")
            except Exception:
                pass
        self.image.Modified()
        self.mapper.Modified()

    def _apply_color_and_opacity(self):
        a = self._data_view()
        lo, hi = _robust_percentiles(a, self.state.p_lo, self.state.p_hi)
        self.state.clim = (float(lo), float(hi))

        ctf = vtk.vtkColorTransferFunction()
        pts = _viridis_points() if (self._cmap or "").lower() == "viridis" else _viridis_points()
        for t, r, g, b in pts:
            ctf.AddRGBPoint(lo + t * (hi - lo), r, g, b)

        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(lo, 0.0)
        otf.AddPoint((lo + hi) / 2.0, 0.15)
        otf.AddPoint(hi, 1.0)

        self.prop.SetColor(ctf)
        self.prop.SetScalarOpacity(otf)
        self.prop.Modified()
        self.mapper.Modified()

    def _apply_rendering_mode(self, mode: str):
        m = (mode or "").lower()
        # Defaults
        self.mapper.SetBlendModeToComposite()
        self.prop.SetScalarOpacityUnitDistance(0.5)
        if m == "mip":
            try:
                self.mapper.SetBlendModeToMaximumIntensity()
            except Exception:
                self.mapper.SetBlendModeToComposite()
        elif m == "translucent":
            self.mapper.SetBlendModeToComposite()
            self.prop.SetScalarOpacityUnitDistance(0.5)
        else:  # "attenuated_mip"
            self.mapper.SetBlendModeToComposite()
            self.prop.SetScalarOpacityUnitDistance(0.15)

    # -------------------- reactive handlers ----------------------------------
    @change("log_view")
    def _on_toggle_log(self, log_view, **_):
        self._log_view = bool(log_view)
        self._update_image_from_grid()
        self._apply_color_and_opacity()
        self.ctrl.view_update()

    @change("p_lo", "p_hi")
    def _on_contrast(self, p_lo, p_hi, **_):
        self._p_lo, self._p_hi = float(p_lo), float(p_hi)
        self._apply_color_and_opacity()
        self.ctrl.view_update()

    @change("rendering")
    def _on_render_mode(self, rendering, **_):
        self._apply_rendering_mode(rendering)
        self.ctrl.view_update()

    # -------------------- UI -------------------------------------------------
    def _build_ui(self):
        with SinglePageWithDrawerLayout(self.server, full_height=True) as layout:
            self.ui = layout

            with layout.toolbar as _tb:
                layout.title.set_text(self._name)
                v3.VSelect(
                    v_model=("rendering",),
                    items=["attenuated_mip", "mip", "translucent"],
                    label="Rendering",
                    density="compact",
                    hide_details=True,
                    style="max-width: 220px",
                )
                v3.VSwitch(
                    v_model=("log_view",),
                    label="Log",
                    density="compact",
                    hide_details=True,
                )
                v3.VSlider(
                    v_model=("p_lo",),
                    min=0,
                    max=20,
                    step=0.1,
                    label="Plo",
                    density="compact",
                    hide_details=True,
                    style="max-width: 200px",
                )
                v3.VSlider(
                    v_model=("p_hi",),
                    min=80,
                    max=100,
                    step=0.1,
                    label="Phi",
                    density="compact",
                    hide_details=True,
                    style="max-width: 200px",
                )
                v3.VSpacer()
                h.Div(
                    "Space: {{ space }} — CLim: {{ clim && clim.length ? (clim[0].toFixed(3) + ' … ' + clim[1].toFixed(3)) : '' }}",
                    classes="mr-4 text-medium-emphasis",
                )

            with layout.drawer:
                h.Div("RSM viewer (trame)", classes="pa-2 text-medium-emphasis")

            with layout.content:
                # Use vtklocal.LocalView to match installed dependency (trame-vtklocal)
                with vtklocal.LocalView(self.rw, throttle_rate=20) as view:
                    # Use synchronous updater to avoid asyncio loop before server.start()
                    self.ctrl.view_update = view.update
                    self.ctrl.view_update_throttle = getattr(view, "update_throttle", view.update)
                    self.ctrl.view_reset_camera = view.reset_camera

    # -------------------- public API -----------------------------------------
    def launch(self, *, address: str = "127.0.0.1", port: int = 1234, open_browser: bool = True):
        """Start the trame server and return self (viewer-like)."""
        self.server.start(address=address, port=port, open_browser=open_browser)
        return self


# -----------------------------------------------------------------------------
# Example usage (drop-in)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with your real paths or set env SPEC / TIFF accordingly
    from rsm3d.rsm3d import RSMBuilder

    spec_file = os.environ.get("SPEC", "/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/setup_6oct23")
    tiff_dir = os.environ.get("TIFF", "/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/data_6oct23_tiff")
    scan_list = (17,)  # any list/tuple of scans

    builder = RSMBuilder(
        spec_file,
        tiff_dir,
        selected_scans=scan_list,
        ub_includes_2pi=True,
        center_is_one_based=False,
    )
    Q_samp, hkl, intensity = builder.compute_full()

    grid, (xax, yax, zax) = builder.regrid_xu(
        space="hkl",
        grid_shape=(100, 100, 100),
        fuzzy=True,
        normalize="mean",
        stream=False,  # streaming not wired in this minimal viewer
    )

    viz = RSMTrameViewer(
        grid,
        (xax, yax, zax),
        space="hkl",
        name="RSM (trame)",
        log_view=True,
        contrast_percentiles=(1.0, 99.8),
        cmap="viridis",
        rendering="attenuated_mip",
    )
    # Use open_browser=True if you want it to auto-open
    viewer = viz.launch(open_browser=False)

# #!/usr/bin/env -S uv run --script
# #
# # /// script
# # requires-python = ">=3.11"
# # dependencies = [
# #     "trame>=3.10",
# #     "trame-components>=2.5",
# #     "trame-vtklocal",
# #     "trame-vuetify",
# #     "vtk==9.5.0rc2",
# #     "xrayutilities",
# # ]
# #
# # [[tool.uv.index]]
# # url = "https://wheels.vtk.org"
# # ///

# import os
# from matplotlib import axes, image
# import numpy as np
# import vtk

# from trame.app import TrameApp
# from trame.ui.vuetify3 import SinglePageWithDrawerLayout
# from trame.widgets import vtk as vtkw, vuetify3 as v3, html as h
# from trame.decorators import change

# from vtkmodules.vtkCommonDataModel import vtkImageData
# from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkVolume, vtkVolumeProperty
# try:
#     from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
#     HAVE_SMART = True
# except Exception:
#     HAVE_SMART = False
# try:
#     from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkGPUVolumeRayCastMapper
#     HAVE_GPU = True
# except Exception:
#     HAVE_GPU = False
# from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper as vtkCPUVolumeMapper
# from vtkmodules.util.numpy_support import numpy_to_vtk


# # -------------------------- small helpers ------------------------------------

# def _make_mapper():
#     if HAVE_SMART:
#         return vtkSmartVolumeMapper()
#     if HAVE_GPU:
#         return vtkGPUVolumeRayCastMapper()
#     return vtkCPUVolumeMapper()


# def _robust_percentiles(a: np.ndarray, lo=1.0, hi=99.8):
#     a = np.asarray(a, dtype=np.float32)
#     m = np.isfinite(a)
#     if not m.any():
#         return 0.0, 1.0
#     p0, p1 = np.percentile(a[m], [lo, hi])
#     if p0 == p1:
#         p1 = p0 + 1e-6
#     return float(p0), float(p1)


# def _to_image(image: vtkImageData, grid: np.ndarray, axes):
#     """
#     Load a (nx, ny, nz) cube into vtkImageData with name 'Intensity'.
#     Uses SetExtent + SetDimensions + SetSpacing to avoid SmartVolumeMapper
#     failing to find the scalar array. Re‑sets the scalars each call.
#     """
#     nx, ny, nz = [int(len(a)) for a in axes]
#     if grid.shape != (nx, ny, nz):
#         raise ValueError(f"grid shape {grid.shape} != ({nx},{ny},{nz})")

#     xax, yax, zax = axes

#     def _step(ax):
#         if len(ax) < 2:
#             return 1.0
#         d = np.diff(ax)
#         return float(np.mean(d))

#     dx, dy, dz = _step(xax), _step(yax), _step(zax)

#     # Define extent explicitly (required for some VTK mappers)
#     image.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
#     image.SetDimensions(nx, ny, nz)
#     image.SetSpacing(dx, dy, dz)
#     image.SetOrigin(float(xax[0]), float(yax[0]), float(zax[0]))

#     flat = np.ascontiguousarray(grid.ravel(order="C"), dtype=np.float32)
#     scalars = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
#     scalars.SetName("Intensity")

#     pd = image.GetPointData()
#     pd.SetScalars(scalars)            # becomes active automatically
#     # (Re)affirm active scalars for safety
#     pd.SetActiveScalars("Intensity")
#     image.Modified()


# def _viridis_points():
#     # A few key points from the Viridis colormap (0..1)
#     return [
#         (0.00, 0.267004, 0.004874, 0.329415),
#         (0.10, 0.282327, 0.094955, 0.417331),
#         (0.25, 0.253935, 0.265254, 0.529983),
#         (0.50, 0.163625, 0.471133, 0.558148),
#         (0.70, 0.134692, 0.658636, 0.517649),
#         (0.85, 0.477504, 0.821444, 0.318195),
#         (1.00, 0.993248, 0.906157, 0.143936),
#     ]

# class RSMTrameViewer(TrameApp):

#     def __init__(
#         self,
#         grid: np.ndarray,
#         axes,
#         *,
#         space: str = "hkl",
#         name: str = "RSM",
#         log_view: bool = True,
#         contrast_percentiles = (1.0, 99.8),
#         cmap: str = "viridis",
#         rendering: str = "attenuated_mip",
#         server=None,
#     ):
#         super().__init__(server)

#         self._raw_grid = np.asarray(grid, dtype=np.float32)
#         self._axes = axes
#         self._space = space
#         self._name = name
#         self._log_view = bool(log_view)
#         self._p_lo, self._p_hi = map(float, contrast_percentiles)
#         self._cmap = cmap
#         self._rendering = rendering

#         # --- VTK objects
#         self.ren = vtkRenderer()
#         self.rw  = vtkRenderWindow()
#         self.rw.AddRenderer(self.ren)
#         self.rw.OffScreenRenderingOn()

#         self.image  = vtkImageData()
#         self._init_empty_image()          # NEW: allocate dummy scalar BEFORE mapper connection

#         self.mapper = _make_mapper()
#         if hasattr(self.mapper, "SetScalarModeToUsePointData"):
#             self.mapper.SetScalarModeToUsePointData()
#         elif hasattr(self.mapper, "SetScalarMode"):
#             try:
#                 self.mapper.SetScalarMode(vtk.VTK_SCALAR_MODE_USE_POINT_DATA)
#             except Exception:
#                 pass
#         self.mapper.SetInputData(self.image)   # connect immediately (prevents 0-connection error)

#         self.prop = vtkVolumeProperty()
#         self.prop.SetInterpolationTypeToLinear()
#         self.prop.ShadeOff()
#         self.prop.SetScalarOpacityUnitDistance(0.5)

#         self.volume = vtkVolume()
#         self.volume.SetMapper(self.mapper)
#         self.volume.SetProperty(self.prop)
#         self.ren.AddVolume(self.volume)
#         self.ren.SetBackground(0.10, 0.10, 0.12)

#         # --- State / UI
#         s = self.state
#         s.trame__title = name
#         s.status = "Initializing…"
#         s.error = ""
#         s.space = space
#         s.log_view = self._log_view
#         s.p_lo = self._p_lo
#         s.p_hi = self._p_hi
#         s.rendering = self._rendering

#         self._build_ui()

#         # Populate real data AFTER mapper is already wired
#         self._update_image_from_grid()
#         self._apply_rendering_mode(s.rendering)
#         self._apply_color_and_opacity()
#         self.ren.ResetCamera()
#         try:
#             self.mapper.Update()
#             self.rw.Render()
#         except Exception:
#             pass
#         s.status = "Ready"

#     # NEW helper
#     def _init_empty_image(self):
#         """Allocate a 1-voxel image so mapper has a valid input immediately."""
#         self.image.SetExtent(0, 0, 0, 0, 0, 0)
#         self.image.SetDimensions(1, 1, 1)
#         self.image.SetSpacing(1.0, 1.0, 1.0)
#         self.image.SetOrigin(0.0, 0.0, 0.0)
#         self.image.AllocateScalars(vtk.VTK_FLOAT, 1)
#         arr = self.image.GetPointData().GetScalars()
#         if arr:
#             arr.FillComponent(0, 0.0)
#             arr.SetName("Intensity")

#     # -------------------- data mapping ---------------------------------------
#     def _data_view(self):
#         a = self._raw_grid
#         if self.state.log_view:
#             a = np.log1p(np.maximum(0.0, a))
#         return a

#     def _update_image_from_grid(self):
#         a = self._data_view()
#         _to_image(self.image, a, self._axes)
#         # Do NOT reset SetInputData (connection already established)
#         if hasattr(self.mapper, "SetRequestedArrayName"):
#             self.mapper.SetRequestedArrayName("Intensity")
#         elif hasattr(self.mapper, "SelectScalarArray"):
#             self.mapper.SelectScalarArray("Intensity")
#         self.image.Modified()
#         self.mapper.Modified()

#     def _apply_color_and_opacity(self):
#         a = self._data_view()
#         lo, hi = _robust_percentiles(a, self.state.p_lo, self.state.p_hi)
#         self.state.clim = (float(lo), float(hi))

#         ctf = vtk.vtkColorTransferFunction()
#         if (self._cmap or "").lower() == "viridis":
#             pts = _viridis_points()
#         else:
#             pts = _viridis_points()
#         for t, r, g, b in pts:
#             ctf.AddRGBPoint(lo + t*(hi-lo), r, g, b)

#         otf = vtk.vtkPiecewiseFunction()
#         otf.AddPoint(lo, 0.0)
#         otf.AddPoint((lo+hi)/2.0, 0.15)
#         otf.AddPoint(hi, 1.0)

#         self.prop.SetColor(ctf)
#         self.prop.SetScalarOpacity(otf)
#         self.prop.Modified()
#         self.mapper.Modified()

#     def _apply_rendering_mode(self, mode: str):
#         m = (mode or "").lower()
#         # Defaults
#         self.mapper.SetBlendModeToComposite()
#         self.prop.SetScalarOpacityUnitDistance(0.5)
#         if m == "mip":
#             # Maximum intensity projection
#             try:
#                 self.mapper.SetBlendModeToMaximumIntensity()
#             except Exception:
#                 self.mapper.SetBlendModeToComposite()
#         elif m == "translucent":
#             self.mapper.SetBlendModeToComposite()
#             self.prop.SetScalarOpacityUnitDistance(0.5)
#         else:  # "attenuated_mip" (approximation using composite + shorter unit distance)
#             self.mapper.SetBlendModeToComposite()
#             self.prop.SetScalarOpacityUnitDistance(0.15)

#     # -------------------- reactive handlers ----------------------------------
#     @change("log_view")
#     def _on_toggle_log(self, log_view, **_):
#         self._log_view = bool(log_view)
#         self._update_image_from_grid()
#         self._apply_color_and_opacity()
#         self.ctrl.view_update()

#     @change("p_lo", "p_hi")
#     def _on_contrast(self, p_lo, p_hi, **_):
#         self._p_lo, self._p_hi = float(p_lo), float(p_hi)
#         self._apply_color_and_opacity()
#         self.ctrl.view_update()

#     @change("rendering")
#     def _on_render_mode(self, rendering, **_):
#         self._apply_rendering_mode(rendering)
#         self.ctrl.view_update()

#     # -------------------- UI -------------------------------------------------
#     def _build_ui(self):
#         with SinglePageWithDrawerLayout(self.server, full_height=True) as layout:
#             self.ui = layout

#             with layout.toolbar as tb:
#                 layout.title.set_text(self._name)
#                 v3.VSelect(
#                     v_model=("rendering",),
#                     items=["attenuated_mip", "mip", "translucent"],
#                     label="Rendering",
#                     density="compact",
#                     hide_details=True,
#                     style="max-width: 220px",
#                 )
#                 v3.VSwitch(v_model=("log_view",), label="Log", density="compact", hide_details=True)
#                 v3.VSlider(v_model=("p_lo",), min=0, max=20, step=0.1, label="Plo", density="compact", hide_details=True, style="max-width: 200px")
#                 v3.VSlider(v_model=("p_hi",), min=80, max=100, step=0.1, label="Phi", density="compact", hide_details=True, style="max-width: 200px")
#                 v3.VSpacer()
#                 h.Div("Space: {{ space }} — CLim: {{ clim && clim.length ? (clim[0].toFixed(3) + ' … ' + clim[1].toFixed(3)) : '' }}", classes="mr-4 text-medium-emphasis")

#             with layout.drawer:
#                 h.Div("RSM viewer (trame)", classes="pa-2 text-medium-emphasis")

#             with layout.content:
#                 with vtkw.VtkRemoteView(self.rw, interactive_ratio=1.0) as view:
                  
                  
#                     # Use synchronous updater to avoid asyncio loop before server.start()
#                     self.ctrl.view_update = view.update
#                     self.ctrl.view_update_throttle = view.update   # keep API uniform
#                     self.ctrl.view_reset_camera = view.reset_camera

#     # -------------------- public API -----------------------------------------
#     def launch(self, *, address: str = "127.0.0.1", port: int = 1234, open_browser: bool = True):
#         """Start the trame server and return self (viewer-like)."""
#         self.server.start(address=address, port=port, open_browser=open_browser)
#         return self


# # ---------------------- Example usage (drop-in) ------------------------------
# if __name__ == "__main__":
#     # Example showing how to replace RSMNapariViewer in your workflow
#     from rsm3d.rsm3d import RSMBuilder

#     spec_file = os.environ.get("SPEC", "/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/setup_6oct23")
#     tiff_dir  = os.environ.get("TIFF", "/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/data_6oct23_tiff")
#     scan_list = (17,)  # any list/tuple of scans

#     builder = RSMBuilder(
#         spec_file, tiff_dir,
#         selected_scans=scan_list,
#         ub_includes_2pi=True,
#         center_is_one_based=False,
#     )
#     Q_samp, hkl, intensity = builder.compute_full()

#     grid, (xax, yax, zax) = builder.regrid_xu(
#         space="hkl",
#         grid_shape=(100, 100, 100),
#         fuzzy=True,
#         normalize="mean",
#         stream=False,  # streaming not wired in this minimal viewer
#     )

#     viz = RSMTrameViewer(
#         grid, (xax, yax, zax),
#         space="hkl",
#         name="RSM (trame)",
#         log_view=True,
#         contrast_percentiles=(1.0, 99.8),
#         cmap="viridis",
#         rendering="attenuated_mip",
#     )
#     viewer = viz.launch(open_browser=False)
