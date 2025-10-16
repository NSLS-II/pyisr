#     "trame>=3.10",
#     "trame-vtklocal",
#     "trame-vuetify",
#     "vtk==9.5.0rc2",
# ]

from pathlib import Path
from trame.app import TrameApp
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vtklocal, vuetify3 as v3, html as h

import vtk
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
from vtkmodules.vtkCommonCore import vtkFloatArray

# ---------------------------------------------------------------------
# Hard‑coded path to your .vtr file
VTR_PATH = "/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/rsm_hkl.vtr"
# ---------------------------------------------------------------------


class VTRViewer(TrameApp):
    def __init__(self, vtr_path: str, server=None):
        super().__init__(server)
        self._vtr_path = vtr_path
        self._setup_vtk()
        self._build_ui()

    # -----------------------------------------------------------------
    def _ensure_point_scalars(self, dataset):
        """
        Ensure a point-data scalar array exists:
          1) If point arrays exist: activate first if none active.
          2) Else if cell arrays exist: convert cell->point.
          3) Else synthesize a zero scalar array.
        """
        pd = dataset.GetPointData()
        if pd and pd.GetNumberOfArrays() > 0:
            if pd.GetScalars() is None:
                pd.SetActiveScalars(pd.GetArrayName(0))
            return dataset

        cd = dataset.GetCellData()
        if cd and cd.GetNumberOfArrays() > 0:
            if cd.GetScalars() is None:
                cd.SetActiveScalars(cd.GetArrayName(0))
            c2p = vtkCellDataToPointData()
            c2p.SetInputData(dataset)
            c2p.PassCellDataOn()
            c2p.Update()
            out = c2p.GetOutput()
            opd = out.GetPointData()
            if opd.GetScalars() is None and opd.GetNumberOfArrays() > 0:
                opd.SetActiveScalars(opd.GetArrayName(0))
            return out

        # Synthesize
        arr = vtkFloatArray()
        arr.SetName("synthetic")
        npts = dataset.GetNumberOfPoints()
        arr.SetNumberOfValues(npts)
        for i in range(npts):
            arr.SetValue(i, 0.0)
        dataset.GetPointData().AddArray(arr)
        dataset.GetPointData().SetActiveScalars("synthetic")
        return dataset

    # -----------------------------------------------------------------
    def _setup_vtk(self):
        p = Path(self._vtr_path)
        if not p.is_file():
            raise FileNotFoundError(f".vtr file not found: {p}")

        reader = vtkXMLRectilinearGridReader()
        reader.SetFileName(str(p))
        reader.Update()
        rgrid = reader.GetOutput()

        rgrid = self._ensure_point_scalars(rgrid)
        scalars = rgrid.GetPointData().GetScalars()
        smin, smax = scalars.GetRange()
        if not (smax > smin):
            smax = smin + 1.0

        self._scalar_name = scalars.GetName() or "scalars"
        self._scalar_range = (smin, smax)

        self.ren = vtk.vtkRenderer()
        self.rw = vtk.vtkRenderWindow()
        self.rw.AddRenderer(self.ren)

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(rgrid)
        if hasattr(mapper, "SetScalarModeToUsePointData"):
            mapper.SetScalarModeToUsePointData()

        prop = vtk.vtkVolumeProperty()
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOff()

        lo, hi = self._scalar_range
        span = hi - lo

        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(lo, 0.0)
        otf.AddPoint(lo + 0.05 * span, 0.0)
        otf.AddPoint(lo + 0.50 * span, 0.35)
        otf.AddPoint(hi, 0.9)
        prop.SetScalarOpacity(otf)

        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(lo + 0.5 * span, 0.95, 0.45, 0.25)
        ctf.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        prop.SetColor(ctf)

        self._otf = otf
        self._ctf = ctf
        self._vol_prop = prop
        self._mapper = mapper

        vol = vtk.vtkVolume()
        vol.SetMapper(mapper)
        vol.SetProperty(prop)

        self.ren.AddVolume(vol)
        self.ren.SetBackground(0.10, 0.10, 0.12)
        self.ren.ResetCamera()

    # -----------------------------------------------------------------
    def _apply_window(self):
        s = self.state
        lo, hi = self._scalar_range
        w_lo = max(lo, min(hi, s.window_lo))
        w_hi = max(lo, min(hi, s.window_hi))
        if w_hi <= w_lo:
            w_hi = w_lo + (hi - lo) * 0.01

        self._otf.RemoveAllPoints()
        span = w_hi - w_lo
        self._otf.AddPoint(w_lo, 0.0)
        self._otf.AddPoint(w_lo + 0.05 * span, 0.0)
        self._otf.AddPoint(w_lo + 0.55 * span, s.mid_opacity)
        self._otf.AddPoint(w_hi, 1.0)

        self._vol_prop.SetScalarOpacity(self._otf)
        self.ctrl.view_update()

    # -----------------------------------------------------------------
    def _build_ui(self):
        s = self.state
        lo, hi = self._scalar_range
        s.window_lo = lo
        s.window_hi = hi
        s.mid_opacity = 0.35
        s.title = f"VTR Volume: {Path(self._vtr_path).name}"
        s.scalar_name = self._scalar_name
        s.range_text = f"{lo:.3g} … {hi:.3g}"

        with SinglePageWithDrawerLayout(self.server, title=s.title) as layout:
            with layout.toolbar:
                v3.VTextField(
                    v_model=("window_lo",),
                    type="number",
                    label="Win Lo",
                    density="compact",
                    style="max-width:110px",
                    hide_details=True,
                )
                v3.VTextField(
                    v_model=("window_hi",),
                    type="number",
                    label="Win Hi",
                    density="compact",
                    style="max-width:110px",
                    hide_details=True,
                )
                v3.VSlider(
                    v_model=("mid_opacity",),
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    label="Mid α",
                    density="compact",
                    style="max-width:180px",
                    hide_details=True,
                )
                v3.VBtn("Apply", click="apply_window()", density="compact")
                v3.VSpacer()
                h.Div("{{ scalar_name }} range: {{ range_text }}", classes="mr-4 text-caption")

            with layout.content:
                with vtklocal.LocalView(self.rw, throttle_rate=30) as view:
                    self.ctrl.view_update = view.update
                    self.ctrl.view_reset_camera = view.reset_camera

        @self.server.controller.add("apply_window")
        def _apply_window_ctrl():
            self._apply_window()

        @s.change("window_lo", "window_hi", "mid_opacity")
        def _auto_apply(**_):
            self._apply_window()

    # -----------------------------------------------------------------


if __name__ == "__main__":
    VTRViewer(VTR_PATH, server=True)
