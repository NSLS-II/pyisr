#!/usr/bin/env python3
"""
RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

Column 1 = Load Data + editable Experiment/Detector params (from YAML or defaults)
Column 2 = Build RSM Map + Regrid (now supports OPTIONAL crop before regrid)
Column 3 = View RSM + Status

Crop behavior:
- If enabled, we rebuild a fresh RSMBuilder (from the edited YAML) to AVOID cumulative cropping,
  then call builder.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax)) before regrid.

Requirements (example):
  pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Tuple

from qtpy import QtCore, QtWidgets
import napari
from napari.utils.notifications import show_info, show_error

from magicgui.widgets import (
    Container, Label,
    FileEdit, TextEdit, LineEdit,
    CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
)

# YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from rsm3d.data_io import RSMDataLoader
from rsm3d.rsm3d     import RSMBuilder
from rsm3d.data_viz  import RSMNapariViewer


# ────────────────────────── defaults for your parameters ──────────────────────────
DETECTOR_DEFAULTS: Dict[str, Any] = {
    "distance": 0.78105,    # meters
    "pitch": 7.5e-05,       # meters
    "ycenter": 257,         # pixel index
    "xcenter": 515,         # pixel index
    "xpixels": 1030,        # number of pixels (X)
    "ypixels": 514,         # number of pixels (Y)
    "energy": 11.470,       # keV
    "wavelength": None,     # or float
}

# ────────────────────────── small helpers ──────────────────────────
def _HSeparator(height: int = 10):
    w = Label(value="")
    try:
        w.native.setFrameShape(QtWidgets.QFrame.HLine)
        w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedHeight(height)
    except Exception:
        pass
    return w

def _parse_scan_list(text: str) -> List[int]:
    if not text or not text.strip():
        return []
    out = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(f"Bad scan range: '{part}'")
        else:
            if part.isdigit():
                out.add(int(part))
            else:
                raise ValueError(f"Bad scan id: '{part}'")
    return sorted(out)

def _parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
    if text is None:
        return (200, None, None)
    text = text.strip()
    if not text:
        return (200, None, None)
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 1:
        parts += ["*", "*"]
    if len(parts) != 3:
        raise ValueError("Grid must be 'x,y,z' (y/z may be '*' or empty)")
    def _one(p):
        if p in ("*", "", None):
            return None
        if not p.isdigit():
            raise ValueError(f"Grid size must be integer or '*', got '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid sizes must be > 0")
        return v
    x, y, z = (_one(p) for p in parts)
    if x is None:
        raise ValueError("Grid x (first value) is required")
    return x, y, z

def _open_tiffs_in_napari(tiff_dir: str):
    viewer = napari.Viewer()
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    opened = False
    for pat in patterns:
        try:
            viewer.open(os.path.join(tiff_dir, pat))
            opened = True
        except Exception:
            pass
    if not opened:
        show_error("No TIFF files found (patterns: *.tif, *.tiff).")
    return viewer

# ────────────────────────── YAML params ──────────────────────────
def _prefill_params_from_yaml(setup_path: str, widgets: Dict[str, Any]):
    if not (yaml and setup_path and os.path.isfile(setup_path)):
        return
    try:
        data = yaml.safe_load(open(setup_path, "r", encoding="utf-8")) or {}
        exp = {}
        if isinstance(data, dict):
            exp = data.get("ExperimentSetup") or data.get("experiment_setup") or {}
        if not isinstance(exp, dict):
            return
        for k, w in widgets.items():
            if k in exp:
                val = exp[k]
                try:
                    if isinstance(w, FloatSpinBox):
                        if val is None or val == "None":
                            continue
                        w.value = float(val)
                    elif isinstance(w, SpinBox):
                        w.value = int(val)
                    elif isinstance(w, LineEdit):
                        w.value = "" if val is None else str(val)
                except Exception:
                    pass
    except Exception:
        pass

def _collect_param_values(widgets: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, w in widgets.items():
        if isinstance(w, FloatSpinBox):
            out[k] = float(w.value)
        elif isinstance(w, SpinBox):
            out[k] = int(w.value)
        elif isinstance(w, LineEdit):
            txt = (w.value or "").strip()
            if txt == "" or txt.lower() == "none":
                out[k] = None
            else:
                try:
                    out[k] = float(txt)
                except Exception:
                    out[k] = txt
        else:
            out[k] = w.value
    return out

def _write_edited_setup_yaml(original_yaml: str, param_overrides: Dict[str, Any]) -> str:
    if yaml is None:
        return original_yaml
    try:
        base = {}
        if original_yaml and os.path.isfile(original_yaml):
            base = yaml.safe_load(open(original_yaml, "r", encoding="utf-8")) or {}
        if not isinstance(base, dict):
            base = {}
        exp = base.get("ExperimentSetup") or base.get("experiment_setup") or {}
        if not isinstance(exp, dict):
            exp = {}
        exp.update(param_overrides)
        base["ExperimentSetup"] = exp
        tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
        out_path = os.path.join(tmpdir, "edited_setup.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(base, f, sort_keys=False)
        return out_path
    except Exception as e:
        show_error(f"Failed to write edited setup YAML: {e}")
        return original_yaml


# ────────────────────────── build the app ──────────────────────────
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # ── Column 1: Load Data + editable params
    spec_file = FileEdit(mode="r", label="SPEC file")
    setup_file = FileEdit(mode="r", label="YAML setup (optional)")
    tiff_dir   = FileEdit(mode="d", label="TIFF folder")
    scans      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
    only_hkl   = CheckBox(label="Only HKL scans", value=True)
    btn_load   = PushButton(text="Load Data")
    btn_tiff   = PushButton(text="View TIFFs in napari")

    title_params = Label(value="<b>Experiment / detector parameters (editable)</b>")
    distance_w   = FloatSpinBox(label="distance (m)",  min=0.0, max=10.0,   step=1e-5, value=DETECTOR_DEFAULTS["distance"])
    pitch_w      = FloatSpinBox(label="pitch (m)",     min=0.0, max=1e-2,   step=1e-6, value=DETECTOR_DEFAULTS["pitch"])
    ycenter_w    = SpinBox(label="ycenter (px)",       min=0,   max=10000,  step=1,    value=DETECTOR_DEFAULTS["ycenter"])
    xcenter_w    = SpinBox(label="xcenter (px)",       min=0,   max=10000,  step=1,    value=DETECTOR_DEFAULTS["xcenter"])
    xpixels_w    = SpinBox(label="xpixels",            min=1,   max=100000, step=1,    value=DETECTOR_DEFAULTS["xpixels"])
    ypixels_w    = SpinBox(label="ypixels",            min=1,   max=100000, step=1,    value=DETECTOR_DEFAULTS["ypixels"])
    energy_w     = FloatSpinBox(label="energy (keV)",  min=0.0, max=200.0,  step=0.001,value=DETECTOR_DEFAULTS["energy"])
    wavelength_w = LineEdit(label="wavelength (Å or None)", value="None" if DETECTOR_DEFAULTS["wavelength"] is None else str(DETECTOR_DEFAULTS["wavelength"]))

    PARAM_WIDGETS: Dict[str, Any] = {
        "distance": distance_w,
        "pitch": pitch_w,
        "ycenter": ycenter_w,
        "xcenter": xcenter_w,
        "xpixels": xpixels_w,
        "ypixels": ypixels_w,
        "energy": energy_w,
        "wavelength": wavelength_w,
    }

    col1_cont = Container(
        layout="vertical",
        widgets=[
            spec_file, setup_file, tiff_dir, scans, only_hkl,
            btn_load, btn_tiff, _HSeparator(),
            title_params,
            distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
        ],
    )

    # ── Column 2: Build + Regrid (with optional crop)
    # Build
    ub_2pi          = CheckBox(label="UB includes 2π", value=True)
    center_one_based= CheckBox(label="1-based center", value=False)
    btn_build       = PushButton(text="Build RSM Map")
    # Regrid
    space       = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
    grid_shape  = LineEdit(label="Grid (x,y,z)", value="200,*,*")
    fuzzy       = CheckBox(label="Fuzzy gridder", value=True)
    fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
    normalize   = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")

    # NEW: optional crop controls
    crop_enable = CheckBox(label="Crop the RSM", value=False)
    y_min_w = SpinBox(label="y_min (px)", min=0, max=50000, step=1, value=240)
    y_max_w = SpinBox(label="y_max (px)", min=0, max=50000, step=1, value=510)
    x_min_w = SpinBox(label="x_min (px)", min=0, max=50000, step=1, value=380)
    x_max_w = SpinBox(label="x_max (px)", min=0, max=50000, step=1, value=610)

    btn_regrid  = PushButton(text="Regrid")

    col2_cont = Container(
        layout="vertical",
        widgets=[
            Label(value="<b>Build options</b>"),
            ub_2pi, center_one_based, btn_build,
            _HSeparator(),
            Label(value="<b>Regrid options</b>"),
            space, grid_shape, fuzzy, fuzzy_width, normalize,
            _HSeparator(),
            Label(value="<b>Optional crop (pixel bounds)</b>"),
            crop_enable, y_min_w, y_max_w, x_min_w, x_max_w,
            btn_regrid,
        ],
    )

    # ── Column 3: View
    log_view  = CheckBox(label="Log view", value=True)
    cmap      = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
    rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
    contrast  = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
    btn_view  = PushButton(text="View RSM")
    status    = TextEdit(label="Status", value="")
    try:
        status.native.setReadOnly(True)
        status.native.setMinimumHeight(120)
    except Exception:
        pass

    col3_cont = Container(
        layout="vertical",
        widgets=[log_view, cmap, rendering, contrast, btn_view, _HSeparator(), status],
    )

    # ── wrap columns in a QSplitter
    def _wrap(title: str, container: Container) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
        t = QtWidgets.QLabel(f"<b>{title}</b>")
        v.addWidget(t)
        v.addWidget(container.native, 1)
        return host

    w_col1 = _wrap("Load Data", col1_cont)
    w_col2 = _wrap("Build RSM Map", col2_cont)
    w_col3 = _wrap("View RSM", col3_cont)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    splitter.addWidget(w_col1)
    splitter.addWidget(w_col2)
    splitter.addWidget(w_col3)
    splitter.setHandleWidth(2)
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 1)
    splitter.setSizes([400, 400, 400])

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("RSM3D")
    win.setCentralWidget(splitter)
    win.resize(1280, 720)
    win.show()

    # ── App state
    state: Dict[str, Any] = dict(
        loader=None,
        builder=None,
        Q=None, hkl=None, intensity=None,
        grid=None, edges=None,
        edited_setup_path=None,
    )

    # ── status helper
    def set_status(msg: str):
        status.value = msg
        try:
            show_info(msg)
        except Exception:
            pass

    # ── handlers
    def on_view_tiffs():
        d = tiff_dir.value
        if not d or not os.path.isdir(d):
            show_error("Please select a valid TIFF folder first.")
            return
        _open_tiffs_in_napari(d)
        set_status("Opened TIFFs in napari.")

    def on_load():
        try:
            if not spec_file.value or not os.path.isfile(spec_file.value):
                raise FileNotFoundError("Select a valid SPEC file.")
            if setup_file.value and not os.path.isfile(setup_file.value):
                raise FileNotFoundError("YAML setup path does not exist.")
            if not tiff_dir.value or not os.path.isdir(tiff_dir.value):
                raise NotADirectoryError("Select a valid TIFF folder.")
            scan_list = _parse_scan_list(scans.value or "")
            if not scan_list:
                raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

            set_status(f"Loading scans {scan_list}…")
            loader = RSMDataLoader(
                spec_file.value,
                setup_file.value,
                tiff_dir.value,
                selected_scans=scan_list,
                process_hklscan_only=bool(only_hkl.value),
            )
            loader.load()

            # Prefill parameter widgets from YAML if present
            _prefill_params_from_yaml(setup_file.value, PARAM_WIDGETS)

            state.update(loader=loader, builder=None, Q=None, hkl=None, intensity=None, grid=None, edges=None, edited_setup_path=None)
            set_status("Data loaded. Parameters are editable in column 1.")
        except Exception as e:
            show_error(f"Load error: {e}")
            set_status(f"Load failed: {e}")

    def on_build():
        try:
            if state["loader"] is None:
                raise RuntimeError("Load data first.")

            # Collect parameter edits and write a temp YAML with ExperimentSetup overrides
            overrides = _collect_param_values(PARAM_WIDGETS)
            edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
            state["edited_setup_path"] = edited_yaml

            # Rebuild loader to use edited YAML
            scan_list = _parse_scan_list(scans.value or "")
            loader2 = RSMDataLoader(
                spec_file.value,
                edited_yaml,
                tiff_dir.value,
                selected_scans=scan_list,
                process_hklscan_only=bool(only_hkl.value),
            )
            set_status("Reloading with edited ExperimentSetup…")
            loader2.load()

            set_status("Computing Q/HKL/intensity…")
            builder = RSMBuilder(
                loader2,
                ub_includes_2pi=bool(ub_2pi.value),
                center_is_one_based=bool(center_one_based.value),
            )
            Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)

            state.update(loader=loader2, builder=builder, Q=Q_samp, hkl=hkl_arr, intensity=intensity_arr, grid=None, edges=None)
            set_status("RSM map built.")
        except Exception as e:
            show_error(f"Build error: {e}")
            set_status(f"Build failed: {e}")

    def on_regrid():
        try:
            if state["builder"] is None:
                raise RuntimeError("Build the RSM map first.")
            gx, gy, gz = _parse_grid_shape(grid_shape.value)

            # Work on a fresh builder if cropping is enabled (avoid cumulative crops)
            b_work = state["builder"]
            if crop_enable.value:
                ymin, ymax = int(y_min_w.value), int(y_max_w.value)
                xmin, xmax = int(x_min_w.value), int(x_max_w.value)
                if ymin >= ymax or xmin >= xmax:
                    raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
                if state["loader"] is None:
                    raise RuntimeError("Internal error: loader missing; run Build again.")
                set_status(f"Cropping to y=({ymin},{ymax}), x=({xmin},{xmax})…")
                # Recreate builder from current loader + options
                b_work = RSMBuilder(
                    state["loader"],
                    ub_includes_2pi=bool(ub_2pi.value),
                    center_is_one_based=bool(center_one_based.value),
                )
                b_work.compute_full(verbose=False)
                # Perform crop (by pixel indices)
                b_work.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

            set_status(f"Regridding to {space.value.upper()} grid {(gx, gy, gz)}…")
            kw = dict(
                space=space.value,
                grid_shape=(gx, gy, gz),
                fuzzy=bool(fuzzy.value),
                normalize=normalize.value,
                stream=True,
            )
            if fuzzy.value and float(fuzzy_width.value) > 0:
                kw["width"] = float(fuzzy_width.value)

            grid, edges = b_work.regrid_xu(**kw)
            state.update(grid=grid, edges=edges)
            set_status("Regrid completed.")
        except Exception as e:
            show_error(f"Regrid error: {e}")
            set_status(f"Regrid failed: {e}")

    def on_view():
        try:
            if state["grid"] is None or state["edges"] is None:
                raise RuntimeError("Regrid first.")
            try:
                lo_str, hi_str = [p.strip() for p in (contrast.value or "1,99.8").split(",")]
                pct_lo, pct_hi = float(lo_str), float(hi_str)
                if not (0 <= pct_lo < pct_hi <= 100):
                    raise ValueError
            except Exception:
                raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

            viz = RSMNapariViewer(
                state["grid"],
                state["edges"],
                space=space.value,
                name="RSM3D",
                log_view=bool(log_view.value),
                contrast_percentiles=(pct_lo, pct_hi),
                cmap=cmap.value,
                rendering=rendering.value,
            )
            viz.launch()  # returns napari.Viewer
            set_status("RSM viewer opened.")
        except Exception as e:
            show_error(f"View error: {e}")
            set_status(f"View failed: {e}")

    # connect
    btn_tiff.clicked.connect(on_view_tiffs)
    btn_load.clicked.connect(on_load)
    btn_build.clicked.connect(on_build)
    btn_regrid.clicked.connect(on_regrid)
    btn_view.clicked.connect(on_view)

    # go
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

# Column 1 = Load Data + editable Experiment/Detector parameters (from YAML or provided defaults)
# Column 2 = Build RSM Map + Regrid
# Column 3 = View RSM + Status

# Edits to the parameters are applied when you click "Build RSM Map":
# - We merge the edited values into ExperimentSetup in a temp YAML
# - Then rebuild the loader and compute the RSM

# Requirements (example):
#   pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import re
# import sys
# import tempfile
# from typing import Any, Dict, List, Tuple

# from qtpy import QtCore, QtWidgets
# import napari
# from napari.utils.notifications import show_info, show_error

# from magicgui.widgets import (
#     Container, Label,
#     FileEdit, TextEdit, LineEdit,
#     CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
# )

# # YAML support
# try:
#     import yaml  # type: ignore
# except Exception:
#     yaml = None

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ────────────────────────── defaults for your parameters ──────────────────────────
# DETECTOR_DEFAULTS: Dict[str, Any] = {
#     # Experiment / detector parameters
#     "distance": 0.78105,    # meters
#     "pitch": 7.5e-05,       # meters
#     "ycenter": 257,         # pixel index
#     "xcenter": 515,         # pixel index
#     "xpixels": 1030,        # number of pixels (X)
#     "ypixels": 514,         # number of pixels (Y)
#     "energy": 11.470,       # KeV
#     "wavelength": None,     # or float
# }

# # ────────────────────────── small helpers ──────────────────────────
# def _HSeparator(height: int = 10):
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def _parse_scan_list(text: str) -> List[int]:
#     """Accepts: '17, 18-22,30' → [17,18,19,20,21,22,30]"""
#     if not text or not text.strip():
#         return []
#     out = set()
#     for part in re.split(r"[,\s]+", text.strip()):
#         if not part:
#             continue
#         if "-" in part:
#             a, b = part.split("-", 1)
#             a, b = a.strip(), b.strip()
#             if a.isdigit() and b.isdigit():
#                 lo, hi = int(a), int(b)
#                 if lo > hi:
#                     lo, hi = hi, lo
#                 out.update(range(lo, hi + 1))
#             else:
#                 raise ValueError(f"Bad scan range: '{part}'")
#         else:
#             if part.isdigit():
#                 out.add(int(part))
#             else:
#                 raise ValueError(f"Bad scan id: '{part}'")
#     return sorted(out)

# def _parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     """
#     'x,y,z' where y/z may be omitted or '*'.
#     '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
#     """
#     if text is None:
#         return (200, None, None)
#     text = text.strip()
#     if not text:
#         return (200, None, None)
#     parts = [p.strip() for p in text.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*' or empty)")
#     def _one(p):
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     x, y, z = (_one(p) for p in parts)
#     if x is None:
#         raise ValueError("Grid x (first value) is required")
#     return x, y, z

# def _open_tiffs_in_napari(tiff_dir: str):
#     """Open TIFF stack(s) in napari via glob patterns."""
#     viewer = napari.Viewer()
#     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
#     opened = False
#     for pat in patterns:
#         try:
#             viewer.open(os.path.join(tiff_dir, pat))
#             opened = True
#         except Exception:
#             pass
#     if not opened:
#         show_error("No TIFF files found (patterns: *.tif, *.tiff).")
#     return viewer

# # ────────────────────────── YAML handling for parameters ──────────────────────────
# def _prefill_params_from_yaml(setup_path: str, widgets: Dict[str, Any]):
#     """Load ExperimentSetup from YAML (if present) and prefill widgets."""
#     if not (yaml and setup_path and os.path.isfile(setup_path)):
#         return
#     try:
#         data = yaml.safe_load(open(setup_path, "r", encoding="utf-8")) or {}
#         exp = {}
#         if isinstance(data, dict):
#             exp = data.get("ExperimentSetup") or data.get("experiment_setup") or {}
#         if not isinstance(exp, dict):
#             return
#         for k, w in widgets.items():
#             if k in exp:
#                 val = exp[k]
#                 try:
#                     if isinstance(w, (FloatSpinBox,)):
#                         if val is None or val == "None":
#                             continue
#                         w.value = float(val)
#                     elif isinstance(w, (SpinBox,)):
#                         w.value = int(val)
#                     elif isinstance(w, (LineEdit,)):
#                         w.value = "" if val is None else str(val)
#                 except Exception:
#                     # ignore bad types; leave defaults
#                     pass
#     except Exception:
#         pass

# def _collect_param_values(widgets: Dict[str, Any]) -> Dict[str, Any]:
#     """Read current values from the parameter widgets."""
#     out: Dict[str, Any] = {}
#     for k, w in widgets.items():
#         if isinstance(w, FloatSpinBox):
#             out[k] = float(w.value)
#         elif isinstance(w, SpinBox):
#             out[k] = int(w.value)
#         elif isinstance(w, LineEdit):
#             txt = (w.value or "").strip()
#             if txt == "" or txt.lower() == "none":
#                 out[k] = None
#             else:
#                 # try to coerce to float, else keep as string
#                 try:
#                     out[k] = float(txt)
#                 except Exception:
#                     out[k] = txt
#         else:
#             out[k] = w.value
#     return out

# def _write_edited_setup_yaml(original_yaml: str, param_overrides: Dict[str, Any]) -> str:
#     """Merge overrides into ExperimentSetup and write to a temp YAML; return path."""
#     if yaml is None:
#         return original_yaml
#     try:
#         base = {}
#         if original_yaml and os.path.isfile(original_yaml):
#             base = yaml.safe_load(open(original_yaml, "r", encoding="utf-8")) or {}
#         if not isinstance(base, dict):
#             base = {}
#         exp = base.get("ExperimentSetup") or base.get("experiment_setup") or {}
#         if not isinstance(exp, dict):
#             exp = {}
#         exp.update(param_overrides)
#         base["ExperimentSetup"] = exp
#         tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
#         out_path = os.path.join(tmpdir, "edited_setup.yaml")
#         with open(out_path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(base, f, sort_keys=False)
#         return out_path
#     except Exception as e:
#         show_error(f"Failed to write edited setup YAML: {e}")
#         return original_yaml


# # ────────────────────────── build the app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     # ── Column 1: Load Data + your editable parameters
#     spec_file = FileEdit(mode="r", label="SPEC file")
#     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
#     tiff_dir   = FileEdit(mode="d", label="TIFF folder")
#     scans      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl   = CheckBox(label="Only HKL scans", value=True)
#     btn_load   = PushButton(text="Load Data")
#     btn_tiff   = PushButton(text="View TIFFs in napari")

#     # Parameter editors (type-aware widgets)
#     title_params = Label(value="<b>Experiment / detector parameters (editable)</b>")
#     distance_w   = FloatSpinBox(label="distance (m)",  min=0.0, max=10.0, step=1e-5, value=DETECTOR_DEFAULTS["distance"])
#     pitch_w      = FloatSpinBox(label="pitch (m)",     min=0.0, max=1e-2, step=1e-6, value=DETECTOR_DEFAULTS["pitch"])
#     ycenter_w    = SpinBox(label="ycenter (px)",       min=0, max=10000, step=1, value=DETECTOR_DEFAULTS["ycenter"])
#     xcenter_w    = SpinBox(label="xcenter (px)",       min=0, max=10000, step=1, value=DETECTOR_DEFAULTS["xcenter"])
#     xpixels_w    = SpinBox(label="xpixels",            min=1, max=100000, step=1, value=DETECTOR_DEFAULTS["xpixels"])
#     ypixels_w    = SpinBox(label="ypixels",            min=1, max=100000, step=1, value=DETECTOR_DEFAULTS["ypixels"])
#     energy_w     = FloatSpinBox(label="energy (keV)",  min=0.0, max=200.0, step=0.001, value=DETECTOR_DEFAULTS["energy"])
#     wavelength_w = LineEdit(label="wavelength (Å or None)", value="None" if DETECTOR_DEFAULTS["wavelength"] is None else str(DETECTOR_DEFAULTS["wavelength"]))

#     PARAM_WIDGETS: Dict[str, Any] = {
#         "distance": distance_w,
#         "pitch": pitch_w,
#         "ycenter": ycenter_w,
#         "xcenter": xcenter_w,
#         "xpixels": xpixels_w,
#         "ypixels": ypixels_w,
#         "energy": energy_w,
#         "wavelength": wavelength_w,
#     }

#     col1_cont = Container(
#         layout="vertical",
#         widgets=[
#             spec_file, setup_file, tiff_dir, scans, only_hkl,
#             btn_load, btn_tiff, _HSeparator(),
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#         ],
#     )

#     # ── Column 2: Build + Regrid
#     ub_2pi          = CheckBox(label="UB includes 2π", value=True)
#     center_one_based= CheckBox(label="1-based center", value=False)
#     btn_build       = PushButton(text="Build RSM Map")

#     space       = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
#     grid_shape  = LineEdit(label="Grid (x,y,z)", value="200,*,*")
#     fuzzy       = CheckBox(label="Fuzzy gridder", value=True)
#     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
#     normalize   = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
#     btn_regrid  = PushButton(text="Regrid")

#     col2_cont = Container(
#         layout="vertical",
#         widgets=[
#             Label(value="<b>Build options</b>"),
#             ub_2pi, center_one_based, btn_build,
#             _HSeparator(),
#             Label(value="<b>Regrid options</b>"),
#             space, grid_shape, fuzzy, fuzzy_width, normalize, btn_regrid,
#         ],
#     )

#     # ── Column 3: View
#     log_view  = CheckBox(label="Log view", value=True)
#     cmap      = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
#     rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
#     contrast  = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
#     btn_view  = PushButton(text="View RSM")
#     status    = TextEdit(label="Status", value="")
#     try:
#         status.native.setReadOnly(True)
#         status.native.setMinimumHeight(120)
#     except Exception:
#         pass

#     col3_cont = Container(
#         layout="vertical",
#         widgets=[log_view, cmap, rendering, contrast, btn_view, _HSeparator(), status],
#     )

#     # ── wrap each column with a titled QWidget and place in QSplitter
#     def _wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
#         t = QtWidgets.QLabel(f"<b>{title}</b>")
#         v.addWidget(t)
#         v.addWidget(container.native, 1)
#         return host

#     w_col1 = _wrap("① Load Data + Parameters", col1_cont)
#     w_col2 = _wrap("② Build RSM Map  &  ③ Regrid", col2_cont)
#     w_col3 = _wrap("④ View RSM", col3_cont)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w_col1)
#     splitter.addWidget(w_col2)
#     splitter.addWidget(w_col3)
#     splitter.setHandleWidth(2)
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     splitter.setSizes([400, 400, 400])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Parameters in Column 1")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ── App state
#     state: Dict[str, Any] = dict(
#         loader=None,
#         builder=None,
#         Q=None, hkl=None, intensity=None,
#         grid=None, edges=None,
#         edited_setup_path=None,
#     )

#     # ── status helper
#     def set_status(msg: str):
#         status.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ── button handlers
#     def on_view_tiffs():
#         d = tiff_dir.value
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder first.")
#             return
#         _open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load():
#         try:
#             if not spec_file.value or not os.path.isfile(spec_file.value):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if setup_file.value and not os.path.isfile(setup_file.value):
#                 raise FileNotFoundError("YAML setup path does not exist.")
#             if not tiff_dir.value or not os.path.isdir(tiff_dir.value):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = _parse_scan_list(scans.value or "")
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             loader = RSMDataLoader(
#                 spec_file.value,
#                 setup_file.value,
#                 tiff_dir.value,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl.value),
#             )
#             loader.load()

#             # Prefill parameter widgets from YAML if present
#             _prefill_params_from_yaml(setup_file.value, PARAM_WIDGETS)

#             state.update(loader=loader, builder=None, Q=None, hkl=None, intensity=None, grid=None, edges=None, edited_setup_path=None)
#             set_status("Data loaded. Parameters are editable in column 1.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_status(f"Load failed: {e}")

#     def on_build():
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")

#             # Collect parameter edits and write a temp YAML with ExperimentSetup overrides
#             overrides = _collect_param_values(PARAM_WIDGETS)
#             edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
#             state["edited_setup_path"] = edited_yaml

#             # Rebuild loader to use edited YAML
#             scan_list = _parse_scan_list(scans.value or "")
#             loader2 = RSMDataLoader(
#                 spec_file.value,
#                 edited_yaml,
#                 tiff_dir.value,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl.value),
#             )
#             set_status("Reloading with edited ExperimentSetup…")
#             loader2.load()

#             set_status("Computing Q/HKL/intensity…")
#             builder = RSMBuilder(
#                 loader2,
#                 ub_includes_2pi=bool(ub_2pi.value),
#                 center_is_one_based=bool(center_one_based.value),
#             )
#             Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)

#             state.update(loader=loader2, builder=builder, Q=Q_samp, hkl=hkl_arr, intensity=intensity_arr, grid=None, edges=None)
#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")

#     def on_regrid():
#         try:
#             builder = state["builder"]
#             if builder is None:
#                 raise RuntimeError("Build the RSM map first.")
#             gx, gy, gz = _parse_grid_shape(grid_shape.value)
#             set_status(f"Regridding to {space.value.upper()} grid {(gx, gy, gz)}…")
#             kw = dict(
#                 space=space.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy.value),
#                 normalize=normalize.value,
#                 stream=True,
#             )
#             if fuzzy.value and float(fuzzy_width.value) > 0:
#                 kw["width"] = float(fuzzy_width.value)
#             grid, edges = builder.regrid_xu(**kw)
#             state.update(grid=grid, edges=edges)
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")

#     def on_view():
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")
#             try:
#                 lo_str, hi_str = [p.strip() for p in (contrast.value or "1,99.8").split(",")]
#                 pct_lo, pct_hi = float(lo_str), float(hi_str)
#                 if not (0 <= pct_lo < pct_hi <= 100):
#                     raise ValueError
#             except Exception:
#                 raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

#             viz = RSMNapariViewer(
#                 state["grid"],
#                 state["edges"],
#                 space=space.value,
#                 name="RSM3D",
#                 log_view=bool(log_view.value),
#                 contrast_percentiles=(pct_lo, pct_hi),
#                 cmap=cmap.value,
#                 rendering=rendering.value,
#             )
#             viz.launch()  # returns napari.Viewer
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     # connect
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)

#     # go
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# RSM3D app: Qt main window + magicgui panels inside a 3-pane QSplitter.

# Columns:
#   [1] Load Data  -> choose files/scans; editable ExperimentSetup; preview TIFFs
#   [2] Build + Regrid -> compute RSM, then regrid controls
#   [3] View RSM   -> viewer controls + status

# Requirements (example):
#   pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import re
# import sys
# import tempfile
# from typing import Any, Dict, List, Tuple

# from qtpy import QtCore, QtWidgets
# import napari
# from napari.utils.notifications import show_info, show_error

# from magicgui.widgets import (
#     Container, Label,
#     FileEdit, TextEdit, LineEdit,
#     CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
# )

# # Optional YAML for ExperimentSetup parsing
# try:
#     import yaml  # type: ignore
# except Exception:  # pragma: no cover
#     yaml = None

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ────────────────────────── helpers ──────────────────────────
# def _parse_scan_list(text: str) -> List[int]:
#     """Accepts: '17, 18-22,30' → [17,18,19,20,21,22,30]"""
#     if not text or not text.strip():
#         return []
#     out = set()
#     for part in re.split(r"[,\s]+", text.strip()):
#         if not part:
#             continue
#         if "-" in part:
#             a, b = part.split("-", 1)
#             a, b = a.strip(), b.strip()
#             if a.isdigit() and b.isdigit():
#                 lo, hi = int(a), int(b)
#                 if lo > hi:
#                     lo, hi = hi, lo
#                 out.update(range(lo, hi + 1))
#             else:
#                 raise ValueError(f"Bad scan range: '{part}'")
#         else:
#             if part.isdigit():
#                 out.add(int(part))
#             else:
#                 raise ValueError(f"Bad scan id: '{part}'")
#     return sorted(out)


# def _parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     """
#     'x,y,z' where y/z may be omitted or '*'.
#     '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
#     """
#     if text is None:
#         return (200, None, None)
#     text = text.strip()
#     if not text:
#         return (200, None, None)
#     parts = [p.strip() for p in text.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*' or empty)")
#     def _one(p):
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     x, y, z = (_one(p) for p in parts)
#     if x is None:
#         raise ValueError("Grid x (first value) is required")
#     return x, y, z


# def _open_tiffs_in_napari(tiff_dir: str):
#     """Open TIFF stack(s) in napari via glob patterns."""
#     viewer = napari.Viewer()
#     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
#     opened = False
#     for pat in patterns:
#         try:
#             viewer.open(os.path.join(tiff_dir, pat))
#             opened = True
#         except Exception:
#             pass
#     if not opened:
#         show_error("No TIFF files found (patterns: *.tif, *.tiff).")
#     return viewer


# # ────────────── dynamic ExperimentSetup editor (editable) ─────────────
# def _widget_for_value(key: str, val: Any):
#     """Return a magicgui widget appropriate for type(val)."""
#     label = key
#     if isinstance(val, bool):
#         w = CheckBox(label=label, value=bool(val))
#     elif isinstance(val, int):
#         w = SpinBox(label=label, min=-10_000_000, max=10_000_000, step=1, value=int(val))
#     elif isinstance(val, float):
#         w = FloatSpinBox(label=label, min=-1e12, max=1e12, step=0.1, value=float(val))
#     else:
#         # serialize lists/dicts to YAML-ish text for edit
#         if isinstance(val, (list, dict)):
#             try:
#                 txt = yaml.safe_dump(val, sort_keys=False) if yaml else str(val)
#             except Exception:
#                 txt = str(val)
#             w = TextEdit(label=label, value=txt)
#         else:
#             w = LineEdit(label=label, value=str(val))
#     return w


# def _build_setup_editor(exp_setup: Dict[str, Any]) -> Tuple[Container, Dict[str, Any]]:
#     """Build a vertical container of editable widgets for ExperimentSetup."""
#     fields: Dict[str, Any] = {}
#     widgets: List = [Label(value="ExperimentSetup (editable)"), _HSeparator()]
#     for k in sorted(exp_setup.keys()):
#         w = _widget_for_value(k, exp_setup[k])
#         widgets.append(w)
#         fields[k] = exp_setup[k]
#     col = Container(layout="vertical", widgets=widgets)
#     # Make it scrollable by putting into a QWidget with a scroll area (Qt-side)
#     try:
#         col.native.setMinimumWidth(320)
#     except Exception:
#         pass
#     return col, fields


# def _collect_setup_overrides(editor_container: Container) -> Dict[str, Any]:
#     """Read current values from editor_container widgets."""
#     overrides: Dict[str, Any] = {}
#     for w in editor_container:
#         if isinstance(w, Label):  # skip title/separators
#             continue
#         k = getattr(w, "label", None) or getattr(w, "name", None)
#         if not k:
#             continue
#         if isinstance(w, CheckBox):
#             overrides[k] = bool(w.value)
#         elif isinstance(w, SpinBox):
#             overrides[k] = int(w.value)
#         elif isinstance(w, FloatSpinBox):
#             overrides[k] = float(w.value)
#         elif isinstance(w, LineEdit):
#             overrides[k] = w.value
#         elif isinstance(w, TextEdit):
#             txt = w.value or ""
#             if yaml:
#                 try:
#                     parsed = yaml.safe_load(txt)
#                     overrides[k] = parsed
#                 except Exception:
#                     overrides[k] = txt
#             else:
#                 overrides[k] = txt
#         else:
#             overrides[k] = w.value
#     return overrides


# def _load_experiment_setup(setup_file: str) -> Dict[str, Any]:
#     """Return ExperimentSetup dict (or empty) from YAML file."""
#     if not setup_file or not os.path.isfile(setup_file) or yaml is None:
#         return {}
#     try:
#         data = yaml.safe_load(open(setup_file, "r", encoding="utf-8"))
#         if isinstance(data, dict):
#             exp = data.get("ExperimentSetup") or data.get("experiment_setup")
#             if isinstance(exp, dict):
#                 return exp
#     except Exception:
#         pass
#     return {}


# def _write_edited_setup_yaml(original_yaml: str, overrides: Dict[str, Any]) -> str:
#     """Write a temp YAML with ExperimentSetup merged with overrides; return path."""
#     if yaml is None:
#         return original_yaml
#     try:
#         base = {}
#         if original_yaml and os.path.isfile(original_yaml):
#             base = yaml.safe_load(open(original_yaml, "r", encoding="utf-8")) or {}
#         if not isinstance(base, dict):
#             base = {}
#         exp = base.get("ExperimentSetup") or base.get("experiment_setup") or {}
#         if not isinstance(exp, dict):
#             exp = {}
#         exp.update(overrides)
#         base["ExperimentSetup"] = exp
#         tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
#         out_path = os.path.join(tmpdir, "edited_setup.yaml")
#         with open(out_path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(base, f, sort_keys=False)
#         return out_path
#     except Exception as e:
#         show_error(f"Failed to write edited setup YAML: {e}")
#         return original_yaml


# # ────────────────────────── tiny UI helpers ──────────────────────────
# def _HSeparator(height: int = 10):
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w


# def _wrap_in_column(widget: Container, title: str) -> QtWidgets.QWidget:
#     """Wrap a magicgui Container in a QWidget with margins and a title."""
#     host = QtWidgets.QWidget()
#     lay = QtWidgets.QVBoxLayout(host)
#     lay.setContentsMargins(8, 8, 8, 8)
#     lay.setSpacing(6)
#     t = QtWidgets.QLabel(f"<b>{title}</b>")
#     lay.addWidget(t)
#     lay.addWidget(widget.native, 1)
#     return host


# # ────────────────────────── build the app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     # ── Column 1: Load Data + editable ExperimentSetup
#     spec_file = FileEdit(mode="r", label="SPEC file")
#     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
#     tiff_dir = FileEdit(mode="d", label="TIFF folder")
#     scans = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl = CheckBox(label="Only HKL scans", value=True)
#     btn_load = PushButton(text="Load Data")
#     btn_tiff = PushButton(text="View TIFFs in napari")

#     exp_editor_placeholder = Container(
#         layout="vertical",
#         widgets=[_HSeparator(), Label(value="ExperimentSetup editor appears here after Load")],
#     )

#     col1_cont = Container(
#         layout="vertical",
#         widgets=[
#             spec_file, setup_file, tiff_dir, scans, only_hkl,
#             btn_load, btn_tiff,
#             _HSeparator(), exp_editor_placeholder,
#         ],
#     )

#     # ── Column 2: Build + Regrid (combined)
#     # Build
#     ub_2pi = CheckBox(label="UB includes 2π", value=True)
#     center_one_based = CheckBox(label="1-based center", value=False)
#     btn_build = PushButton(text="Build RSM Map")
#     # Regrid
#     space = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
#     grid_shape = LineEdit(label="Grid (x,y,z)", value="200,*,*")
#     fuzzy = CheckBox(label="Fuzzy gridder", value=True)
#     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
#     normalize = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
#     btn_regrid = PushButton(text="Regrid")

#     col2_cont = Container(
#         layout="vertical",
#         widgets=[
#             Label(value="Build options"),
#             ub_2pi, center_one_based, btn_build,
#             _HSeparator(),
#             Label(value="Regrid options"),
#             space, grid_shape, fuzzy, fuzzy_width, normalize, btn_regrid,
#         ],
#     )

#     # ── Column 3: View RSM
#     log_view = CheckBox(label="Log view", value=True)
#     cmap = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
#     rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
#     contrast = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
#     btn_view = PushButton(text="View RSM")
#     status = TextEdit(label="Status", value="")
#     try:
#         status.native.setReadOnly(True)
#         status.native.setMinimumHeight(120)
#     except Exception:
#         pass

#     col3_cont = Container(
#         layout="vertical",
#         widgets=[log_view, cmap, rendering, contrast, btn_view, _HSeparator(), status],
#     )

#     # ── Wrap each magicgui column into a QWidget and put them in a QSplitter
#     w_col1 = _wrap_in_column(col1_cont, "① Load Data")
#     w_col2 = _wrap_in_column(col2_cont, "② Build RSM Map  &  ③ Regrid")
#     w_col3 = _wrap_in_column(col3_cont, "④ View RSM")

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w_col1)
#     splitter.addWidget(w_col2)
#     splitter.addWidget(w_col3)

#     # Visible vertical dividers come from QSplitter handles
#     splitter.setHandleWidth(2)
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     # Give equal initial sizes
#     splitter.setSizes([400, 400, 400])

#     # ── Main window
#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — 3 Columns")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ── App state (shared)
#     state: Dict[str, Any] = dict(
#         loader=None,               # RSMDataLoader
#         builder=None,              # RSMBuilder
#         Q=None, hkl=None, intensity=None,
#         grid=None, edges=None,
#         edited_setup_path=None,    # path to temp edited YAML (if any)
#         exp_editor_container=exp_editor_placeholder,
#         exp_editor_loaded=False,
#     )

#     # ── status helper
#     def set_status(msg: str):
#         status.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ── handlers
#     def on_view_tiffs():
#         d = tiff_dir.value
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder first.")
#             return
#         _open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load():
#         try:
#             if not spec_file.value or not os.path.isfile(spec_file.value):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if setup_file.value and not os.path.isfile(setup_file.value):
#                 raise FileNotFoundError("YAML setup path does not exist.")
#             if not tiff_dir.value or not os.path.isdir(tiff_dir.value):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = _parse_scan_list(scans.value or "")
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             loader = RSMDataLoader(
#                 spec_file.value,
#                 setup_file.value,
#                 tiff_dir.value,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["Q"] = state["hkl"] = state["intensity"] = None
#             state["grid"] = state["edges"] = None
#             state["edited_setup_path"] = None

#             # Build editable ExperimentSetup UI
#             exp_dict = _load_experiment_setup(setup_file.value) if setup_file.value else {}
#             state["exp_editor_container"].widgets = []  # clear area
#             if exp_dict:
#                 editor_col, _snapshot = _build_setup_editor(exp_dict)
#                 state["exp_editor_container"].extend([editor_col])
#                 state["exp_editor_loaded"] = True
#                 set_status("Data loaded. ExperimentSetup ready to edit.")
#             else:
#                 state["exp_editor_container"].extend([Label(value="(No ExperimentSetup found in YAML)")])
#                 state["exp_editor_loaded"] = False
#                 set_status("Data loaded. No ExperimentSetup found in YAML.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_status(f"Load failed: {e}")

#     def on_build():
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             # If user edited ExperimentSetup, write a temp YAML and rebuild loader
#             edited_yaml = setup_file.value
#             if state["exp_editor_loaded"]:
#                 editor_cols = [w for w in state["exp_editor_container"] if isinstance(w, Container)]
#                 if editor_cols:
#                     overrides = _collect_setup_overrides(editor_cols[0])
#                     edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
#                     state["edited_setup_path"] = edited_yaml

#             scan_list = _parse_scan_list(scans.value or "")
#             loader2 = RSMDataLoader(
#                 spec_file.value,
#                 edited_yaml,
#                 tiff_dir.value,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl.value),
#             )
#             set_status("Reloading with edited ExperimentSetup…")
#             loader2.load()

#             set_status("Computing Q/HKL/intensity…")
#             builder = RSMBuilder(
#                 loader2,
#                 ub_includes_2pi=bool(ub_2pi.value),
#                 center_is_one_based=bool(center_one_based.value),
#             )
#             Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)

#             state["loader"]   = loader2
#             state["builder"]  = builder
#             state["Q"]        = Q_samp
#             state["hkl"]      = hkl_arr
#             state["intensity"]= intensity_arr
#             state["grid"]     = None
#             state["edges"]    = None

#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")

#     def on_regrid():
#         try:
#             builder = state["builder"]
#             if builder is None:
#                 raise RuntimeError("Build the RSM map first.")
#             gx, gy, gz = _parse_grid_shape(grid_shape.value)
#             set_status(f"Regridding to {space.value.upper()} grid {(gx, gy, gz)}…")
#             kw = dict(
#                 space=space.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy.value),
#                 normalize=normalize.value,
#                 stream=True,
#             )
#             if fuzzy.value and float(fuzzy_width.value) > 0:
#                 kw["width"] = float(fuzzy_width.value)
#             grid, edges = builder.regrid_xu(**kw)
#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")

#     def on_view():
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")
#             try:
#                 lo_str, hi_str = [p.strip() for p in (contrast.value or "1,99.8").split(",")]
#                 pct_lo, pct_hi = float(lo_str), float(hi_str)
#                 if not (0 <= pct_lo < pct_hi <= 100):
#                     raise ValueError
#             except Exception:
#                 raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

#             viz = RSMNapariViewer(
#                 state["grid"],
#                 state["edges"],
#                 space=space.value,
#                 name="RSM3D",
#                 log_view=bool(log_view.value),
#                 contrast_percentiles=(pct_lo, pct_hi),
#                 cmap=cmap.value,
#                 rendering=rendering.value,
#             )
#             viz.launch()  # returns napari.Viewer
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     # ── connect buttons
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)

#     # ── go
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()