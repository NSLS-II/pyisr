#!/usr/bin/env python3
"""
RSM3D app: single-YAML persistence for all parameters.
- Column 1: Data path boxes -> Setup parameters -> Buttons (Load Data, View TIFFs)
- Column 2: Build + optional Crop + Regrid
- Column 3: View + Status

YAML path:
- Use env var RSM3D_DEFAULTS_YAML if set, else ~/.rsm3d_defaults.yaml
- App loads at startup and writes back immediately on any change.

Requirements (example):
  pip install napari magicgui qtpy pyyaml xrayutilities   # plus your rsm3d package
"""

from __future__ import annotations

import os, pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

from qtpy import QtCore, QtWidgets
import napari
from napari.utils.notifications import show_info, show_error

from magicgui.widgets import (
    Container, Label,
    FileEdit, TextEdit, LineEdit,
    CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
)

import yaml  # required

from rsm3d.data_io import RSMDataLoader
from rsm3d.rsm3d     import RSMBuilder
from rsm3d.data_viz  import RSMNapariViewer


# ────────────────────────── YAML utils ──────────────────────────
# import os, pathlib
DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
os.environ.setdefault(DEFAULTS_ENV, str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()))
# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"

def yaml_path() -> str:
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

def ensure_yaml(path: str):
    """Create minimal structure if missing (no hard-coded default values)."""
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seed = {
        "data": {
            "spec_file": None,
            "tiff_dir": None,
            "scans": "",
            "only_hkl": None,
        },
        "experiment": {
            "distance": None,
            "pitch": None,
            "ycenter": None,
            "xcenter": None,
            "xpixels": None,
            "ypixels": None,
            "energy": None,
            "wavelength": None,
        },
        "build": {
            "ub_includes_2pi": None,
            "center_is_one_based": None,
        },
        "crop": {
            "enable": None,
            "y_min": None, "y_max": None,
            "x_min": None, "x_max": None,
        },
        "regrid": {
            "space": None,        # "hkl"/"q"
            "grid_shape": "",     # e.g. "200,*,*"
            "fuzzy": None,
            "fuzzy_width": None,
            "normalize": None,    # "mean"/"sum"
        },
        "view": {
            "log_view": None,
            "cmap": None,         # e.g. "inferno"
            "rendering": None,    # "attenuated_mip"/"mip"/"translucent"
            "contrast_lo": None,  # %
            "contrast_hi": None,  # %
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(seed, f, sort_keys=False)

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except Exception:
        return {}

def save_yaml(path: str, doc: Dict[str, Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    except Exception as e:
        show_error(f"Failed to write YAML: {e}")


# ────────────────────────── helpers ──────────────────────────
def HSep(h: int = 10):
    w = Label(value="")
    try:
        w.native.setFrameShape(QtWidgets.QFrame.HLine)
        w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedHeight(h)
    except Exception:
        pass
    return w

def parse_scan_list(text: str) -> List[int]:
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

def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
    if text is None:
        return (None, None, None)
    s = text.strip()
    if not s:
        return (None, None, None)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1:
        parts += ["*", "*"]
    if len(parts) != 3:
        raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
    def one(p):
        if p in ("*", "", None):
            return None
        if not p.isdigit():
            raise ValueError(f"Grid size must be integer or '*', got '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid sizes must be > 0")
        return v
    return tuple(one(p) for p in parts)  # type: ignore

def open_tiffs_in_napari(tiff_dir: str):
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


# ────────────────────────── app ──────────────────────────
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    ypath = yaml_path()
    ensure_yaml(ypath)
    ydoc = load_yaml(ypath)

    # Flag to suppress save while populating from YAML
    setting_up = False

    # ---------- Column 1: Data paths -> Setup (experiment) -> Buttons ----------
    # Data path boxes FIRST
    spec_file_w = FileEdit(mode="r", label="SPEC file")
    tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
    scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
    only_hkl_w  = CheckBox(label="Only HKL scans")

    # Setup parameters (experiment/detector)
    title_params = Label(value="<b>Experiment / Detector</b>")
    distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
    pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
    ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
    xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
    xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
    ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
    energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
    wavelength_w = LineEdit(label="wavelength (Å or None)")

    # Buttons at the BOTTOM
    btn_load = PushButton(text="Load Data")
    btn_tiff = PushButton(text="View TIFFs in napari")

    col1 = Container(
        layout="vertical",
        widgets=[
            # Data first
            spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
            HSep(),
            # Then setup parameters
            title_params,
            distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
            HSep(),
            # Buttons last
            btn_load, btn_tiff,
        ],
    )

    # ---------- Column 2: Build + Crop + Regrid ----------
    title_build = Label(value="<b>Build</b>")
    ub_2pi_w    = CheckBox(label="UB includes 2π")
    center_one_based_w = CheckBox(label="1-based center")
    btn_build   = PushButton(text="Build RSM Map")

    title_regrid = Label(value="<b>Regrid</b>")
    space_w      = ComboBox(label="Space", choices=["hkl", "q"])
    grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
    fuzzy_w      = CheckBox(label="Fuzzy gridder")
    fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
    normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

    title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
    crop_enable_w= CheckBox(label="Crop before regrid")
    y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
    y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
    x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
    x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

    btn_regrid   = PushButton(text="Regrid")

    col2 = Container(
        layout="vertical",
        widgets=[
            title_build, ub_2pi_w, center_one_based_w, btn_build,
            HSep(),
            title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
            HSep(),
            title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
            btn_regrid,
        ],
    )

    # ---------- Column 3: View ----------
    title_view   = Label(value="<b>View</b>")
    log_view_w   = CheckBox(label="Log view")
    cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
    rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
    contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
    contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)
    btn_view     = PushButton(text="View RSM")
    status_w     = TextEdit(label="Status", value="")
    try:
        status_w.native.setReadOnly(True)
        status_w.native.setMinimumHeight(120)
    except Exception:
        pass

    col3 = Container(
        layout="vertical",
        widgets=[title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w, btn_view, HSep(), status_w],
    )

    # ---------- Wrap in a 3-way splitter ----------
    def wrap(title: str, container: Container) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
        t = QtWidgets.QLabel(f"<b>{title}</b>")
        v.addWidget(t)
        v.addWidget(container.native, 1)
        return host

    w1 = wrap("① Data & Setup", col1)
    w2 = wrap("② Build  &  ③ Regrid", col2)
    w3 = wrap("④ View", col3)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    splitter.addWidget(w1); splitter.addWidget(w2); splitter.addWidget(w3)
    splitter.setHandleWidth(2); splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
    splitter.setSizes([400, 400, 400])

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("RSM3D (Qt + magicgui) — Single YAML")
    win.setCentralWidget(splitter)
    win.resize(1280, 720)
    win.show()

    # ---------- bind widgets to YAML (load → UI, UI → save) ----------
    ydoc = load_yaml(ypath)  # reload after UI built

    # Map: section -> { key: widget }
    W: Dict[str, Dict[str, Any]] = {
        "data": {
            "spec_file": spec_file_w,
            "tiff_dir":  tiff_dir_w,
            "scans":     scans_w,
            "only_hkl":  only_hkl_w,
        },
        "experiment": {
            "distance":   distance_w,
            "pitch":      pitch_w,
            "ycenter":    ycenter_w,
            "xcenter":    xcenter_w,
            "xpixels":    xpixels_w,
            "ypixels":    ypixels_w,
            "energy":     energy_w,
            "wavelength": wavelength_w,
        },
        "build": {
            "ub_includes_2pi":    ub_2pi_w,
            "center_is_one_based": center_one_based_w,
        },
        "crop": {
            "enable": crop_enable_w,
            "y_min":  y_min_w, "y_max": y_max_w,
            "x_min":  x_min_w, "x_max": x_max_w,
        },
        "regrid": {
            "space":      space_w,
            "grid_shape": grid_shape_w,
            "fuzzy":      fuzzy_w,
            "fuzzy_width":fuzzy_width_w,
            "normalize":  normalize_w,
        },
        "view": {
            "log_view":   log_view_w,
            "cmap":       cmap_w,
            "rendering":  rendering_w,
            "contrast_lo":contrast_lo_w,
            "contrast_hi":contrast_hi_w,
        },
    }

    def set_widget(widget, value):
        try:
            if value is None:
                return
            if isinstance(widget, (FloatSpinBox, SpinBox)):
                widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
            elif isinstance(widget, CheckBox):
                widget.value = bool(value)
            elif isinstance(widget, ComboBox):
                # only set if valid
                if str(value) in widget.choices:
                    widget.value = str(value)
            elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
                widget.value = str(value)
        except Exception:
            pass

    # populate UI
    setting_up = True
    for section, mapping in W.items():
        vals = ydoc.get(section, {})
        for key, widget in mapping.items():
            set_widget(widget, vals.get(key, None))
    # ensure structure present
    for s in W:
        ydoc.setdefault(s, {})
    save_yaml(ypath, ydoc)
    setting_up = False

    # save on change
    def val_for_yaml(widget, section: str, key: str):
        if section == "experiment" and key == "wavelength":
            txt = (widget.value or "").strip()
            if txt.lower() in {"", "none", "null"}:
                return None
            try:
                return float(txt)
            except Exception:
                return txt
        if isinstance(widget, FloatSpinBox):
            return float(widget.value)
        if isinstance(widget, SpinBox):
            return int(widget.value)
        if isinstance(widget, CheckBox):
            return bool(widget.value)
        if isinstance(widget, ComboBox):
            return str(widget.value)
        if isinstance(widget, (LineEdit, TextEdit, FileEdit)):
            return str(widget.value)
        return widget.value

    def on_changed(section: str, key: str, widget):
        nonlocal ydoc
        if setting_up:
            return
        ydoc.setdefault(section, {})
        ydoc[section][key] = val_for_yaml(widget, section, key)
        save_yaml(ypath, ydoc)

    for section, mapping in W.items():
        for key, widget in mapping.items():
            widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

    # ---------- status helper ----------
    def set_status(msg: str):
        status_w.value = msg
        try:
            show_info(msg)
        except Exception:
            pass

    # ---------- actions ----------
    state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

    def on_view_tiffs():
        d = (tiff_dir_w.value or "").strip()
        if not d or not os.path.isdir(d):
            show_error("Please select a valid TIFF folder (column 1).")
            return
        open_tiffs_in_napari(d)
        set_status("Opened TIFFs in napari.")

    def on_load():
        try:
            spec = (spec_file_w.value or "").strip()
            tdir = (tiff_dir_w.value or "").strip()
            if not spec or not os.path.isfile(spec):
                raise FileNotFoundError("Select a valid SPEC file.")
            if not tdir or not os.path.isdir(tdir):
                raise NotADirectoryError("Select a valid TIFF folder.")
            scan_list = parse_scan_list(scans_w.value or "")
            if not scan_list:
                raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

            set_status(f"Loading scans {scan_list}…")
            # No setup YAML: pass None as setup file
            loader = RSMDataLoader(
                spec,
                None,          # <-- no setup yaml
                tdir,
                selected_scans=scan_list,
                process_hklscan_only=bool(only_hkl_w.value),
            )
            loader.load()
            state["loader"] = loader
            state["builder"] = None
            state["grid"] = state["edges"] = None
            set_status("Data loaded.")
        except Exception as e:
            show_error(f"Load error: {e}")
            set_status(f"Load failed: {e}")

    def on_build():
        try:
            if state["loader"] is None:
                raise RuntimeError("Load data first.")
            set_status("Computing Q/HKL/intensity…")
            builder = RSMBuilder(
                state["loader"],
                ub_includes_2pi=bool(ub_2pi_w.value),
                center_is_one_based=bool(center_one_based_w.value),
            )
            builder.compute_full(verbose=False)
            state["builder"] = builder
            state["grid"] = state["edges"] = None
            set_status("RSM map built.")
        except Exception as e:
            show_error(f"Build error: {e}")
            set_status(f"Build failed: {e}")

    def on_regrid():
        try:
            b = state["builder"]
            if b is None:
                raise RuntimeError("Build the RSM map first.")
            # optional crop (fresh builder to avoid cumulative crops)
            if bool(crop_enable_w.value):
                ymin, ymax = int(y_min_w.value), int(y_max_w.value)
                xmin, xmax = int(x_min_w.value), int(x_max_w.value)
                if ymin >= ymax or xmin >= xmax:
                    raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
                # rebuild from loader to avoid double crops
                if state["loader"] is None:
                    raise RuntimeError("Internal error: loader missing; run Build again.")
                b = RSMBuilder(
                    state["loader"],
                    ub_includes_2pi=bool(ub_2pi_w.value),
                    center_is_one_based=bool(center_one_based_w.value),
                )
                b.compute_full(verbose=False)
                b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

            gx, gy, gz = parse_grid_shape(grid_shape_w.value)
            if gx is None:
                raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")
            kw = dict(
                space=space_w.value,
                grid_shape=(gx, gy, gz),
                fuzzy=bool(fuzzy_w.value),
                normalize=normalize_w.value,
                stream=True,
            )
            if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
                kw["width"] = float(fuzzy_width_w.value)

            set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
            grid, edges = b.regrid_xu(**kw)
            state["grid"], state["edges"] = grid, edges
            set_status("Regrid completed.")
        except Exception as e:
            show_error(f"Regrid error: {e}")
            set_status(f"Regrid failed: {e}")

    def on_view():
        try:
            if state["grid"] is None or state["edges"] is None:
                raise RuntimeError("Regrid first.")
            lo = contrast_lo_w.value
            hi = contrast_hi_w.value
            if lo is None or hi is None:
                raise ValueError("Set contrast low/high percentages.")
            lo = float(lo); hi = float(hi)
            if not (0 <= lo < hi <= 100):
                raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

            viz = RSMNapariViewer(
                state["grid"],
                state["edges"],
                space=space_w.value,
                name="RSM3D",
                log_view=bool(log_view_w.value),
                contrast_percentiles=(lo, hi),
                cmap=cmap_w.value,
                rendering=rendering_w.value,
            )
            viz.launch()
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

    # run
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

# - Column 1: All editable parameters (Experiment/Detector + Data). "Load Data" and "View TIFFs" at the bottom.
# - Column 2: Build RSM Map + Regrid (with optional crop before regrid)
# - Column 3: View RSM + Status

# Persistence:
# - Reads all parameters from a YAML file at startup (no hard-coded defaults here).
# - Any UI change immediately writes back to that YAML.
# - That same YAML is passed to RSMDataLoader as the "setup" (so ExperimentSetup is honored).

# YAML path resolution:
# - If env var RSM3D_DEFAULTS_YAML is set -> use it
# - Else -> ~/.rsm3d_defaults.yaml

# Requirements (example):
#   pip install napari magicgui qtpy pyyaml xrayutilities   # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# from qtpy import QtCore, QtWidgets
# import napari
# from napari.utils.notifications import show_info, show_error

# from magicgui.widgets import (
#     Container, Label,
#     FileEdit, TextEdit, LineEdit,
#     CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
# )

# # YAML
# import yaml  # required

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ────────────────────────── YAML path & helpers ──────────────────────────
# DEFAULTS_ENV = "~/pyprojects/pyisr/examples/rsm3d_defaults.yaml"

# def _yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def _ensure_yaml_exists(path: str):
#     """Create file with minimal sections if it doesn't exist. Values remain empty/null."""
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {
#             "spec_file": None,
#             "tiff_dir": None,
#             "scans": "",          # "17, 18-22, 30"
#             "only_hkl": None,     # bool
#         },
#         "experiment": {
#             "distance": None,
#             "pitch": None,
#             "ycenter": None,
#             "xcenter": None,
#             "xpixels": None,
#             "ypixels": None,
#             "energy": None,
#             "wavelength": None,
#         },
#         "build": {
#             "ub_includes_2pi": None,
#             "center_is_one_based": None,
#         },
#         "crop": {
#             "enable": None,
#             "y_min": None, "y_max": None,
#             "x_min": None, "x_max": None,
#         },
#         "regrid": {
#             "space": None,                 # "hkl" | "q"
#             "grid_shape": "",              # e.g. "200,*,*"
#             "fuzzy": None,                 # bool
#             "fuzzy_width": None,           # float
#             "normalize": None,             # "mean" | "sum"
#         },
#         "view": {
#             "log_view": None,              # bool
#             "cmap": None,                  # e.g. "inferno"
#             "rendering": None,             # "attenuated_mip" | "mip" | "translucent"
#             "contrast_lo": None,           # float (percent)
#             "contrast_hi": None,           # float (percent)
#         },
#         # For RSMDataLoader, we will pass this file path as setup YAML.
#         # Your loader should read ExperimentSetup from here if needed.
#         "ExperimentSetup": {}  # will be filled as user edits experiment params
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def _load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def _save_yaml(path: str, doc: Dict[str, Any]):
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")


# # ────────────────────────── misc helpers ──────────────────────────
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
#         return (None, None, None)
#     text = text.strip()
#     if not text:
#         return (None, None, None)
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
#     return x, y, z

# def _open_tiffs_in_napari(tiff_dir: str):
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


# # ────────────────────────── build the app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     # Resolve YAML, ensure it exists, load it
#     ypath = _yaml_path()
#     _ensure_yaml_exists(ypath)
#     ydoc = _load_yaml(ypath)

#     # A flag to avoid write-back while we are setting values programmatically
#     setting_up = False

#     # ------------- Column 1: All parameters (Experiment/Detector + Data) -------------
#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e6, max=1e6, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e6, max=1e6, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e6, max=1e6, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")

#     title_data   = Label(value="<b>Data</b>")
#     spec_file_w  = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w   = FileEdit(mode="d", label="TIFF folder")
#     scans_w      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl_w   = CheckBox(label="Only HKL scans")

#     # Buttons at the bottom per your request
#     btn_load   = PushButton(text="Load Data")
#     btn_tiff   = PushButton(text="View TIFFs in napari")

#     col1_cont = Container(
#         layout="vertical",
#         widgets=[
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             _HSeparator(),
#             title_data,
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             _HSeparator(),
#             btn_load, btn_tiff,
#         ],
#     )

#     # ------------- Column 2: Build + Crop + Regrid -------------
#     title_build  = Label(value="<b>Build</b>")
#     ub_2pi_w     = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build    = PushButton(text="Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed", tooltip="Examples: 200,*,* or 256,256,256")
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e6, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="Regrid")

#     col2_cont = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w, btn_build,
#             _HSeparator(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             _HSeparator(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             btn_regrid,
#         ],
#     )

#     # ------------- Column 3: View -------------
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)", min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)
#     btn_view     = PushButton(text="View RSM")
#     status_w     = TextEdit(label="Status", value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(120)
#     except Exception:
#         pass

#     col3_cont = Container(
#         layout="vertical",
#         widgets=[title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w, btn_view, _HSeparator(), status_w],
#     )

#     # ------------- Wrap in QSplitter -------------
#     def _wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
#         t = QtWidgets.QLabel(f"<b>{title}</b>")
#         v.addWidget(t)
#         v.addWidget(container.native, 1)
#         return host

#     w_col1 = _wrap("① Parameters & Data", col1_cont)
#     w_col2 = _wrap("② Build  &  ③ Regrid", col2_cont)
#     w_col3 = _wrap("④ View", col3_cont)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w_col1); splitter.addWidget(w_col2); splitter.addWidget(w_col3)
#     splitter.setHandleWidth(2); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([400, 400, 400])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — YAML-backed controls")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ------------- In-memory config doc (kept in sync with YAML) -------------
#     ydoc = _load_yaml(ypath)  # reload after UI creation
#     # Map each (section, key) to its widget and conversion functions
#     F = {
#         "experiment": {
#             "distance":   (distance_w,   float,  lambda v: v),
#             "pitch":      (pitch_w,      float,  lambda v: v),
#             "ycenter":    (ycenter_w,    int,    lambda v: v),
#             "xcenter":    (xcenter_w,    int,    lambda v: v),
#             "xpixels":    (xpixels_w,    int,    lambda v: v),
#             "ypixels":    (ypixels_w,    int,    lambda v: v),
#             "energy":     (energy_w,     float,  lambda v: v),
#             # wavelength handled specially (None or float/string)
#             "wavelength": (wavelength_w, str,    lambda v: v),
#         },
#         "data": {
#             "spec_file": (spec_file_w,  str, lambda v: v),
#             "tiff_dir":  (tiff_dir_w,   str, lambda v: v),
#             "scans":     (scans_w,      str, lambda v: v),
#             "only_hkl":  (only_hkl_w,   bool,lambda v: v),
#         },
#         "build": {
#             "ub_includes_2pi":   (ub_2pi_w,          bool, lambda v: v),
#             "center_is_one_based": (center_one_based_w, bool, lambda v: v),
#         },
#         "crop": {
#             "enable": (crop_enable_w, bool, lambda v: v),
#             "y_min":  (y_min_w,       int,  lambda v: v),
#             "y_max":  (y_max_w,       int,  lambda v: v),
#             "x_min":  (x_min_w,       int,  lambda v: v),
#             "x_max":  (x_max_w,       int,  lambda v: v),
#         },
#         "regrid": {
#             "space":      (space_w,      str,  lambda v: v),
#             "grid_shape": (grid_shape_w, str,  lambda v: v),
#             "fuzzy":      (fuzzy_w,      bool, lambda v: v),
#             "fuzzy_width":(fuzzy_width_w,float,lambda v: v),
#             "normalize":  (normalize_w,  str,  lambda v: v),
#         },
#         "view": {
#             "log_view":   (log_view_w,   bool, lambda v: v),
#             "cmap":       (cmap_w,       str,  lambda v: v),
#             "rendering":  (rendering_w,  str,  lambda v: v),
#             "contrast_lo":(contrast_lo_w,float,lambda v: v),
#             "contrast_hi":(contrast_hi_w,float,lambda v: v),
#         },
#     }

#     # ------------- load YAML → widgets -------------
#     def _set_widget_from_yaml(widget, v):
#         # For None -> leave widget default. For others, set sensibly.
#         try:
#             if v is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(v) if isinstance(widget, FloatSpinBox) else int(v)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(v)
#             elif isinstance(widget, ComboBox):
#                 if v in widget.choices:
#                     widget.value = v
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(v)
#         except Exception:
#             pass

#     setting_up = True
#     for section, mapping in F.items():
#         sec = ydoc.get(section, {})
#         for key, (w, _ctype, _to_yaml) in mapping.items():
#             _set_widget_from_yaml(w, sec.get(key, None))
#     # Also mirror ExperimentSetup subset (used by RSMDataLoader) from "experiment"
#     ydoc.setdefault("ExperimentSetup", {})
#     ydoc["ExperimentSetup"].update({k: ydoc.get("experiment", {}).get(k, None)
#                                     for k in ["distance", "pitch", "ycenter", "xcenter", "xpixels", "ypixels", "energy", "wavelength"]})
#     _save_yaml(ypath, ydoc)
#     setting_up = False

#     # ------------- widgets → YAML on change -------------
#     def _val_for_yaml(widget, section: str, key: str):
#         # Special handling for wavelength (None accepted)
#         if section == "experiment" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt == "" or txt.lower() in {"none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt  # keep as text if not a float
#         # Normal cases
#         if isinstance(widget, FloatSpinBox):
#             return float(widget.value)
#         if isinstance(widget, SpinBox):
#             return int(widget.value)
#         if isinstance(widget, CheckBox):
#             return bool(widget.value)
#         if isinstance(widget, ComboBox):
#             return str(widget.value)
#         if isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#             return str(widget.value)
#         return widget.value

#     def _on_changed(section: str, key: str, widget):
#         nonlocal ydoc
#         if setting_up:
#             return
#         # Update in-memory doc
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = _val_for_yaml(widget, section, key)
#         # Keep ExperimentSetup mirrored from "experiment" section
#         if section == "experiment":
#             ydoc.setdefault("ExperimentSetup", {})
#             ydoc["ExperimentSetup"][key] = ydoc["experiment"][key]
#         # Persist
#         _save_yaml(ypath, ydoc)

#     # connect all widgets for persistence
#     for section, mapping in F.items():
#         for key, (w, _ctype, _to_yaml) in mapping.items():
#             w.changed.connect(lambda *_, s=section, k=key, ww=w: _on_changed(s, k, ww))

#     # ------------- Status helper -------------
#     def set_status(msg: str):
#         status_w.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ------------- actions -------------
#     def on_view_tiffs():
#         d = (tiff_dir_w.value or "").strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         _open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load():
#         try:
#             spec = (spec_file_w.value or "").strip()
#             tdir = (tiff_dir_w.value or "").strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = _parse_scan_list(scans_w.value or "")
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             # Use our persistent YAML as the setup file so ExperimentSetup is honored
#             loader = RSMDataLoader(
#                 spec,
#                 _yaml_path(),
#                 tdir,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_status(f"Load failed: {e}")

#     def on_build():
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             set_status("Computing Q/HKL/intensity…")
#             builder = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)
#             state["builder"] = builder
#             state["Q"], state["hkl"], state["intensity"] = Q_samp, hkl_arr, intensity_arr
#             state["grid"] = state["edges"] = None
#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")

#     def on_regrid():
#         try:
#             b = state["builder"]
#             if b is None:
#                 raise RuntimeError("Build the RSM map first.")
#             # Optional crop (fresh builder to avoid cumulative crops)
#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
#                 # Recompute a fresh builder from the loader to avoid double crops
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = _parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")
#             kw = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and float(fuzzy_width_w.value) > 0:
#                 kw["width"] = float(fuzzy_width_w.value)

#             set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#             grid, edges = b.regrid_xu(**kw)
#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")

#     def on_view():
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             viz = RSMNapariViewer(
#                 state["grid"],
#                 state["edges"],
#                 space=space_w.value,
#                 name="RSM3D",
#                 log_view=bool(log_view_w.value),
#                 contrast_percentiles=(lo, hi),
#                 cmap=cmap_w.value,
#                 rendering=rendering_w.value,
#             )
#             viz.launch()  # returns napari.Viewer
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     # ------------- connect -------------
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)

#     # ------------- App state -------------
#     state: Dict[str, Any] = dict(loader=None, builder=None, Q=None, hkl=None, intensity=None, grid=None, edges=None)

#     # ------------- go -------------
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

# NEW:
# - Default parameters are stored in a YAML file and loaded on startup.
# - Button "Save as default" writes the current values back to that YAML.

# Defaults YAML path:
# - If env var RSM3D_DEFAULTS_YAML is set, use that.
# - Otherwise, use ~/.rsm3d_defaults.yaml

# Columns:
#   [1] Load Data + editable Experiment/Detector params (defaults prefilled from YAML)
#   [2] Build RSM Map + Regrid (supports OPTIONAL crop before regrid)
#   [3] View RSM + Status

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
#     # ^ expects your RSMDataLoader to accept (spec_file, setup_file, tiff_dir, ...)
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # # ────────────────────────── defaults / YAML handling ──────────────────────────
# DEFAULTS_TEMPLATE: Dict[str, Any] = {
#     "ExperimentSetup": {
#         # Experiment / detector parameters
#         "distance": 0.78105,     # meters
#         "pitch": 7.5e-05,        # meters
#         "ycenter": 257,          # pixel index
#         "xcenter": 515,          # pixel index
#         "xpixels": 1030,         # number of pixels (X)
#         "ypixels": 514,          # number of pixels (Y)
#         "energy": 11.470,        # keV
#         "wavelength": None,      # or float (Å)
#     }
# }

# DEFAULTS_ENV = "./exp_setup.yaml"  # env var to override defaults path

# def _defaults_path() -> str:
#     path = os.environ.get(DEFAULTS_ENV, "").strip()
#     if path:
#         return os.path.abspath(os.path.expanduser(path))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def _ensure_defaults_file(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     if yaml is None:
#         # No YAML library available; we can't create the file, but app still runs.
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(DEFAULTS_TEMPLATE, f, sort_keys=False)

# def _load_defaults(path: str) -> Dict[str, Any]:
#     if yaml is None or not os.path.isfile(path):
#         return DEFAULTS_TEMPLATE.copy()
#     try:
#         data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#         if not isinstance(data, dict):
#             return DEFAULTS_TEMPLATE.copy()
#         # Normalize key
#         exp = data.get("ExperimentSetup") or data.get("experiment_setup")
#         if not isinstance(exp, dict):
#             data["ExperimentSetup"] = DEFAULTS_TEMPLATE["ExperimentSetup"].copy()
#         return data
#     except Exception:
#         return DEFAULTS_TEMPLATE.copy()

# def _save_defaults(path: str, experiment_setup: Dict[str, Any]) -> None:
#     if yaml is None:
#         show_error("PyYAML is not installed; cannot write defaults YAML.")
#         return
#     data = {"ExperimentSetup": experiment_setup}
#     try:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(data, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to save defaults YAML: {e}")


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


# # ────────────────────────── build the app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     # Resolve & ensure defaults YAML
#     dpath = _defaults_path()
#     _ensure_defaults_file(dpath)
#     defaults_doc = _load_defaults(dpath)
#     defaults = defaults_doc.get("ExperimentSetup", {})

#     # ── Column 1: Load Data + editable params (prefilled from defaults YAML)
#     spec_file = FileEdit(mode="r", label="SPEC file")
#     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
#     tiff_dir   = FileEdit(mode="d", label="TIFF folder")
#     scans      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl   = CheckBox(label="Only HKL scans", value=True)
#     btn_load   = PushButton(text="Load Data")
#     btn_tiff   = PushButton(text="View TIFFs in napari")
#     btn_save_defaults = PushButton(text="Save as default")

#     title_params = Label(value="<b>Experiment / detector parameters (editable; loaded from defaults YAML)</b>")
#     # Pull defaults or fallback to template values
#     defval = lambda k, fallback: defaults.get(k, fallback)

#     distance_w   = FloatSpinBox(label="distance (m)",  min=0.0, max=10.0,   step=1e-5, value=float(defval("distance", 0.78105)))
#     pitch_w      = FloatSpinBox(label="pitch (m)",     min=0.0, max=1e-2,   step=1e-6, value=float(defval("pitch", 7.5e-05)))
#     ycenter_w    = SpinBox(label="ycenter (px)",       min=0,   max=10000,  step=1,    value=int(defval("ycenter", 257)))
#     xcenter_w    = SpinBox(label="xcenter (px)",       min=0,   max=10000,  step=1,    value=int(defval("xcenter", 515)))
#     xpixels_w    = SpinBox(label="xpixels",            min=1,   max=100000, step=1,    value=int(defval("xpixels", 1030)))
#     ypixels_w    = SpinBox(label="ypixels",            min=1,   max=100000, step=1,    value=int(defval("ypixels", 514)))
#     energy_w     = FloatSpinBox(label="energy (keV)",  min=0.0, max=200.0,  step=0.001,value=float(defval("energy", 11.470)))
#     # wavelength may be None or a float
#     _wval = defval("wavelength", None)
#     wavelength_w = LineEdit(label="wavelength (Å or None)", value=("None" if _wval in (None, "None", "") else str(_wval)))

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
#             _HSeparator(),
#             Label(value=f"<i>Defaults YAML:</i> {dpath}"),
#             btn_save_defaults,
#         ],
#     )

#     # ── Column 2: Build + Regrid (with optional crop)
#     ub_2pi          = CheckBox(label="UB includes 2π", value=True)
#     center_one_based= CheckBox(label="1-based center", value=False)
#     btn_build       = PushButton(text="Build RSM Map")

#     space       = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
#     grid_shape  = LineEdit(label="Grid (x,y,z)", value="200,*,*")
#     fuzzy       = CheckBox(label="Fuzzy gridder", value=True)
#     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
#     normalize   = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")

#     crop_enable = CheckBox(label="Crop before regrid", value=False)
#     y_min_w = SpinBox(label="y_min (px)", min=0, max=50000, step=1, value=240)
#     y_max_w = SpinBox(label="y_max (px)", min=0, max=50000, step=1, value=510)
#     x_min_w = SpinBox(label="x_min (px)", min=0, max=50000, step=1, value=380)
#     x_max_w = SpinBox(label="x_max (px)", min=0, max=50000, step=1, value=610)

#     btn_regrid  = PushButton(text="Regrid")

#     col2_cont = Container(
#         layout="vertical",
#         widgets=[
#             Label(value="<b>Build options</b>"),
#             ub_2pi, center_one_based, btn_build,
#             _HSeparator(),
#             Label(value="<b>Regrid options</b>"),
#             space, grid_shape, fuzzy, fuzzy_width, normalize,
#             _HSeparator(),
#             Label(value="<b>Optional crop (pixel bounds)</b>"),
#             crop_enable, y_min_w, y_max_w, x_min_w, x_max_w,
#             btn_regrid,
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

#     # ── wrap columns in a QSplitter
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
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Defaults from YAML")
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
#         defaults_path=dpath,
#     )

#     # ── status helper
#     def set_status(msg: str):
#         status.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ── parameter utils
#     def _collect_param_values() -> Dict[str, Any]:
#         out: Dict[str, Any] = {}
#         for k, w in PARAM_WIDGETS.items():
#             if isinstance(w, FloatSpinBox):
#                 out[k] = float(w.value)
#             elif isinstance(w, SpinBox):
#                 out[k] = int(w.value)
#             elif isinstance(w, LineEdit):
#                 txt = (w.value or "").strip()
#                 if txt == "" or txt.lower() == "none":
#                     out[k] = None
#                 else:
#                     try:
#                         out[k] = float(txt)
#                     except Exception:
#                         out[k] = txt
#             else:
#                 out[k] = w.value
#         return out

#     # ── handlers
#     def on_save_defaults():
#         try:
#             vals = _collect_param_values()
#             _save_defaults(state["defaults_path"], vals)
#             set_status(f"Saved defaults to {state['defaults_path']}")
#         except Exception as e:
#             show_error(f"Save defaults error: {e}")
#             set_status(f"Save defaults failed: {e}")

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

#             # If a specific setup YAML is provided and has ExperimentSetup keys,
#             # we can optionally prefill the widgets from it (does not alter defaults).
#             if yaml and setup_file.value and os.path.isfile(setup_file.value):
#                 try:
#                     data = yaml.safe_load(open(setup_file.value, "r", encoding="utf-8")) or {}
#                     exp = data.get("ExperimentSetup") or data.get("experiment_setup") or {}
#                     if isinstance(exp, dict):
#                         # Only fill known keys
#                         for k, w in PARAM_WIDGETS.items():
#                             if k in exp:
#                                 val = exp[k]
#                                 if isinstance(w, FloatSpinBox):
#                                     if val is not None and val != "None":
#                                         w.value = float(val)
#                                 elif isinstance(w, SpinBox):
#                                     w.value = int(val)
#                                 elif isinstance(w, LineEdit):
#                                     w.value = "None" if val in (None, "None", "") else str(val)
#                 except Exception:
#                     pass

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
#             overrides = _collect_param_values()
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
#             if state["builder"] is None:
#                 raise RuntimeError("Build the RSM map first.")
#             gx, gy, gz = _parse_grid_shape(grid_shape.value)

#             # Optional crop on a fresh builder to avoid cumulative cropping
#             b_work = state["builder"]
#             if crop_enable.value:
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 set_status(f"Cropping to y=({ymin},{ymax}), x=({xmin},{xmax})…")
#                 b_work = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi.value),
#                     center_is_one_based=bool(center_one_based.value),
#                 )
#                 b_work.compute_full(verbose=False)
#                 b_work.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

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

#             grid, edges = b_work.regrid_xu(**kw)
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
#     btn_save_defaults.clicked.connect(on_save_defaults)
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
# RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

# Column 1 = Load Data + editable Experiment/Detector params (from YAML or defaults)
# Column 2 = Build RSM Map + Regrid (now supports OPTIONAL crop before regrid)
# Column 3 = View RSM + Status

# Crop behavior:
# - If enabled, we rebuild a fresh RSMBuilder (from the edited YAML) to AVOID cumulative cropping,
#   then call builder.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax)) before regrid.

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
#     "distance": 0.78105,    # meters
#     "pitch": 7.5e-05,       # meters
#     "ycenter": 257,         # pixel index
#     "xcenter": 515,         # pixel index
#     "xpixels": 1030,        # number of pixels (X)
#     "ypixels": 514,         # number of pixels (Y)
#     "energy": 11.470,       # keV
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

# # ────────────────────────── YAML params ──────────────────────────
# def _prefill_params_from_yaml(setup_path: str, widgets: Dict[str, Any]):
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
#                     if isinstance(w, FloatSpinBox):
#                         if val is None or val == "None":
#                             continue
#                         w.value = float(val)
#                     elif isinstance(w, SpinBox):
#                         w.value = int(val)
#                     elif isinstance(w, LineEdit):
#                         w.value = "" if val is None else str(val)
#                 except Exception:
#                     pass
#     except Exception:
#         pass

# def _collect_param_values(widgets: Dict[str, Any]) -> Dict[str, Any]:
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
#                 try:
#                     out[k] = float(txt)
#                 except Exception:
#                     out[k] = txt
#         else:
#             out[k] = w.value
#     return out

# def _write_edited_setup_yaml(original_yaml: str, param_overrides: Dict[str, Any]) -> str:
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

#     # ── Column 1: Load Data + editable params
#     spec_file = FileEdit(mode="r", label="SPEC file")
#     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
#     tiff_dir   = FileEdit(mode="d", label="TIFF folder")
#     scans      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl   = CheckBox(label="Only HKL scans", value=True)
#     btn_load   = PushButton(text="Load Data")
#     btn_tiff   = PushButton(text="View TIFFs in napari")

#     title_params = Label(value="<b>Experiment / detector parameters (editable)</b>")
#     distance_w   = FloatSpinBox(label="distance (m)",  min=0.0, max=10.0,   step=1e-5, value=DETECTOR_DEFAULTS["distance"])
#     pitch_w      = FloatSpinBox(label="pitch (m)",     min=0.0, max=1e-2,   step=1e-6, value=DETECTOR_DEFAULTS["pitch"])
#     ycenter_w    = SpinBox(label="ycenter (px)",       min=0,   max=10000,  step=1,    value=DETECTOR_DEFAULTS["ycenter"])
#     xcenter_w    = SpinBox(label="xcenter (px)",       min=0,   max=10000,  step=1,    value=DETECTOR_DEFAULTS["xcenter"])
#     xpixels_w    = SpinBox(label="xpixels",            min=1,   max=100000, step=1,    value=DETECTOR_DEFAULTS["xpixels"])
#     ypixels_w    = SpinBox(label="ypixels",            min=1,   max=100000, step=1,    value=DETECTOR_DEFAULTS["ypixels"])
#     energy_w     = FloatSpinBox(label="energy (keV)",  min=0.0, max=200.0,  step=0.001,value=DETECTOR_DEFAULTS["energy"])
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

#     # ── Column 2: Build + Regrid (with optional crop)
#     # Build
#     ub_2pi          = CheckBox(label="UB includes 2π", value=True)
#     center_one_based= CheckBox(label="1-based center", value=False)
#     btn_build       = PushButton(text="Build RSM Map")
#     # Regrid
#     space       = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
#     grid_shape  = LineEdit(label="Grid (x,y,z)", value="200,*,*")
#     fuzzy       = CheckBox(label="Fuzzy gridder", value=True)
#     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
#     normalize   = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")

#     # NEW: optional crop controls
#     crop_enable = CheckBox(label="Crop the RSM", value=False)
#     y_min_w = SpinBox(label="y_min (px)", min=0, max=50000, step=1, value=240)
#     y_max_w = SpinBox(label="y_max (px)", min=0, max=50000, step=1, value=510)
#     x_min_w = SpinBox(label="x_min (px)", min=0, max=50000, step=1, value=380)
#     x_max_w = SpinBox(label="x_max (px)", min=0, max=50000, step=1, value=610)

#     btn_regrid  = PushButton(text="Regrid")

#     col2_cont = Container(
#         layout="vertical",
#         widgets=[
#             Label(value="<b>Build options</b>"),
#             ub_2pi, center_one_based, btn_build,
#             _HSeparator(),
#             Label(value="<b>Regrid options</b>"),
#             space, grid_shape, fuzzy, fuzzy_width, normalize,
#             _HSeparator(),
#             Label(value="<b>Optional crop (pixel bounds)</b>"),
#             crop_enable, y_min_w, y_max_w, x_min_w, x_max_w,
#             btn_regrid,
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

#     # ── wrap columns in a QSplitter
#     def _wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
#         t = QtWidgets.QLabel(f"<b>{title}</b>")
#         v.addWidget(t)
#         v.addWidget(container.native, 1)
#         return host

#     w_col1 = _wrap("Load Data", col1_cont)
#     w_col2 = _wrap("Build RSM Map", col2_cont)
#     w_col3 = _wrap("View RSM", col3_cont)

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
#     win.setWindowTitle("RSM3D")
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
#             if state["builder"] is None:
#                 raise RuntimeError("Build the RSM map first.")
#             gx, gy, gz = _parse_grid_shape(grid_shape.value)

#             # Work on a fresh builder if cropping is enabled (avoid cumulative crops)
#             b_work = state["builder"]
#             if crop_enable.value:
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 set_status(f"Cropping to y=({ymin},{ymax}), x=({xmin},{xmax})…")
#                 # Recreate builder from current loader + options
#                 b_work = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi.value),
#                     center_is_one_based=bool(center_one_based.value),
#                 )
#                 b_work.compute_full(verbose=False)
#                 # Perform crop (by pixel indices)
#                 b_work.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

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

#             grid, edges = b_work.regrid_xu(**kw)
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

# # #!/usr/bin/env python3
# # """
# # RSM3D app: Qt main window + magicgui panels in a 3-pane QSplitter.

# # Column 1 = Load Data + editable Experiment/Detector parameters (from YAML or provided defaults)
# # Column 2 = Build RSM Map + Regrid
# # Column 3 = View RSM + Status

# # Edits to the parameters are applied when you click "Build RSM Map":
# # - We merge the edited values into ExperimentSetup in a temp YAML
# # - Then rebuild the loader and compute the RSM

# # Requirements (example):
# #   pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
# # """

# # from __future__ import annotations

# # import os
# # import re
# # import sys
# # import tempfile
# # from typing import Any, Dict, List, Tuple

# # from qtpy import QtCore, QtWidgets
# # import napari
# # from napari.utils.notifications import show_info, show_error

# # from magicgui.widgets import (
# #     Container, Label,
# #     FileEdit, TextEdit, LineEdit,
# #     CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
# # )

# # # YAML support
# # try:
# #     import yaml  # type: ignore
# # except Exception:
# #     yaml = None

# # from rsm3d.data_io import RSMDataLoader
# # from rsm3d.rsm3d     import RSMBuilder
# # from rsm3d.data_viz  import RSMNapariViewer


# # # ────────────────────────── defaults for your parameters ──────────────────────────
# # DETECTOR_DEFAULTS: Dict[str, Any] = {
# #     # Experiment / detector parameters
# #     "distance": 0.78105,    # meters
# #     "pitch": 7.5e-05,       # meters
# #     "ycenter": 257,         # pixel index
# #     "xcenter": 515,         # pixel index
# #     "xpixels": 1030,        # number of pixels (X)
# #     "ypixels": 514,         # number of pixels (Y)
# #     "energy": 11.470,       # KeV
# #     "wavelength": None,     # or float
# # }

# # # ────────────────────────── small helpers ──────────────────────────
# # def _HSeparator(height: int = 10):
# #     w = Label(value="")
# #     try:
# #         w.native.setFrameShape(QtWidgets.QFrame.HLine)
# #         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
# #         w.native.setLineWidth(1)
# #         w.native.setFixedHeight(height)
# #     except Exception:
# #         pass
# #     return w

# # def _parse_scan_list(text: str) -> List[int]:
# #     """Accepts: '17, 18-22,30' → [17,18,19,20,21,22,30]"""
# #     if not text or not text.strip():
# #         return []
# #     out = set()
# #     for part in re.split(r"[,\s]+", text.strip()):
# #         if not part:
# #             continue
# #         if "-" in part:
# #             a, b = part.split("-", 1)
# #             a, b = a.strip(), b.strip()
# #             if a.isdigit() and b.isdigit():
# #                 lo, hi = int(a), int(b)
# #                 if lo > hi:
# #                     lo, hi = hi, lo
# #                 out.update(range(lo, hi + 1))
# #             else:
# #                 raise ValueError(f"Bad scan range: '{part}'")
# #         else:
# #             if part.isdigit():
# #                 out.add(int(part))
# #             else:
# #                 raise ValueError(f"Bad scan id: '{part}'")
# #     return sorted(out)

# # def _parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
# #     """
# #     'x,y,z' where y/z may be omitted or '*'.
# #     '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
# #     """
# #     if text is None:
# #         return (200, None, None)
# #     text = text.strip()
# #     if not text:
# #         return (200, None, None)
# #     parts = [p.strip() for p in text.split(",")]
# #     if len(parts) == 1:
# #         parts += ["*", "*"]
# #     if len(parts) != 3:
# #         raise ValueError("Grid must be 'x,y,z' (y/z may be '*' or empty)")
# #     def _one(p):
# #         if p in ("*", "", None):
# #             return None
# #         if not p.isdigit():
# #             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
# #         v = int(p)
# #         if v <= 0:
# #             raise ValueError("Grid sizes must be > 0")
# #         return v
# #     x, y, z = (_one(p) for p in parts)
# #     if x is None:
# #         raise ValueError("Grid x (first value) is required")
# #     return x, y, z

# # def _open_tiffs_in_napari(tiff_dir: str):
# #     """Open TIFF stack(s) in napari via glob patterns."""
# #     viewer = napari.Viewer()
# #     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
# #     opened = False
# #     for pat in patterns:
# #         try:
# #             viewer.open(os.path.join(tiff_dir, pat))
# #             opened = True
# #         except Exception:
# #             pass
# #     if not opened:
# #         show_error("No TIFF files found (patterns: *.tif, *.tiff).")
# #     return viewer

# # # ────────────────────────── YAML handling for parameters ──────────────────────────
# # def _prefill_params_from_yaml(setup_path: str, widgets: Dict[str, Any]):
# #     """Load ExperimentSetup from YAML (if present) and prefill widgets."""
# #     if not (yaml and setup_path and os.path.isfile(setup_path)):
# #         return
# #     try:
# #         data = yaml.safe_load(open(setup_path, "r", encoding="utf-8")) or {}
# #         exp = {}
# #         if isinstance(data, dict):
# #             exp = data.get("ExperimentSetup") or data.get("experiment_setup") or {}
# #         if not isinstance(exp, dict):
# #             return
# #         for k, w in widgets.items():
# #             if k in exp:
# #                 val = exp[k]
# #                 try:
# #                     if isinstance(w, (FloatSpinBox,)):
# #                         if val is None or val == "None":
# #                             continue
# #                         w.value = float(val)
# #                     elif isinstance(w, (SpinBox,)):
# #                         w.value = int(val)
# #                     elif isinstance(w, (LineEdit,)):
# #                         w.value = "" if val is None else str(val)
# #                 except Exception:
# #                     # ignore bad types; leave defaults
# #                     pass
# #     except Exception:
# #         pass

# # def _collect_param_values(widgets: Dict[str, Any]) -> Dict[str, Any]:
# #     """Read current values from the parameter widgets."""
# #     out: Dict[str, Any] = {}
# #     for k, w in widgets.items():
# #         if isinstance(w, FloatSpinBox):
# #             out[k] = float(w.value)
# #         elif isinstance(w, SpinBox):
# #             out[k] = int(w.value)
# #         elif isinstance(w, LineEdit):
# #             txt = (w.value or "").strip()
# #             if txt == "" or txt.lower() == "none":
# #                 out[k] = None
# #             else:
# #                 # try to coerce to float, else keep as string
# #                 try:
# #                     out[k] = float(txt)
# #                 except Exception:
# #                     out[k] = txt
# #         else:
# #             out[k] = w.value
# #     return out

# # def _write_edited_setup_yaml(original_yaml: str, param_overrides: Dict[str, Any]) -> str:
# #     """Merge overrides into ExperimentSetup and write to a temp YAML; return path."""
# #     if yaml is None:
# #         return original_yaml
# #     try:
# #         base = {}
# #         if original_yaml and os.path.isfile(original_yaml):
# #             base = yaml.safe_load(open(original_yaml, "r", encoding="utf-8")) or {}
# #         if not isinstance(base, dict):
# #             base = {}
# #         exp = base.get("ExperimentSetup") or base.get("experiment_setup") or {}
# #         if not isinstance(exp, dict):
# #             exp = {}
# #         exp.update(param_overrides)
# #         base["ExperimentSetup"] = exp
# #         tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
# #         out_path = os.path.join(tmpdir, "edited_setup.yaml")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             yaml.safe_dump(base, f, sort_keys=False)
# #         return out_path
# #     except Exception as e:
# #         show_error(f"Failed to write edited setup YAML: {e}")
# #         return original_yaml


# # # ────────────────────────── build the app ──────────────────────────
# # def main():
# #     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# #     # ── Column 1: Load Data + your editable parameters
# #     spec_file = FileEdit(mode="r", label="SPEC file")
# #     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
# #     tiff_dir   = FileEdit(mode="d", label="TIFF folder")
# #     scans      = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
# #     only_hkl   = CheckBox(label="Only HKL scans", value=True)
# #     btn_load   = PushButton(text="Load Data")
# #     btn_tiff   = PushButton(text="View TIFFs in napari")

# #     # Parameter editors (type-aware widgets)
# #     title_params = Label(value="<b>Experiment / detector parameters (editable)</b>")
# #     distance_w   = FloatSpinBox(label="distance (m)",  min=0.0, max=10.0, step=1e-5, value=DETECTOR_DEFAULTS["distance"])
# #     pitch_w      = FloatSpinBox(label="pitch (m)",     min=0.0, max=1e-2, step=1e-6, value=DETECTOR_DEFAULTS["pitch"])
# #     ycenter_w    = SpinBox(label="ycenter (px)",       min=0, max=10000, step=1, value=DETECTOR_DEFAULTS["ycenter"])
# #     xcenter_w    = SpinBox(label="xcenter (px)",       min=0, max=10000, step=1, value=DETECTOR_DEFAULTS["xcenter"])
# #     xpixels_w    = SpinBox(label="xpixels",            min=1, max=100000, step=1, value=DETECTOR_DEFAULTS["xpixels"])
# #     ypixels_w    = SpinBox(label="ypixels",            min=1, max=100000, step=1, value=DETECTOR_DEFAULTS["ypixels"])
# #     energy_w     = FloatSpinBox(label="energy (keV)",  min=0.0, max=200.0, step=0.001, value=DETECTOR_DEFAULTS["energy"])
# #     wavelength_w = LineEdit(label="wavelength (Å or None)", value="None" if DETECTOR_DEFAULTS["wavelength"] is None else str(DETECTOR_DEFAULTS["wavelength"]))

# #     PARAM_WIDGETS: Dict[str, Any] = {
# #         "distance": distance_w,
# #         "pitch": pitch_w,
# #         "ycenter": ycenter_w,
# #         "xcenter": xcenter_w,
# #         "xpixels": xpixels_w,
# #         "ypixels": ypixels_w,
# #         "energy": energy_w,
# #         "wavelength": wavelength_w,
# #     }

# #     col1_cont = Container(
# #         layout="vertical",
# #         widgets=[
# #             spec_file, setup_file, tiff_dir, scans, only_hkl,
# #             btn_load, btn_tiff, _HSeparator(),
# #             title_params,
# #             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
# #         ],
# #     )

# #     # ── Column 2: Build + Regrid
# #     ub_2pi          = CheckBox(label="UB includes 2π", value=True)
# #     center_one_based= CheckBox(label="1-based center", value=False)
# #     btn_build       = PushButton(text="Build RSM Map")

# #     space       = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
# #     grid_shape  = LineEdit(label="Grid (x,y,z)", value="200,*,*")
# #     fuzzy       = CheckBox(label="Fuzzy gridder", value=True)
# #     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
# #     normalize   = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
# #     btn_regrid  = PushButton(text="Regrid")

# #     col2_cont = Container(
# #         layout="vertical",
# #         widgets=[
# #             Label(value="<b>Build options</b>"),
# #             ub_2pi, center_one_based, btn_build,
# #             _HSeparator(),
# #             Label(value="<b>Regrid options</b>"),
# #             space, grid_shape, fuzzy, fuzzy_width, normalize, btn_regrid,
# #         ],
# #     )

# #     # ── Column 3: View
# #     log_view  = CheckBox(label="Log view", value=True)
# #     cmap      = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
# #     rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
# #     contrast  = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
# #     btn_view  = PushButton(text="View RSM")
# #     status    = TextEdit(label="Status", value="")
# #     try:
# #         status.native.setReadOnly(True)
# #         status.native.setMinimumHeight(120)
# #     except Exception:
# #         pass

# #     col3_cont = Container(
# #         layout="vertical",
# #         widgets=[log_view, cmap, rendering, contrast, btn_view, _HSeparator(), status],
# #     )

# #     # ── wrap each column with a titled QWidget and place in QSplitter
# #     def _wrap(title: str, container: Container) -> QtWidgets.QWidget:
# #         host = QtWidgets.QWidget()
# #         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
# #         t = QtWidgets.QLabel(f"<b>{title}</b>")
# #         v.addWidget(t)
# #         v.addWidget(container.native, 1)
# #         return host

# #     w_col1 = _wrap("① Load Data + Parameters", col1_cont)
# #     w_col2 = _wrap("② Build RSM Map  &  ③ Regrid", col2_cont)
# #     w_col3 = _wrap("④ View RSM", col3_cont)

# #     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
# #     splitter.addWidget(w_col1)
# #     splitter.addWidget(w_col2)
# #     splitter.addWidget(w_col3)
# #     splitter.setHandleWidth(2)
# #     splitter.setChildrenCollapsible(False)
# #     splitter.setStretchFactor(0, 1)
# #     splitter.setStretchFactor(1, 1)
# #     splitter.setStretchFactor(2, 1)
# #     splitter.setSizes([400, 400, 400])

# #     win = QtWidgets.QMainWindow()
# #     win.setWindowTitle("RSM3D (Qt + magicgui) — Parameters in Column 1")
# #     win.setCentralWidget(splitter)
# #     win.resize(1280, 720)
# #     win.show()

# #     # ── App state
# #     state: Dict[str, Any] = dict(
# #         loader=None,
# #         builder=None,
# #         Q=None, hkl=None, intensity=None,
# #         grid=None, edges=None,
# #         edited_setup_path=None,
# #     )

# #     # ── status helper
# #     def set_status(msg: str):
# #         status.value = msg
# #         try:
# #             show_info(msg)
# #         except Exception:
# #             pass

# #     # ── button handlers
# #     def on_view_tiffs():
# #         d = tiff_dir.value
# #         if not d or not os.path.isdir(d):
# #             show_error("Please select a valid TIFF folder first.")
# #             return
# #         _open_tiffs_in_napari(d)
# #         set_status("Opened TIFFs in napari.")

# #     def on_load():
# #         try:
# #             if not spec_file.value or not os.path.isfile(spec_file.value):
# #                 raise FileNotFoundError("Select a valid SPEC file.")
# #             if setup_file.value and not os.path.isfile(setup_file.value):
# #                 raise FileNotFoundError("YAML setup path does not exist.")
# #             if not tiff_dir.value or not os.path.isdir(tiff_dir.value):
# #                 raise NotADirectoryError("Select a valid TIFF folder.")
# #             scan_list = _parse_scan_list(scans.value or "")
# #             if not scan_list:
# #                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

# #             set_status(f"Loading scans {scan_list}…")
# #             loader = RSMDataLoader(
# #                 spec_file.value,
# #                 setup_file.value,
# #                 tiff_dir.value,
# #                 selected_scans=scan_list,
# #                 process_hklscan_only=bool(only_hkl.value),
# #             )
# #             loader.load()

# #             # Prefill parameter widgets from YAML if present
# #             _prefill_params_from_yaml(setup_file.value, PARAM_WIDGETS)

# #             state.update(loader=loader, builder=None, Q=None, hkl=None, intensity=None, grid=None, edges=None, edited_setup_path=None)
# #             set_status("Data loaded. Parameters are editable in column 1.")
# #         except Exception as e:
# #             show_error(f"Load error: {e}")
# #             set_status(f"Load failed: {e}")

# #     def on_build():
# #         try:
# #             if state["loader"] is None:
# #                 raise RuntimeError("Load data first.")

# #             # Collect parameter edits and write a temp YAML with ExperimentSetup overrides
# #             overrides = _collect_param_values(PARAM_WIDGETS)
# #             edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
# #             state["edited_setup_path"] = edited_yaml

# #             # Rebuild loader to use edited YAML
# #             scan_list = _parse_scan_list(scans.value or "")
# #             loader2 = RSMDataLoader(
# #                 spec_file.value,
# #                 edited_yaml,
# #                 tiff_dir.value,
# #                 selected_scans=scan_list,
# #                 process_hklscan_only=bool(only_hkl.value),
# #             )
# #             set_status("Reloading with edited ExperimentSetup…")
# #             loader2.load()

# #             set_status("Computing Q/HKL/intensity…")
# #             builder = RSMBuilder(
# #                 loader2,
# #                 ub_includes_2pi=bool(ub_2pi.value),
# #                 center_is_one_based=bool(center_one_based.value),
# #             )
# #             Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)

# #             state.update(loader=loader2, builder=builder, Q=Q_samp, hkl=hkl_arr, intensity=intensity_arr, grid=None, edges=None)
# #             set_status("RSM map built.")
# #         except Exception as e:
# #             show_error(f"Build error: {e}")
# #             set_status(f"Build failed: {e}")

# #     def on_regrid():
# #         try:
# #             builder = state["builder"]
# #             if builder is None:
# #                 raise RuntimeError("Build the RSM map first.")
# #             gx, gy, gz = _parse_grid_shape(grid_shape.value)
# #             set_status(f"Regridding to {space.value.upper()} grid {(gx, gy, gz)}…")
# #             kw = dict(
# #                 space=space.value,
# #                 grid_shape=(gx, gy, gz),
# #                 fuzzy=bool(fuzzy.value),
# #                 normalize=normalize.value,
# #                 stream=True,
# #             )
# #             if fuzzy.value and float(fuzzy_width.value) > 0:
# #                 kw["width"] = float(fuzzy_width.value)
# #             grid, edges = builder.regrid_xu(**kw)
# #             state.update(grid=grid, edges=edges)
# #             set_status("Regrid completed.")
# #         except Exception as e:
# #             show_error(f"Regrid error: {e}")
# #             set_status(f"Regrid failed: {e}")

# #     def on_view():
# #         try:
# #             if state["grid"] is None or state["edges"] is None:
# #                 raise RuntimeError("Regrid first.")
# #             try:
# #                 lo_str, hi_str = [p.strip() for p in (contrast.value or "1,99.8").split(",")]
# #                 pct_lo, pct_hi = float(lo_str), float(hi_str)
# #                 if not (0 <= pct_lo < pct_hi <= 100):
# #                     raise ValueError
# #             except Exception:
# #                 raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

# #             viz = RSMNapariViewer(
# #                 state["grid"],
# #                 state["edges"],
# #                 space=space.value,
# #                 name="RSM3D",
# #                 log_view=bool(log_view.value),
# #                 contrast_percentiles=(pct_lo, pct_hi),
# #                 cmap=cmap.value,
# #                 rendering=rendering.value,
# #             )
# #             viz.launch()  # returns napari.Viewer
# #             set_status("RSM viewer opened.")
# #         except Exception as e:
# #             show_error(f"View error: {e}")
# #             set_status(f"View failed: {e}")

# #     # connect
# #     btn_tiff.clicked.connect(on_view_tiffs)
# #     btn_load.clicked.connect(on_load)
# #     btn_build.clicked.connect(on_build)
# #     btn_regrid.clicked.connect(on_regrid)
# #     btn_view.clicked.connect(on_view)

# #     # go
# #     sys.exit(app.exec_())


# # if __name__ == "__main__":
# #     main()

# # #!/usr/bin/env python3
# # """
# # RSM3D app: Qt main window + magicgui panels inside a 3-pane QSplitter.

# # Columns:
# #   [1] Load Data  -> choose files/scans; editable ExperimentSetup; preview TIFFs
# #   [2] Build + Regrid -> compute RSM, then regrid controls
# #   [3] View RSM   -> viewer controls + status

# # Requirements (example):
# #   pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
# # """

# # from __future__ import annotations

# # import os
# # import re
# # import sys
# # import tempfile
# # from typing import Any, Dict, List, Tuple

# # from qtpy import QtCore, QtWidgets
# # import napari
# # from napari.utils.notifications import show_info, show_error

# # from magicgui.widgets import (
# #     Container, Label,
# #     FileEdit, TextEdit, LineEdit,
# #     CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
# # )

# # # Optional YAML for ExperimentSetup parsing
# # try:
# #     import yaml  # type: ignore
# # except Exception:  # pragma: no cover
# #     yaml = None

# # from rsm3d.data_io import RSMDataLoader
# # from rsm3d.rsm3d     import RSMBuilder
# # from rsm3d.data_viz  import RSMNapariViewer


# # # ────────────────────────── helpers ──────────────────────────
# # def _parse_scan_list(text: str) -> List[int]:
# #     """Accepts: '17, 18-22,30' → [17,18,19,20,21,22,30]"""
# #     if not text or not text.strip():
# #         return []
# #     out = set()
# #     for part in re.split(r"[,\s]+", text.strip()):
# #         if not part:
# #             continue
# #         if "-" in part:
# #             a, b = part.split("-", 1)
# #             a, b = a.strip(), b.strip()
# #             if a.isdigit() and b.isdigit():
# #                 lo, hi = int(a), int(b)
# #                 if lo > hi:
# #                     lo, hi = hi, lo
# #                 out.update(range(lo, hi + 1))
# #             else:
# #                 raise ValueError(f"Bad scan range: '{part}'")
# #         else:
# #             if part.isdigit():
# #                 out.add(int(part))
# #             else:
# #                 raise ValueError(f"Bad scan id: '{part}'")
# #     return sorted(out)


# # def _parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
# #     """
# #     'x,y,z' where y/z may be omitted or '*'.
# #     '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
# #     """
# #     if text is None:
# #         return (200, None, None)
# #     text = text.strip()
# #     if not text:
# #         return (200, None, None)
# #     parts = [p.strip() for p in text.split(",")]
# #     if len(parts) == 1:
# #         parts += ["*", "*"]
# #     if len(parts) != 3:
# #         raise ValueError("Grid must be 'x,y,z' (y/z may be '*' or empty)")
# #     def _one(p):
# #         if p in ("*", "", None):
# #             return None
# #         if not p.isdigit():
# #             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
# #         v = int(p)
# #         if v <= 0:
# #             raise ValueError("Grid sizes must be > 0")
# #         return v
# #     x, y, z = (_one(p) for p in parts)
# #     if x is None:
# #         raise ValueError("Grid x (first value) is required")
# #     return x, y, z


# # def _open_tiffs_in_napari(tiff_dir: str):
# #     """Open TIFF stack(s) in napari via glob patterns."""
# #     viewer = napari.Viewer()
# #     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
# #     opened = False
# #     for pat in patterns:
# #         try:
# #             viewer.open(os.path.join(tiff_dir, pat))
# #             opened = True
# #         except Exception:
# #             pass
# #     if not opened:
# #         show_error("No TIFF files found (patterns: *.tif, *.tiff).")
# #     return viewer


# # # ────────────── dynamic ExperimentSetup editor (editable) ─────────────
# # def _widget_for_value(key: str, val: Any):
# #     """Return a magicgui widget appropriate for type(val)."""
# #     label = key
# #     if isinstance(val, bool):
# #         w = CheckBox(label=label, value=bool(val))
# #     elif isinstance(val, int):
# #         w = SpinBox(label=label, min=-10_000_000, max=10_000_000, step=1, value=int(val))
# #     elif isinstance(val, float):
# #         w = FloatSpinBox(label=label, min=-1e12, max=1e12, step=0.1, value=float(val))
# #     else:
# #         # serialize lists/dicts to YAML-ish text for edit
# #         if isinstance(val, (list, dict)):
# #             try:
# #                 txt = yaml.safe_dump(val, sort_keys=False) if yaml else str(val)
# #             except Exception:
# #                 txt = str(val)
# #             w = TextEdit(label=label, value=txt)
# #         else:
# #             w = LineEdit(label=label, value=str(val))
# #     return w


# # def _build_setup_editor(exp_setup: Dict[str, Any]) -> Tuple[Container, Dict[str, Any]]:
# #     """Build a vertical container of editable widgets for ExperimentSetup."""
# #     fields: Dict[str, Any] = {}
# #     widgets: List = [Label(value="ExperimentSetup (editable)"), _HSeparator()]
# #     for k in sorted(exp_setup.keys()):
# #         w = _widget_for_value(k, exp_setup[k])
# #         widgets.append(w)
# #         fields[k] = exp_setup[k]
# #     col = Container(layout="vertical", widgets=widgets)
# #     # Make it scrollable by putting into a QWidget with a scroll area (Qt-side)
# #     try:
# #         col.native.setMinimumWidth(320)
# #     except Exception:
# #         pass
# #     return col, fields


# # def _collect_setup_overrides(editor_container: Container) -> Dict[str, Any]:
# #     """Read current values from editor_container widgets."""
# #     overrides: Dict[str, Any] = {}
# #     for w in editor_container:
# #         if isinstance(w, Label):  # skip title/separators
# #             continue
# #         k = getattr(w, "label", None) or getattr(w, "name", None)
# #         if not k:
# #             continue
# #         if isinstance(w, CheckBox):
# #             overrides[k] = bool(w.value)
# #         elif isinstance(w, SpinBox):
# #             overrides[k] = int(w.value)
# #         elif isinstance(w, FloatSpinBox):
# #             overrides[k] = float(w.value)
# #         elif isinstance(w, LineEdit):
# #             overrides[k] = w.value
# #         elif isinstance(w, TextEdit):
# #             txt = w.value or ""
# #             if yaml:
# #                 try:
# #                     parsed = yaml.safe_load(txt)
# #                     overrides[k] = parsed
# #                 except Exception:
# #                     overrides[k] = txt
# #             else:
# #                 overrides[k] = txt
# #         else:
# #             overrides[k] = w.value
# #     return overrides


# # def _load_experiment_setup(setup_file: str) -> Dict[str, Any]:
# #     """Return ExperimentSetup dict (or empty) from YAML file."""
# #     if not setup_file or not os.path.isfile(setup_file) or yaml is None:
# #         return {}
# #     try:
# #         data = yaml.safe_load(open(setup_file, "r", encoding="utf-8"))
# #         if isinstance(data, dict):
# #             exp = data.get("ExperimentSetup") or data.get("experiment_setup")
# #             if isinstance(exp, dict):
# #                 return exp
# #     except Exception:
# #         pass
# #     return {}


# # def _write_edited_setup_yaml(original_yaml: str, overrides: Dict[str, Any]) -> str:
# #     """Write a temp YAML with ExperimentSetup merged with overrides; return path."""
# #     if yaml is None:
# #         return original_yaml
# #     try:
# #         base = {}
# #         if original_yaml and os.path.isfile(original_yaml):
# #             base = yaml.safe_load(open(original_yaml, "r", encoding="utf-8")) or {}
# #         if not isinstance(base, dict):
# #             base = {}
# #         exp = base.get("ExperimentSetup") or base.get("experiment_setup") or {}
# #         if not isinstance(exp, dict):
# #             exp = {}
# #         exp.update(overrides)
# #         base["ExperimentSetup"] = exp
# #         tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
# #         out_path = os.path.join(tmpdir, "edited_setup.yaml")
# #         with open(out_path, "w", encoding="utf-8") as f:
# #             yaml.safe_dump(base, f, sort_keys=False)
# #         return out_path
# #     except Exception as e:
# #         show_error(f"Failed to write edited setup YAML: {e}")
# #         return original_yaml


# # # ────────────────────────── tiny UI helpers ──────────────────────────
# # def _HSeparator(height: int = 10):
# #     w = Label(value="")
# #     try:
# #         w.native.setFrameShape(QtWidgets.QFrame.HLine)
# #         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
# #         w.native.setLineWidth(1)
# #         w.native.setFixedHeight(height)
# #     except Exception:
# #         pass
# #     return w


# # def _wrap_in_column(widget: Container, title: str) -> QtWidgets.QWidget:
# #     """Wrap a magicgui Container in a QWidget with margins and a title."""
# #     host = QtWidgets.QWidget()
# #     lay = QtWidgets.QVBoxLayout(host)
# #     lay.setContentsMargins(8, 8, 8, 8)
# #     lay.setSpacing(6)
# #     t = QtWidgets.QLabel(f"<b>{title}</b>")
# #     lay.addWidget(t)
# #     lay.addWidget(widget.native, 1)
# #     return host


# # # ────────────────────────── build the app ──────────────────────────
# # def main():
# #     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# #     # ── Column 1: Load Data + editable ExperimentSetup
# #     spec_file = FileEdit(mode="r", label="SPEC file")
# #     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
# #     tiff_dir = FileEdit(mode="d", label="TIFF folder")
# #     scans = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
# #     only_hkl = CheckBox(label="Only HKL scans", value=True)
# #     btn_load = PushButton(text="Load Data")
# #     btn_tiff = PushButton(text="View TIFFs in napari")

# #     exp_editor_placeholder = Container(
# #         layout="vertical",
# #         widgets=[_HSeparator(), Label(value="ExperimentSetup editor appears here after Load")],
# #     )

# #     col1_cont = Container(
# #         layout="vertical",
# #         widgets=[
# #             spec_file, setup_file, tiff_dir, scans, only_hkl,
# #             btn_load, btn_tiff,
# #             _HSeparator(), exp_editor_placeholder,
# #         ],
# #     )

# #     # ── Column 2: Build + Regrid (combined)
# #     # Build
# #     ub_2pi = CheckBox(label="UB includes 2π", value=True)
# #     center_one_based = CheckBox(label="1-based center", value=False)
# #     btn_build = PushButton(text="Build RSM Map")
# #     # Regrid
# #     space = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
# #     grid_shape = LineEdit(label="Grid (x,y,z)", value="200,*,*")
# #     fuzzy = CheckBox(label="Fuzzy gridder", value=True)
# #     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
# #     normalize = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
# #     btn_regrid = PushButton(text="Regrid")

# #     col2_cont = Container(
# #         layout="vertical",
# #         widgets=[
# #             Label(value="Build options"),
# #             ub_2pi, center_one_based, btn_build,
# #             _HSeparator(),
# #             Label(value="Regrid options"),
# #             space, grid_shape, fuzzy, fuzzy_width, normalize, btn_regrid,
# #         ],
# #     )

# #     # ── Column 3: View RSM
# #     log_view = CheckBox(label="Log view", value=True)
# #     cmap = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
# #     rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
# #     contrast = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
# #     btn_view = PushButton(text="View RSM")
# #     status = TextEdit(label="Status", value="")
# #     try:
# #         status.native.setReadOnly(True)
# #         status.native.setMinimumHeight(120)
# #     except Exception:
# #         pass

# #     col3_cont = Container(
# #         layout="vertical",
# #         widgets=[log_view, cmap, rendering, contrast, btn_view, _HSeparator(), status],
# #     )

# #     # ── Wrap each magicgui column into a QWidget and put them in a QSplitter
# #     w_col1 = _wrap_in_column(col1_cont, "① Load Data")
# #     w_col2 = _wrap_in_column(col2_cont, "② Build RSM Map  &  ③ Regrid")
# #     w_col3 = _wrap_in_column(col3_cont, "④ View RSM")

# #     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
# #     splitter.addWidget(w_col1)
# #     splitter.addWidget(w_col2)
# #     splitter.addWidget(w_col3)

# #     # Visible vertical dividers come from QSplitter handles
# #     splitter.setHandleWidth(2)
# #     splitter.setChildrenCollapsible(False)
# #     splitter.setStretchFactor(0, 1)
# #     splitter.setStretchFactor(1, 1)
# #     splitter.setStretchFactor(2, 1)
# #     # Give equal initial sizes
# #     splitter.setSizes([400, 400, 400])

# #     # ── Main window
# #     win = QtWidgets.QMainWindow()
# #     win.setWindowTitle("RSM3D (Qt + magicgui) — 3 Columns")
# #     win.setCentralWidget(splitter)
# #     win.resize(1280, 720)
# #     win.show()

# #     # ── App state (shared)
# #     state: Dict[str, Any] = dict(
# #         loader=None,               # RSMDataLoader
# #         builder=None,              # RSMBuilder
# #         Q=None, hkl=None, intensity=None,
# #         grid=None, edges=None,
# #         edited_setup_path=None,    # path to temp edited YAML (if any)
# #         exp_editor_container=exp_editor_placeholder,
# #         exp_editor_loaded=False,
# #     )

# #     # ── status helper
# #     def set_status(msg: str):
# #         status.value = msg
# #         try:
# #             show_info(msg)
# #         except Exception:
# #             pass

# #     # ── handlers
# #     def on_view_tiffs():
# #         d = tiff_dir.value
# #         if not d or not os.path.isdir(d):
# #             show_error("Please select a valid TIFF folder first.")
# #             return
# #         _open_tiffs_in_napari(d)
# #         set_status("Opened TIFFs in napari.")

# #     def on_load():
# #         try:
# #             if not spec_file.value or not os.path.isfile(spec_file.value):
# #                 raise FileNotFoundError("Select a valid SPEC file.")
# #             if setup_file.value and not os.path.isfile(setup_file.value):
# #                 raise FileNotFoundError("YAML setup path does not exist.")
# #             if not tiff_dir.value or not os.path.isdir(tiff_dir.value):
# #                 raise NotADirectoryError("Select a valid TIFF folder.")
# #             scan_list = _parse_scan_list(scans.value or "")
# #             if not scan_list:
# #                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

# #             set_status(f"Loading scans {scan_list}…")
# #             loader = RSMDataLoader(
# #                 spec_file.value,
# #                 setup_file.value,
# #                 tiff_dir.value,
# #                 selected_scans=scan_list,
# #                 process_hklscan_only=bool(only_hkl.value),
# #             )
# #             loader.load()
# #             state["loader"] = loader
# #             state["builder"] = None
# #             state["Q"] = state["hkl"] = state["intensity"] = None
# #             state["grid"] = state["edges"] = None
# #             state["edited_setup_path"] = None

# #             # Build editable ExperimentSetup UI
# #             exp_dict = _load_experiment_setup(setup_file.value) if setup_file.value else {}
# #             state["exp_editor_container"].widgets = []  # clear area
# #             if exp_dict:
# #                 editor_col, _snapshot = _build_setup_editor(exp_dict)
# #                 state["exp_editor_container"].extend([editor_col])
# #                 state["exp_editor_loaded"] = True
# #                 set_status("Data loaded. ExperimentSetup ready to edit.")
# #             else:
# #                 state["exp_editor_container"].extend([Label(value="(No ExperimentSetup found in YAML)")])
# #                 state["exp_editor_loaded"] = False
# #                 set_status("Data loaded. No ExperimentSetup found in YAML.")
# #         except Exception as e:
# #             show_error(f"Load error: {e}")
# #             set_status(f"Load failed: {e}")

# #     def on_build():
# #         try:
# #             if state["loader"] is None:
# #                 raise RuntimeError("Load data first.")
# #             # If user edited ExperimentSetup, write a temp YAML and rebuild loader
# #             edited_yaml = setup_file.value
# #             if state["exp_editor_loaded"]:
# #                 editor_cols = [w for w in state["exp_editor_container"] if isinstance(w, Container)]
# #                 if editor_cols:
# #                     overrides = _collect_setup_overrides(editor_cols[0])
# #                     edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
# #                     state["edited_setup_path"] = edited_yaml

# #             scan_list = _parse_scan_list(scans.value or "")
# #             loader2 = RSMDataLoader(
# #                 spec_file.value,
# #                 edited_yaml,
# #                 tiff_dir.value,
# #                 selected_scans=scan_list,
# #                 process_hklscan_only=bool(only_hkl.value),
# #             )
# #             set_status("Reloading with edited ExperimentSetup…")
# #             loader2.load()

# #             set_status("Computing Q/HKL/intensity…")
# #             builder = RSMBuilder(
# #                 loader2,
# #                 ub_includes_2pi=bool(ub_2pi.value),
# #                 center_is_one_based=bool(center_one_based.value),
# #             )
# #             Q_samp, hkl_arr, intensity_arr = builder.compute_full(verbose=False)

# #             state["loader"]   = loader2
# #             state["builder"]  = builder
# #             state["Q"]        = Q_samp
# #             state["hkl"]      = hkl_arr
# #             state["intensity"]= intensity_arr
# #             state["grid"]     = None
# #             state["edges"]    = None

# #             set_status("RSM map built.")
# #         except Exception as e:
# #             show_error(f"Build error: {e}")
# #             set_status(f"Build failed: {e}")

# #     def on_regrid():
# #         try:
# #             builder = state["builder"]
# #             if builder is None:
# #                 raise RuntimeError("Build the RSM map first.")
# #             gx, gy, gz = _parse_grid_shape(grid_shape.value)
# #             set_status(f"Regridding to {space.value.upper()} grid {(gx, gy, gz)}…")
# #             kw = dict(
# #                 space=space.value,
# #                 grid_shape=(gx, gy, gz),
# #                 fuzzy=bool(fuzzy.value),
# #                 normalize=normalize.value,
# #                 stream=True,
# #             )
# #             if fuzzy.value and float(fuzzy_width.value) > 0:
# #                 kw["width"] = float(fuzzy_width.value)
# #             grid, edges = builder.regrid_xu(**kw)
# #             state["grid"], state["edges"] = grid, edges
# #             set_status("Regrid completed.")
# #         except Exception as e:
# #             show_error(f"Regrid error: {e}")
# #             set_status(f"Regrid failed: {e}")

# #     def on_view():
# #         try:
# #             if state["grid"] is None or state["edges"] is None:
# #                 raise RuntimeError("Regrid first.")
# #             try:
# #                 lo_str, hi_str = [p.strip() for p in (contrast.value or "1,99.8").split(",")]
# #                 pct_lo, pct_hi = float(lo_str), float(hi_str)
# #                 if not (0 <= pct_lo < pct_hi <= 100):
# #                     raise ValueError
# #             except Exception:
# #                 raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

# #             viz = RSMNapariViewer(
# #                 state["grid"],
# #                 state["edges"],
# #                 space=space.value,
# #                 name="RSM3D",
# #                 log_view=bool(log_view.value),
# #                 contrast_percentiles=(pct_lo, pct_hi),
# #                 cmap=cmap.value,
# #                 rendering=rendering.value,
# #             )
# #             viz.launch()  # returns napari.Viewer
# #             set_status("RSM viewer opened.")
# #         except Exception as e:
# #             show_error(f"View error: {e}")
# #             set_status(f"View failed: {e}")

# #     # ── connect buttons
# #     btn_tiff.clicked.connect(on_view_tiffs)
# #     btn_load.clicked.connect(on_load)
# #     btn_build.clicked.connect(on_build)
# #     btn_regrid.clicked.connect(on_regrid)
# #     btn_view.clicked.connect(on_view)

# #     # ── go
# #     sys.exit(app.exec_())


# # if __name__ == "__main__":
# #     main()