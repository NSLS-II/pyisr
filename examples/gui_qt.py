#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSM3D app — single-YAML persistence (simple, synchronous pipeline)

Changes in this version:
• "View RSM" is ABOVE "Export to VTK" at the left bottom of Column 3
• Clicking the 📂 symbol on the DATA file opens a menu: "Pick File…" or "Pick Folder…"

YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
"""

from __future__ import annotations

import os
import pathlib
import numpy as np
import re
import sys
from typing import Any, Dict, List, Tuple

import yaml
import napari
from qtpy import QtCore, QtWidgets
from napari.utils.notifications import show_error
from magicgui.widgets import (
    CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
    Label, LineEdit, PushButton, SpinBox, TextEdit,
)

from rsm3d.data_io import RSMDataLoader, write_rsm_volume_to_vtr
from rsm3d.data_viz import RSMNapariViewer, IntensityNapariViewer
from rsm3d.rsm3d import RSMBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Theme / QSS
# ─────────────────────────────────────────────────────────────────────────────

APP_QSS = """
QMainWindow { background: #fafafa; }
QGroupBox {
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    margin-top: 12px;
    background: #ffffff;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 6px 10px;
    color: #2c3e50;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.2px;
}
QSplitter::handle {
    background: #e9edf3;
    border-left: 1px solid #d0d4db;
    border-right: 1px solid #ffffff;
}
QLabel { color: #34495e; }
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    border: 1px solid #d4d7dd;
    border-radius: 6px;
    padding: 4px 6px;
    background: #ffffff;
}
QPushButton {
    background: #eef2f7;
    border: 1px solid #d4d7dd;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover { background: #e6ebf3; }
QPushButton:pressed { background: #dfe5ee; }
"""

RUN_ALL_QSS = """
QPushButton#RunAllPrimary {
    background: #ff9800;
    color: #ffffff;
    border: 2px solid #e68900;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 18px;
    font-weight: 800;
}
QPushButton#RunAllPrimary:hover { background: #ffa726; }
QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# YAML utils
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
os.environ.setdefault(
    DEFAULTS_ENV,
    str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
)

def yaml_path() -> str:
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

def ensure_yaml(path: str) -> None:
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seed = {
        "data": {"spec_file": None, "data_file": None, "scans": "", "only_hkl": None},
        "ExperimentSetup": {
            "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
            "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
        },
        "build": {"ub_includes_2pi": None, "center_is_one_based": None},
        "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
        "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
        "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
        "export": {"vtr_path": None},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(seed, f, sort_keys=False)

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    except Exception:
        return {}

def save_yaml(path: str, doc: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    except Exception as e:
        show_error(f"Failed to write YAML: {e}")

def as_path_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return os.fspath(v)
    except TypeError:
        return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hsep(height: int = 10) -> Label:
    w = Label(value="")
    try:
        w.native.setFrameShape(QtWidgets.QFrame.HLine)
        w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedHeight(height)
    except Exception:
        pass
    return w

def q_hsep(height: int = 10) -> QtWidgets.QWidget:
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    line.setLineWidth(1)
    line.setFixedHeight(height)
    return line

def parse_scan_list(text: str) -> List[int]:
    if not text or not text.strip():
        return []
    out: set[int] = set()
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
    def one(p: str) -> int | None:
        if p in ("*", "", None):
            return None
        if not p.isdigit():
            raise ValueError(f"Grid size must be integer or '*', got '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid sizes must be > 0")
        return v
    return tuple(one(p) for p in parts)  # type: ignore[return-value]

def open_intensity_in_napari(path: str):
    """Open a single data file or (fallback) load all tiffs from a directory."""
    viewer = napari.Viewer()
    p = path.strip()
    if os.path.isfile(p):
        try:
            viewer.open(p)
            return viewer
        except Exception:
            pass
    # Fallback: scan directory for TIFFs
    d = p if os.path.isdir(p) else os.path.dirname(p)
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    opened = False
    for pat in patterns:
        try:
            viewer.open(os.path.join(d, pat))
            opened = True
        except Exception:
            pass
    if not opened:
        show_error("Could not open intensity. Provide a readable file or a folder with TIFF(s).")
    return viewer

def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)
    lay.addWidget(inner_widget)
    return box

def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    wrapper = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(wrapper)
    v.setContentsMargins(8, 8, 8, 8)
    v.setSpacing(8)
    v.addWidget(inner)
    sc = QtWidgets.QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QtWidgets.QFrame.NoFrame)
    sc.setWidget(wrapper)
    return sc

def set_file_button_symbol(fe: FileEdit, symbol: str = "📂") -> QtWidgets.QPushButton | None:
    """Replace 'Select file' text with a simple symbol and return the button."""
    try:
        for btn in fe.native.findChildren(QtWidgets.QPushButton):
            btn.setText(symbol)
            btn.setMinimumWidth(32)
            btn.setMaximumWidth(36)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            return btn
    except Exception:
        pass
    return None

def attach_dual_picker(fe: FileEdit, button: QtWidgets.QPushButton) -> None:
    """
    Attach a drop-down menu to the FileEdit button that lets the user pick
    either a single file or a folder.
    """
    menu = QtWidgets.QMenu(button)
    act_file = menu.addAction("Pick File…")
    act_dir  = menu.addAction("Pick Folder…")

    def pick_file():
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(button, "Select file", start)
        if path:
            fe.value = path

    def pick_dir():
        start = as_path_str(fe.value).strip() or os.path.expanduser("~")
        path = QtWidgets.QFileDialog.getExistingDirectory(button, "Select folder", start)
        if path:
            fe.value = path

    act_file.triggered.connect(pick_file)
    act_dir.triggered.connect(pick_dir)

    # Show the menu on button click (keep the nice single-symbol look)
    def on_click():
        menu.exec_(button.mapToGlobal(QtCore.QPoint(0, button.height())))
    button.clicked.connect(on_click)


# ─────────────────────────────────────────────────────────────────────────────
# App (synchronous step-by-step pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

    ypath = yaml_path()
    ensure_yaml(ypath)

    # ── Build UI ────────────────────────────────────────────────────────────
    # Column 1: Data & Setup
    spec_file_w = FileEdit(mode="r", label="SPEC file")
    data_file_w = FileEdit(mode="r", label="DATA file")  # will support file OR folder via menu
    scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
    scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
    only_hkl_w  = CheckBox(label="Only HKL scans")

    # turn file buttons into simple symbols + add dual picker on DATA file
    _spec_btn = set_file_button_symbol(spec_file_w, "📂")
    _data_btn = set_file_button_symbol(data_file_w, "📂")
    if _data_btn is not None:
        attach_dual_picker(data_file_w, _data_btn)

    title_params = Label(value="<b>Experiment Setup</b>")
    distance_w   = FloatSpinBox(label="Distance (m)", min=-1e9, max=1e9, step=1e-6)
    pitch_w      = FloatSpinBox(label="Pitch (m)",    min=-1e9, max=1e9, step=1e-9)
    ypixels_w    = SpinBox(label="Detector H (px)", min=0, max=10_000_000, step=1)
    xpixels_w    = SpinBox(label="Detector W (px)", min=0, max=10_000_000, step=1)
    ycenter_w    = SpinBox(label="BeamCenter H (px)", min=0,max=10_000_000, step=1)
    xcenter_w    = SpinBox(label="BeamCenter W (px)", min=0,max=10_000_000, step=1)

    energy_w     = FloatSpinBox(label="Energy (keV)", min=-1e9, max=1e9, step=1e-3)
    wavelength_w = FloatSpinBox(label="Wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
    wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

    # Column 1 bottom buttons
    btn_load = PushButton(text="📂 Load Data")
    btn_intensity = PushButton(text="📈 View Intensity")
    btn_row1 = QtWidgets.QWidget()
    row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
    row1.addWidget(btn_load.native); row1.addWidget(btn_intensity.native); row1.addStretch(1)

    col1 = Container(
        layout="vertical",
        widgets=[
            spec_file_w, data_file_w, scans_w, only_hkl_w,
            hsep(), title_params,
            distance_w, pitch_w, ypixels_w, xpixels_w, ycenter_w, xcenter_w, energy_w, wavelength_w,
            hsep(),
        ],
    )

    # Column 2: Build & Regrid
    title_build = Label(value="<b>RSM Builder</b>")
    ub_2pi_w    = CheckBox(label="UB includes 2π")
    center_one_based_w = CheckBox(label="1-based center")

    title_regrid = Label(value="<b>Grid Settings</b>")
    space_w      = ComboBox(label="Space", choices=["hkl", "q"])
    grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
    grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
    fuzzy_w      = CheckBox(label="Fuzzy gridder")
    fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
    normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

    title_crop   = Label(value="<b>Crop Settings</b>")
    crop_enable_w= CheckBox(label="Crop before regrid")
    y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
    y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
    x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
    x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

    # Column 2 bottom buttons
    btn_build   = PushButton(text="🔧 Build RSM Map")
    btn_regrid  = PushButton(text="🧮 Regrid")
    btn_row2 = QtWidgets.QWidget()
    row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
    row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

    col2 = Container(
        layout="vertical",
        widgets=[
            title_build, ub_2pi_w, center_one_based_w,
            hsep(),
            title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
            hsep(),
            title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
            hsep(),
        ],
    )

    # Column 3: View + Status/Progress + Export controls at bottom
    title_view   = Label(value="<b>Napari Viewer</b>")
    log_view_w   = CheckBox(label="Log view")
    cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
    rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
    contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
    contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

    status_label_w = Label(value="Status / Output")
    status_w       = TextEdit(value="")
    try:
        status_w.native.setReadOnly(True)
        status_w.native.setMinimumHeight(220)
    except Exception:
        pass

    progress = QtWidgets.QProgressBar()
    progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

    # Export path in column 3
    export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")
    set_file_button_symbol(export_vtr_w, "📂")

    # Bottom action buttons for Column 3:
    # LEFT: "View RSM" ABOVE "Export to VTK" (vertical)
    # RIGHT: big "Run All"
    btn_view    = PushButton(text="🔭 View RSM")
    btn_export  = PushButton(text="💾 Export to VTK")
    btn_run_all = PushButton(text="▶️ Run All")
    btn_run_all.native.setObjectName("RunAllPrimary")
    btn_run_all.native.setMinimumHeight(64)

    left_bottom = QtWidgets.QWidget()
    vleft = QtWidgets.QVBoxLayout(left_bottom); vleft.setContentsMargins(0,0,0,0); vleft.setSpacing(8)
    vleft.addWidget(btn_view.native)
    vleft.addWidget(btn_export.native)

    btn_row3 = QtWidgets.QWidget()
    row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(12)
    row3.addWidget(left_bottom)     # left column: View above Export
    row3.addStretch(1)
    row3.addWidget(btn_run_all.native)  # right big button

    col3 = Container(
        layout="vertical",
        widgets=[
            title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
            hsep(),
        ],
    )

    # Wrap columns in groups + scroll areas
    g1 = make_group("Data", col1.native)
    g1_lay = g1.layout()
    g1_lay.addStretch(1)
    g1_lay.addWidget(btn_row1)

    g2 = make_group("Build", col2.native)
    g2_lay = g2.layout()
    g2_lay.addStretch(1)
    g2_lay.addWidget(btn_row2)

    g3 = make_group("View", col3.native)
    g3_lay = g3.layout()
    g3_lay.addWidget(q_hsep())
    g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
    g3_lay.addWidget(status_w.native)
    g3_lay.addWidget(progress)
    g3_lay.addWidget(q_hsep())
    g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
    g3_lay.addWidget(export_vtr_w.native)
    g3_lay.addStretch(1)
    g3_lay.addWidget(q_hsep())
    g3_lay.addWidget(btn_row3)  # bottom row: View above Export (left), Run All (right)

    s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
    splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
    splitter.setSizes([440, 440, 440])

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("RSM3D")
    win.setCentralWidget(splitter)
    win.resize(1320, 820)
    status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
    win.show()

    # ── YAML binding (load → UI; UI → save)
    ydoc = load_yaml(ypath)
    widget_map: Dict[str, Dict[str, Any]] = {
        "data": {"spec_file": spec_file_w, "data_file": data_file_w, "scans": scans_w, "only_hkl": only_hkl_w},
        "ExperimentSetup": {
            "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
            "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
        },
        "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
        "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
        "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
        "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
        "export": {"vtr_path": export_vtr_w},
    }

    def set_widget(widget: Any, value: Any) -> None:
        try:
            if value is None:
                return
            if isinstance(widget, (FloatSpinBox, SpinBox)):
                widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
            elif isinstance(widget, CheckBox):
                widget.value = bool(value)
            elif isinstance(widget, ComboBox):
                sval = str(value)
                if sval in widget.choices:
                    widget.value = sval
            elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
                widget.value = str(value)
        except Exception:
            pass

    for section, mapping in widget_map.items():
        vals = ydoc.get(section, {})
        for key, widget in mapping.items():
            set_widget(widget, vals.get(key, None))
    for s in widget_map:
        ydoc.setdefault(s, {})
    save_yaml(ypath, ydoc)

    def val_for_yaml(widget: Any, section: str, key: str) -> Any:
        if section == "ExperimentSetup" and key == "wavelength":
            txt = str(widget.value).strip()
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

    def on_changed(section: str, key: str, widget: Any) -> None:
        ydoc.setdefault(section, {})
        ydoc[section][key] = val_for_yaml(widget, section, key)
        save_yaml(ypath, ydoc)

    for section, mapping in widget_map.items():
        for key, widget in mapping.items():
            widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

    # ── Status / progress helpers (synchronous) ─────────────────────────────
    def pump(ms: int = 0):
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

    def status(msg: str) -> None:
        try:
            status_w.native.append(msg)
        except Exception:
            status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
        status_bar.showMessage(msg, 3000)

    def set_progress(value: int | None, *, busy: bool = False):
        if busy:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, 100)
            progress.setValue(int(value or 0))

    def set_busy(b: bool):
        for btn in (btn_load, btn_intensity, btn_build, btn_regrid, btn_view, btn_run_all, btn_export):
            try:
                btn.native.setEnabled(not b)
            except Exception:
                pass

    # ── App state
    state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

    # ── Actions ─────────────────────────────────────────────────────────────
    def on_view_intensity():
        loader = state.get("loader")
        if loader is None:
            show_error("Load data first.")
            return

        # Optionally coerce to a plain Python list of 2D arrays:
        _, _, df = loader.load()
        frames = list(df.intensity)   # each item is a 2D array

        viewer = IntensityNapariViewer(
            frames,                               # list of 2D frames
            name="Intensity",
            log_view=True,
            contrast_percentiles=(1.0, 99.8),
            cmap="inferno",
            rendering="attenuated_mip",
            add_timeseries=True,
            add_volume=True,
            scale_tzyx=(1.0, 1.0, 1.0),
            pad_value=np.nan,                     # or 0.0 if you prefer black padding
        ).launch()

        state["intensity_viewer"] = viewer

    def on_load() -> None:
        spec = as_path_str(spec_file_w.value).strip()
        dpath = as_path_str(data_file_w.value).strip()
        try:
            scans = parse_scan_list((scans_w.value or "").strip())
        except Exception as e:
            show_error(str(e)); return

        if not spec or not os.path.isfile(spec):
            show_error("Select a valid SPEC file."); return
        if not (os.path.isfile(dpath) or os.path.isdir(dpath)):
            show_error("Select a valid DATA file (or a folder)."); return
        if not scans:
            show_error("Enter at least one scan (e.g. '17, 18-22')."); return

        set_busy(True); set_progress(None, busy=True); status(f"Loading scans {scans}…"); pump(50)
        try:
            tiff_arg = dpath if os.path.isdir(dpath) else (os.path.dirname(dpath) or ".")
            loader = RSMDataLoader(
                spec,
                yaml_path(),
                tiff_arg,
                selected_scans=scans,
                process_hklscan_only=bool(only_hkl_w.value),
            )
            loader.load()
            state["loader"] = loader
            state["builder"] = None
            state["grid"] = state["edges"] = None
            set_progress(25, busy=False); status("Data loaded.")
        except Exception as e:
            show_error(f"Load error: {e}")
            set_progress(0, busy=False); status(f"Load failed: {e}")
        finally:
            set_busy(False)

    def on_build() -> None:
        if state.get("loader") is None:
            show_error("Load data first."); return

        set_busy(True); set_progress(None, busy=True); status("Computing Q/HKL/intensity…"); pump(50)
        try:
            b = RSMBuilder(
                state["loader"],
                ub_includes_2pi=bool(ub_2pi_w.value),
                center_is_one_based=bool(center_one_based_w.value),
            )
            b.compute_full(verbose=False)
            state["builder"] = b
            state["grid"] = state["edges"] = None
            set_progress(50, busy=False); status("RSM map built.")
        except Exception as e:
            show_error(f"Build error: {e}")
            set_progress(40, busy=False); status(f"Build failed: {e}")
        finally:
            set_busy(False)

    def on_regrid() -> None:
        b = state.get("builder")
        if b is None:
            show_error("Build the RSM map first."); return

        try:
            gx, gy, gz = parse_grid_shape(grid_shape_w.value)
        except Exception as e:
            show_error(str(e)); return
        if gx is None:
            show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

        do_crop = bool(crop_enable_w.value)
        ymin, ymax = int(y_min_w.value), int(y_max_w.value)
        xmin, xmax = int(x_min_w.value), int(x_max_w.value)

        set_busy(True); set_progress(None, busy=True)
        status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…"); pump(50)

        try:
            if do_crop:
                if ymin >= ymax or xmin >= xmax:
                    raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
                if state.get("loader") is None:
                    raise RuntimeError("Internal error: loader missing; run Build again.")
                b = RSMBuilder(
                    state["loader"],
                    ub_includes_2pi=bool(ub_2pi_w.value),
                    center_is_one_based=bool(center_one_based_w.value),
                )
                b.compute_full(verbose=False)
                b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

            kwargs = dict(
                space=space_w.value,
                grid_shape=(gx, gy, gz),
                fuzzy=bool(fuzzy_w.value),
                normalize=normalize_w.value,
                stream=True,
            )
            if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
                kwargs["width"] = float(fuzzy_width_w.value)

            grid, edges = b.regrid_xu(**kwargs)
            state["grid"], state["edges"] = grid, edges
            set_progress(75, busy=False); status("Regrid completed.")
        except Exception as e:
            show_error(f"Regrid error: {e}")
            set_progress(60, busy=False); status(f"Regrid failed: {e}")
        finally:
            set_busy(False)

    def on_view() -> None:
        if state.get("grid") is None or state.get("edges") is None:
            show_error("Regrid first."); return
        try:
            lo = float(contrast_lo_w.value); hi = float(contrast_hi_w.value)
            if not (0 <= lo < hi <= 100):
                raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

            set_progress(None, busy=True); status("Opening RSM viewer…"); pump(50)
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
            set_progress(100, busy=False); status("RSM viewer opened.")
        except Exception as e:
            show_error(f"View error: {e}")
            set_progress(80, busy=False); status(f"View failed: {e}")

    def on_export_vtk() -> None:
        if state.get("grid") is None or state.get("edges") is None:
            show_error("Regrid first, then export."); return
        out_path = as_path_str(export_vtr_w.value).strip()
        if not out_path:
            show_error("Choose an output .vtr file path."); return
        if not out_path.lower().endswith(".vtr"):
            out_path += ".vtr"
        try:
            set_busy(True); set_progress(None, busy=True)
            status(f"Exporting VTK (.vtr) → {out_path}"); pump(50)
            write_rsm_volume_to_vtr(state["grid"], state["edges"], out_path, binary=False, compress=True)
            set_progress(100, busy=False); status(f"Exported: {out_path}")
        except Exception as e:
            show_error(f"Export error: {e}")
            set_progress(0, busy=False); status(f"Export failed: {e}")
        finally:
            set_busy(False)

    def on_run_all() -> None:
        btn_run_all.native.setEnabled(False)
        try:
            set_progress(0, busy=False); status("Running pipeline (Load → Build → Regrid → View)…")
            on_load()
            if state.get("loader") is None: return
            on_build()
            if state.get("builder") is None: return
            on_regrid()
            if state.get("grid") is None or state.get("edges") is None: return
            on_view()
            status("Run All completed.")
        finally:
            btn_run_all.native.setEnabled(True)

    # Connect buttons
    btn_intensity.clicked.connect(on_view_intensity)
    btn_load.clicked.connect(on_load)
    btn_build.clicked.connect(on_build)
    btn_regrid.clicked.connect(on_regrid)
    btn_view.clicked.connect(on_view)
    btn_export.clicked.connect(on_export_vtk)
    btn_run_all.clicked.connect(on_run_all)

    # Keep saving YAML on changes
    def on_changed(section: str, key: str, widget: Any) -> None:
        ydoc.setdefault(section, {})
        ydoc[section][key] = (
            None if (section == "ExperimentSetup" and key == "wavelength"
                     and str(widget.value).strip().lower() in {"", "none", "null"})
            else (float(widget.value) if isinstance(widget, FloatSpinBox)
                  else int(widget.value) if isinstance(widget, SpinBox)
                  else bool(widget.value) if isinstance(widget, CheckBox)
                  else str(widget.value))
        )
        save_yaml(ypath, ydoc)

    for section, mapping in widget_map.items():
        for key, widget in mapping.items():
            widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

    # Run
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(exec_fn())


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (simple, synchronous pipeline)

# Adjustments in this version:
# • Keep ONLY ONE "Export to VTK" button
# • Place that Export button together with "View RSM" at the left bottom of column 3
# • "DATA file" kept (single file); file buttons show 📂 symbol
# • "View Intensity" kept; "View RSM" uses 🔭; Run All highlighted & larger

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error
# from magicgui.widgets import (
#     CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
#     Label, LineEdit, PushButton, SpinBox, TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader, write_rsm_volume_to_vtr
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Theme / QSS
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """

# RUN_ALL_QSS = """
# QPushButton#RunAllPrimary {
#     background: #ff9800;
#     color: #ffffff;
#     border: 2px solid #e68900;
#     border-radius: 10px;
#     padding: 14px 20px;
#     font-size: 18px;
#     font-weight: 800;
# }
# QPushButton#RunAllPrimary:hover { background: #ffa726; }
# QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utils
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "data_file": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#         "export": {"vtr_path": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_intensity_in_napari(path: str):
#     """Open a single data file or (fallback) load all tiffs from a directory."""
#     viewer = napari.Viewer()
#     p = path.strip()
#     if os.path.isfile(p):
#         try:
#             viewer.open(p)
#             return viewer
#         except Exception:
#             pass
#     # Fallback: scan directory for TIFFs
#     d = p if os.path.isdir(p) else os.path.dirname(p)
#     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
#     opened = False
#     for pat in patterns:
#         try:
#             viewer.open(os.path.join(d, pat))
#             opened = True
#         except Exception:
#             pass
#     if not opened:
#         show_error("Could not open intensity. Provide a readable file or a folder with TIFF(s).")
#     return viewer

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc

# def set_file_button_symbol(fe: FileEdit, symbol: str = "📂") -> None:
#     """Replace 'Select file' text with a simple symbol."""
#     try:
#         for btn in fe.native.findChildren(QtWidgets.QPushButton):
#             btn.setText(symbol)
#             btn.setMinimumWidth(32)
#             btn.setMaximumWidth(36)
#             btn.setCursor(QtCore.Qt.PointingHandCursor)
#     except Exception:
#         pass


# # ─────────────────────────────────────────────────────────────────────────────
# # App (synchronous step-by-step pipeline)
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)

#     # ── Build UI ────────────────────────────────────────────────────────────
#     # Column 1: Data & Setup
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     data_file_w = FileEdit(mode="r", label="DATA file")  # was "TIFF folder"
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     # turn file buttons into simple symbols
#     set_file_button_symbol(spec_file_w, "📂")
#     set_file_button_symbol(data_file_w, "📂")

#     title_params = Label(value="<b>Experiment Setup</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = FloatSpinBox(label="wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     # Column 1 bottom buttons
#     btn_load = PushButton(text="📂 Load Data")
#     btn_intensity = PushButton(text="📈 View Intensity")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_intensity.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, data_file_w, scans_w, only_hkl_w,
#             hsep(), title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # Column 2: Build & Regrid
#     title_build = Label(value="<b>RSM Builder</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")

#     title_regrid = Label(value="<b>Grid Settings</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     # Column 2 bottom buttons
#     btn_build   = PushButton(text="🔧 Build RSM Map")
#     btn_regrid  = PushButton(text="🧮 Regrid")
#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # Column 3: View + Status/Progress + Export controls at bottom
#     title_view   = Label(value="<b>Napari Viewer</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#     except Exception:
#         pass

#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

#     # Export path (stays in column 3 content), single Export button goes to bottom row
#     export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")
#     set_file_button_symbol(export_vtr_w, "📂")

#     # Bottom action buttons for Column 3 (LEFT: Export + View, RIGHT: Run All)
#     btn_export  = PushButton(text="💾 Export to VTK")    # single export button
#     btn_view    = PushButton(text="🔭 View RSM")
#     btn_run_all = PushButton(text="▶️ Run All")
#     btn_run_all.native.setObjectName("RunAllPrimary")
#     btn_run_all.native.setMinimumHeight(64)

#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(12)
#     row3.addWidget(btn_export.native)  # left
#     row3.addWidget(btn_view.native)    # left
#     row3.addStretch(1)
#     row3.addWidget(btn_run_all.native) # right

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # Wrap columns in groups + scroll areas
#     g1 = make_group("Data", col1.native)
#     g1_lay = g1.layout()
#     g1_lay.addStretch(1)
#     g1_lay.addWidget(btn_row1)

#     g2 = make_group("Build", col2.native)
#     g2_lay = g2.layout()
#     g2_lay.addStretch(1)
#     g2_lay.addWidget(btn_row2)

#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
#     g3_lay.addWidget(export_vtr_w.native)
#     g3_lay.addStretch(1)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)  # bottom row with Export + View (left) and Run All (right)

#     s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
#     splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save)
#     ydoc = load_yaml(ypath)
#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "data_file": data_file_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#         "export": {"vtr_path": export_vtr_w},
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = str(widget.value).strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers (synchronous) ─────────────────────────────
#     def pump(ms: int = 0):
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(msg: str) -> None:
#         try:
#             status_w.native.append(msg)
#         except Exception:
#             status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
#         status_bar.showMessage(msg, 3000)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     def set_busy(b: bool):
#         for btn in (btn_load, btn_intensity, btn_build, btn_regrid, btn_view, btn_run_all, btn_export):
#             try:
#                 btn.native.setEnabled(b)  # noqa: E999 (PySide handles bool)
#             except Exception:
#                 try:
#                     btn.native.setEnabled(not b)
#                 except Exception:
#                     pass

#     # ── App state
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     # ── Actions ─────────────────────────────────────────────────────────────
#     def on_view_intensity() -> None:
#         p = as_path_str(data_file_w.value).strip()
#         if not p:
#             show_error("Please choose a DATA file (or a folder with TIFFs)."); return
#         status(f"Opening intensity from: {p}")
#         pump(50)
#         open_intensity_in_napari(p)
#         status("Opened intensity in napari.")

#     def on_load() -> None:
#         spec = as_path_str(spec_file_w.value).strip()
#         dpath = as_path_str(data_file_w.value).strip()
#         try:
#             scans = parse_scan_list((scans_w.value or "").strip())
#         except Exception as e:
#             show_error(str(e)); return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file."); return
#         if not (os.path.isfile(dpath) or os.path.isdir(dpath)):
#             show_error("Select a valid DATA file (or a folder)."); return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22')."); return

#         set_busy(True); set_progress(None, busy=True); status(f"Loading scans {scans}…"); pump(50)
#         try:
#             tiff_arg = dpath if os.path.isdir(dpath) else (os.path.dirname(dpath) or ".")
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),
#                 tiff_arg,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_progress(25, busy=False); status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_progress(0, busy=False); status(f"Load failed: {e}")
#         finally:
#             set_busy(False)

#     def on_build() -> None:
#         if state.get("loader") is None:
#             show_error("Load data first."); return

#         set_busy(True); set_progress(None, busy=True); status("Computing Q/HKL/intensity…"); pump(50)
#         try:
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)
#             state["builder"] = b
#             state["grid"] = state["edges"] = None
#             set_progress(50, busy=False); status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_progress(40, busy=False); status(f"Build failed: {e}")
#         finally:
#             set_busy(False)

#     def on_regrid() -> None:
#         b = state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first."); return

#         try:
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         except Exception as e:
#             show_error(str(e)); return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_busy(True); set_progress(None, busy=True)
#         status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…"); pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state.get("loader") is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             state["grid"], state["edges"] = grid, edges
#             set_progress(75, busy=False); status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_progress(60, busy=False); status(f"Regrid failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first."); return
#         try:
#             lo = float(contrast_lo_w.value); hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True); status("Opening RSM viewer…"); pump(50)
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
#             viz.launch()
#             set_progress(100, busy=False); status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_progress(80, busy=False); status(f"View failed: {e}")

#     def on_export_vtk() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first, then export."); return
#         out_path = as_path_str(export_vtr_w.value).strip()
#         if not out_path:
#             show_error("Choose an output .vtr file path."); return
#         if not out_path.lower().endswith(".vtr"):
#             out_path += ".vtr"
#         try:
#             set_busy(True); set_progress(None, busy=True)
#             status(f"Exporting VTK (.vtr) → {out_path}"); pump(50)
#             write_rsm_volume_to_vtr(state["grid"], state["edges"], out_path, binary=False, compress=True)
#             set_progress(100, busy=False); status(f"Exported: {out_path}")
#         except Exception as e:
#             show_error(f"Export error: {e}")
#             set_progress(0, busy=False); status(f"Export failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view_intensity_clicked(): on_view_intensity()
#     def on_load_clicked(): on_load()
#     def on_build_clicked(): on_build()
#     def on_regrid_clicked(): on_regrid()
#     def on_view_clicked(): on_view()
#     def on_export_clicked(): on_export_vtk()

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_progress(0, busy=False); status("Running pipeline (Load → Build → Regrid → View)…")
#             on_load()
#             if state.get("loader") is None: return
#             on_build()
#             if state.get("builder") is None: return
#             on_regrid()
#             if state.get("grid") is None or state.get("edges") is None: return
#             on_view()
#             status("Run All completed.")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect buttons
#     btn_intensity.clicked.connect(on_view_intensity_clicked)
#     btn_load.clicked.connect(on_load_clicked)
#     btn_build.clicked.connect(on_build_clicked)
#     btn_regrid.clicked.connect(on_regrid_clicked)
#     btn_view.clicked.connect(on_view_clicked)
#     btn_export.clicked.connect(on_export_clicked)
#     btn_run_all.clicked.connect(on_run_all)

#     # Keep saving YAML on changes
#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = (
#             None if (section == "ExperimentSetup" and key == "wavelength"
#                      and str(widget.value).strip().lower() in {"", "none", "null"})
#             else (float(widget.value) if isinstance(widget, FloatSpinBox)
#                   else int(widget.value) if isinstance(widget, SpinBox)
#                   else bool(widget.value) if isinstance(widget, CheckBox)
#                   else str(widget.value))
#         )
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # Run
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (simple, synchronous pipeline)

# Changes requested:
# • "TIFF folder" -> "DATA file" (single file)
# • File pick buttons use a simple symbol (📂) instead of text
# • "View TIFFs in napari" -> "View Intensity" (📈)
# • Use a different symbol for "View RSM" (🔭)
# • "Export for Paraview (.vtr)" -> "Export to VTK" and place this button
#   both above and below the "View RSM" button

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error
# from magicgui.widgets import (
#     CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
#     Label, LineEdit, PushButton, SpinBox, TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader, write_rsm_volume_to_vtr
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Theme / QSS
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """

# RUN_ALL_QSS = """
# QPushButton#RunAllPrimary {
#     background: #ff9800;
#     color: #ffffff;
#     border: 2px solid #e68900;
#     border-radius: 10px;
#     padding: 14px 20px;
#     font-size: 18px;
#     font-weight: 800;
# }
# QPushButton#RunAllPrimary:hover { background: #ffa726; }
# QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utils
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "data_file": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#         "export": {"vtr_path": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_intensity_in_napari(path: str):
#     """Open a single data file or (fallback) load all tiffs from a directory."""
#     viewer = napari.Viewer()
#     p = path.strip()
#     if os.path.isfile(p):
#         try:
#             viewer.open(p)
#             return viewer
#         except Exception:
#             pass
#     # Fallback to directory scan if a folder is provided
#     d = p if os.path.isdir(p) else os.path.dirname(p)
#     patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
#     opened = False
#     for pat in patterns:
#         try:
#             viewer.open(os.path.join(d, pat))
#             opened = True
#         except Exception:
#             pass
#     if not opened:
#         show_error("Could not open intensity. Provide a readable file or a folder with TIFF(s).")
#     return viewer

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc

# def set_file_button_symbol(fe: FileEdit, symbol: str = "📂") -> None:
#     """Replace 'Select file' text with a simple symbol."""
#     try:
#         for btn in fe.native.findChildren(QtWidgets.QPushButton):
#             btn.setText(symbol)
#             btn.setMinimumWidth(32)
#             btn.setMaximumWidth(36)
#             btn.setCursor(QtCore.Qt.PointingHandCursor)
#     except Exception:
#         pass


# # ─────────────────────────────────────────────────────────────────────────────
# # App (synchronous step-by-step pipeline)
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)

#     # ── Build UI ────────────────────────────────────────────────────────────
#     # Column 1: Data & Setup
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     data_file_w = FileEdit(mode="r", label="DATA file")  # was "TIFF folder"
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     # turn file buttons into simple symbols
#     set_file_button_symbol(spec_file_w, "📂")
#     set_file_button_symbol(data_file_w, "📂")

#     title_params = Label(value="<b>Experiment Setup</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = FloatSpinBox(label="wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     # Column 1 bottom buttons
#     btn_load = PushButton(text="📂 Load Data")
#     btn_intensity = PushButton(text="📈 View Intensity")  # was "View TIFFs in napari"
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_intensity.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, data_file_w, scans_w, only_hkl_w,
#             hsep(), title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # Column 2: Build & Regrid
#     title_build = Label(value="<b>RSM Builder</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")

#     title_regrid = Label(value="<b>Grid Settings</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     # Column 2 bottom buttons
#     btn_build   = PushButton(text="🔧 Build RSM Map")
#     btn_regrid  = PushButton(text="🧮 Regrid")
#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # Column 3: View + Status/Progress + Export + Bottom buttons
#     title_view   = Label(value="<b>Napari Viewer</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#     except Exception:
#         pass

#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

#     # Export widgets (top + bottom buttons around View RSM)
#     export_label = Label(value="<b>Export</b>")
#     export_vtr_w = FileEdit(mode="w", label="Output VTK (.vtr)")  # label updated
#     set_file_button_symbol(export_vtr_w, "📂")
#     btn_export_top    = PushButton(text="💾 Export to VTK")  # top button
#     btn_export_bottom = PushButton(text="💾 Export to VTK")  # bottom button

#     # Bottom action buttons for Column 3
#     btn_view     = PushButton(text="🔭 View RSM")   # different symbol from View Intensity
#     btn_run_all  = PushButton(text="▶️ Run All")
#     btn_run_all.native.setObjectName("RunAllPrimary")
#     btn_run_all.native.setMinimumHeight(64)

#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(12)
#     row3.addWidget(btn_view.native)
#     row3.addStretch(1)
#     row3.addWidget(btn_run_all.native)

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # Wrap columns in groups + scroll areas
#     g1 = make_group("Data", col1.native)
#     g1_lay = g1.layout()
#     g1_lay.addStretch(1)           # push buttons to bottom of column 1
#     g1_lay.addWidget(btn_row1)

#     g2 = make_group("Build", col2.native)
#     g2_lay = g2.layout()
#     g2_lay.addStretch(1)           # push buttons to bottom of column 2
#     g2_lay.addWidget(btn_row2)

#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
#     g3_lay.addWidget(export_vtr_w.native)
#     g3_lay.addWidget(btn_export_top.native)   # export ABOVE View RSM
#     g3_lay.addStretch(1)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)                # View RSM + Run All
#     g3_lay.addWidget(btn_export_bottom.native)  # export BELOW View RSM

#     def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#         wrapper = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(wrapper)
#         v.setContentsMargins(8, 8, 8, 8)
#         v.setSpacing(8)
#         v.addWidget(inner)
#         sc = QtWidgets.QScrollArea()
#         sc.setWidgetResizable(True)
#         sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#         sc.setWidget(wrapper)
#         return sc

#     s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
#     splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save)
#     ydoc = load_yaml(ypath)
#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "data_file": data_file_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#         "export": {"vtr_path": export_vtr_w},
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = str(widget.value).strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers (synchronous pipeline) ─────────────────────
#     def pump(ms: int = 0):
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(msg: str) -> None:
#         try:
#             status_w.native.append(msg)
#         except Exception:
#             status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
#         status_bar.showMessage(msg, 3000)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     def set_busy(b: bool):
#         for btn in (btn_load, btn_intensity, btn_build, btn_regrid, btn_view, btn_run_all, btn_export_top, btn_export_bottom):
#             try:
#                 btn.native.setEnabled(not b)
#             except Exception:
#                 pass

#     # ── App state
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     # ── Actions (synchronous, UI updates between steps) ─────────────────────
#     def on_view_intensity() -> None:
#         p = as_path_str(data_file_w.value).strip()
#         if not p:
#             show_error("Please choose a DATA file (or a folder with TIFFs)."); return
#         status(f"Opening intensity from: {p}")
#         pump(50)
#         open_intensity_in_napari(p)
#         status("Opened intensity in napari.")

#     def on_load() -> None:
#         spec = as_path_str(spec_file_w.value).strip()
#         dpath = as_path_str(data_file_w.value).strip()   # file or folder
#         try:
#             scans = parse_scan_list((scans_w.value or "").strip())
#         except Exception as e:
#             show_error(str(e)); return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file."); return
#         if not (os.path.isfile(dpath) or os.path.isdir(dpath)):
#             show_error("Select a valid DATA file (or a folder)."); return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22')."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Loading scans {scans}…")
#         pump(50)

#         try:
#             # Pass the chosen path directly to RSMDataLoader; if your version
#             # only accepts a directory, point it to dirname(dpath).
#             tiff_arg = dpath
#             if os.path.isfile(dpath):
#                 # fallback: try directory if your loader insists on folders
#                 tiff_arg = os.path.dirname(dpath) or "."
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),   # single YAML with ExperimentSetup
#                 tiff_arg,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_progress(25, busy=False)
#             status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_progress(0, busy=False)
#             status(f"Load failed: {e}")
#         finally:
#             set_busy(False)

#     def on_build() -> None:
#         if state.get("loader") is None:
#             show_error("Load data first."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status("Computing Q/HKL/intensity…")
#         pump(50)

#         try:
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)
#             state["builder"] = b
#             state["grid"] = state["edges"] = None
#             set_progress(50, busy=False)
#             status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_progress(40, busy=False)
#             status(f"Build failed: {e}")
#         finally:
#             set_busy(False)

#     def on_regrid() -> None:
#         b = state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first."); return

#         try:
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         except Exception as e:
#             show_error(str(e)); return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#         pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state.get("loader") is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             state["grid"], state["edges"] = grid, edges
#             set_progress(75, busy=False)
#             status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_progress(60, busy=False)
#             status(f"Regrid failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first."); return

#         try:
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             status("Opening RSM viewer…")
#             pump(50)

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
#             viz.launch()
#             set_progress(100, busy=False)
#             status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_progress(80, busy=False)
#             status(f"View failed: {e}")

#     # Export to VTK (.vtr) — wired to both top & bottom buttons
#     def _do_export() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first, then export."); return
#         out_path = as_path_str(export_vtr_w.value).strip()
#         if not out_path:
#             show_error("Choose an output .vtr file path."); return
#         if not out_path.lower().endswith(".vtr"):
#             out_path += ".vtr"

#         try:
#             set_busy(True)
#             set_progress(None, busy=True)
#             status(f"Exporting VTK (.vtr) → {out_path}")
#             pump(50)

#             rsm = state["grid"]
#             edges = state["edges"]  # (xax, yax, zax)
#             write_rsm_volume_to_vtr(rsm, edges, out_path, binary=False, compress=True)

#             set_progress(100, busy=False)
#             status(f"Exported: {out_path}")
#         except Exception as e:
#             show_error(f"Export error: {e}")
#             set_progress(0, busy=False)
#             status(f"Export failed: {e}")
#         finally:
#             set_busy(False)

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_progress(0, busy=False)
#             status("Running pipeline (Load → Build → Regrid → View)…")

#             on_load()
#             if state.get("loader") is None: return

#             on_build()
#             if state.get("builder") is None: return

#             on_regrid()
#             if state.get("grid") is None or state.get("edges") is None: return

#             on_view()
#             status("Run All completed.")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect buttons
#     btn_intensity.clicked.connect(on_view_intensity)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)
#     btn_export_top.clicked.connect(_do_export)
#     btn_export_bottom.clicked.connect(_do_export)

#     # Keep saving YAML on changes
#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = (
#             None if (section == "ExperimentSetup" and key == "wavelength"
#                      and str(widget.value).strip().lower() in {"", "none", "null"})
#             else (float(widget.value) if isinstance(widget, FloatSpinBox)
#                   else int(widget.value) if isinstance(widget, SpinBox)
#                   else bool(widget.value) if isinstance(widget, CheckBox)
#                   else str(widget.value))
#         )
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # Run
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (simple, synchronous pipeline)

# Layout
# ------
# • Column 1: Data paths → Experiment/Detector params → Buttons at bottom (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid → Buttons at bottom (Build, Regrid)
# • Column 3: View + Status + Progress + Export, with bottom buttons (View RSM / Run All)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error
# from magicgui.widgets import (
#     CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
#     Label, LineEdit, PushButton, SpinBox, TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader, write_rsm_volume_to_vtr
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Theme / QSS
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """

# RUN_ALL_QSS = """
# QPushButton#RunAllPrimary {
#     background: #ff9800;
#     color: #ffffff;
#     border: 2px solid #e68900;
#     border-radius: 10px;
#     padding: 14px 20px;        /* bigger padding */
#     font-size: 18px;           /* larger font */
#     font-weight: 800;
# }
# QPushButton#RunAllPrimary:hover { background: #ffa726; }
# QPushButton#RunAllPrimary:pressed { background: #fb8c00; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utils
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#         "export": {"vtr_path": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# # ─────────────────────────────────────────────────────────────────────────────
# # App (synchronous step-by-step pipeline)
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS + RUN_ALL_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)

#     # ── Build UI ────────────────────────────────────────────────────────────
#     # Column 1: Data & Setup
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment Setup</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = FloatSpinBox(label="wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     # Column 1 bottom buttons
#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(), title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # Column 2: Build & Regrid
#     title_build = Label(value="<b>RSM Builder</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")

#     title_regrid = Label(value="<b>Grid Settings</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     # Column 2 bottom buttons
#     btn_build   = PushButton(text="🔧 Build RSM Map")
#     btn_regrid  = PushButton(text="🧮 Regrid")
#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # Column 3: View + Status/Progress + Export + Bottom buttons
#     title_view   = Label(value="<b>Napari Viewer</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#     except Exception:
#         pass

#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

#     # Export widgets
#     export_label = Label(value="<b>Export</b>")
#     export_vtr_w = FileEdit(mode="w", label="Output .vtr file")
#     btn_export   = PushButton(text="💾 Export for ParaView (.vtr)")

#     # Bottom action buttons for Column 3
#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")
#     # Make Run All big + highlighted
#     btn_run_all.native.setObjectName("RunAllPrimary")
#     btn_run_all.native.setMinimumHeight(64)  # roughly ~2× default button height

#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(12)
#     row3.addWidget(btn_view.native)
#     row3.addStretch(1)
#     row3.addWidget(btn_run_all.native)  # big button aligned right

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # Wrap columns in groups + scroll areas
#     g1 = make_group("Data", col1.native)
#     g1_lay = g1.layout()
#     g1_lay.addStretch(1)           # push buttons to bottom of column 1
#     g1_lay.addWidget(btn_row1)

#     g2 = make_group("Build", col2.native)
#     g2_lay = g2.layout()
#     g2_lay.addStretch(1)           # push buttons to bottom of column 2
#     g2_lay.addWidget(btn_row2)

#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
#     g3_lay.addWidget(export_vtr_w.native)
#     g3_lay.addWidget(btn_export.native)
#     g3_lay.addStretch(1)           # keep bottom buttons glued to the bottom
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)

#     s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
#     splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save)
#     ydoc = load_yaml(ypath)
#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "tiff_dir": tiff_dir_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#         "export": {"vtr_path": export_vtr_w},
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = str(widget.value).strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers (synchronous pipeline) ─────────────────────
#     def pump(ms: int = 0):
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(msg: str) -> None:
#         try:
#             status_w.native.append(msg)
#         except Exception:
#             status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
#         status_bar.showMessage(msg, 3000)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     def set_busy(b: bool):
#         for btn in (btn_load, btn_tiff, btn_build, btn_regrid, btn_view, btn_run_all, btn_export):
#             try:
#                 btn.native.setEnabled(not b)
#             except Exception:
#                 pass

#     # ── App state
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     # ── Actions (synchronous, UI updates between steps) ─────────────────────
#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         status(f"Opening TIFFs in napari: {d}")
#         pump(50)
#         open_tiffs_in_napari(d)
#         status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         spec = as_path_str(spec_file_w.value).strip()
#         tdir = as_path_str(tiff_dir_w.value).strip()
#         try:
#             scans = parse_scan_list((scans_w.value or "").strip())
#         except Exception as e:
#             show_error(str(e)); return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file."); return
#         if not tdir or not os.path.isdir(tdir):
#             show_error("Select a valid TIFF folder."); return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22')."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Loading scans {scans}…")
#         pump(50)

#         try:
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),   # single YAML with ExperimentSetup
#                 tdir,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_progress(25, busy=False)
#             status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_progress(0, busy=False)
#             status(f"Load failed: {e}")
#         finally:
#             set_busy(False)

#     def on_build() -> None:
#         if state.get("loader") is None:
#             show_error("Load data first."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status("Computing Q/HKL/intensity…")
#         pump(50)

#         try:
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)
#             state["builder"] = b
#             state["grid"] = state["edges"] = None
#             set_progress(50, busy=False)
#             status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_progress(40, busy=False)
#             status(f"Build failed: {e}")
#         finally:
#             set_busy(False)

#     def on_regrid() -> None:
#         b = state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first."); return

#         try:
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         except Exception as e:
#             show_error(str(e)); return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#         pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state.get("loader") is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             state["grid"], state["edges"] = grid, edges
#             set_progress(75, busy=False)
#             status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_progress(60, busy=False)
#             status(f"Regrid failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first."); return

#         try:
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             status("Opening RSM viewer…")
#             pump(50)

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
#             viz.launch()
#             set_progress(100, busy=False)
#             status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_progress(80, busy=False)
#             status(f"View failed: {e}")

#     # Export to ParaView .vtr
#     def on_export() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first, then export."); return
#         out_path = as_path_str(export_vtr_w.value).strip()
#         if not out_path:
#             show_error("Choose an output .vtr file path."); return
#         if not out_path.lower().endswith(".vtr"):
#             out_path += ".vtr"

#         try:
#             set_busy(True)
#             set_progress(None, busy=True)
#             status(f"Exporting VTR → {out_path}")
#             pump(50)

#             rsm = state["grid"]
#             edges = state["edges"]  # (xax, yax, zax)
#             write_rsm_volume_to_vtr(rsm, edges, out_path, binary=False, compress=True)

#             set_progress(100, busy=False)
#             status(f"Exported: {out_path}")
#         except Exception as e:
#             show_error(f"Export error: {e}")
#             set_progress(0, busy=False)
#             status(f"Export failed: {e}")
#         finally:
#             set_busy(False)

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_progress(0, busy=False)
#             status("Running pipeline (Load → Build → Regrid → View)…")

#             on_load()
#             if state.get("loader") is None: return

#             on_build()
#             if state.get("builder") is None: return

#             on_regrid()
#             if state.get("grid") is None or state.get("edges") is None: return

#             on_view()
#             status("Run All completed.")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect buttons
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)
#     btn_export.clicked.connect(on_export)

#     # Keep saving YAML on changes
#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = (
#             None if (section == "ExperimentSetup" and key == "wavelength"
#                      and str(widget.value).strip().lower() in {"", "none", "null"})
#             else (float(widget.value) if isinstance(widget, FloatSpinBox)
#                   else int(widget.value) if isinstance(widget, SpinBox)
#                   else bool(widget.value) if isinstance(widget, CheckBox)
#                   else str(widget.value))
#         )
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # Run
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (simple, synchronous pipeline)

# Layout
# ------
# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid
# • Column 3: View + Status + Progress + (View RSM / Run All / Export at bottom)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error
# from magicgui.widgets import (
#     CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
#     Label, LineEdit, PushButton, SpinBox, TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader, write_rsm_volume_to_vtr
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Theme / QSS
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utils
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#         "export": {"vtr_path": None},   # <— remember your export path
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# # ─────────────────────────────────────────────────────────────────────────────
# # App (synchronous step-by-step pipeline)
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)

#     # ── Build UI ────────────────────────────────────────────────────────────
#     # Column 1: Data & Setup
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment Setup</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = FloatSpinBox(label="wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(), title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # Column 2: Build & Regrid
#     title_build = Label(value="<b>RSM Builder</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="🔧 Build RSM Map")

#     title_regrid = Label(value="<b>Grid Settings</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="🧮 Regrid")

#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # Column 3: View + Status/Progress + Bottom buttons + Export
#     title_view   = Label(value="<b>Napari Viewer</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#     except Exception:
#         pass

#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

#     # Export widgets
#     export_label = Label(value="<b>Export</b>")
#     export_vtr_w = FileEdit(mode="w", label="Output .vtr file")
#     btn_export   = PushButton(text="💾 Export for ParaView (.vtr)")

#     # Bottom action buttons
#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")
#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(8)
#     row3.addWidget(btn_view.native); row3.addWidget(btn_run_all.native); row3.addStretch(1)

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # Wrap columns in groups + scroll areas
#     g1 = make_group("Data", col1.native); g1.layout().addWidget(btn_row1)
#     g2 = make_group("Build", col2.native); g2.layout().addWidget(btn_row2)
#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     # Export section goes above bottom buttons
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Export</b>"))
#     g3_lay.addWidget(export_vtr_w.native)
#     g3_lay.addWidget(btn_export.native)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)

#     s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
#     splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save)
#     ydoc = load_yaml(ypath)
#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "tiff_dir": tiff_dir_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#         "export": {"vtr_path": export_vtr_w},   # <— persist output path
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = str(widget.value).strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers (synchronous pipeline) ─────────────────────
#     def pump(ms: int = 0):
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(msg: str) -> None:
#         try:
#             status_w.native.append(msg)
#         except Exception:
#             status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
#         status_bar.showMessage(msg, 3000)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     def set_busy(b: bool):
#         for btn in (btn_load, btn_build, btn_regrid, btn_view, btn_run_all, btn_export):
#             try:
#                 btn.native.setEnabled(not b)
#             except Exception:
#                 pass

#     # ── App state
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     # ── Actions (synchronous, UI updates between steps) ─────────────────────
#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         status(f"Opening TIFFs in napari: {d}")
#         pump(50)
#         open_tiffs_in_napari(d)
#         status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         spec = as_path_str(spec_file_w.value).strip()
#         tdir = as_path_str(tiff_dir_w.value).strip()
#         try:
#             scans = parse_scan_list((scans_w.value or "").strip())
#         except Exception as e:
#             show_error(str(e)); return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file."); return
#         if not tdir or not os.path.isdir(tdir):
#             show_error("Select a valid TIFF folder."); return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22')."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Loading scans {scans}…")
#         pump(50)

#         try:
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),   # single YAML with ExperimentSetup
#                 tdir,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_progress(25, busy=False)
#             status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_progress(0, busy=False)
#             status(f"Load failed: {e}")
#         finally:
#             set_busy(False)

#     def on_build() -> None:
#         if state.get("loader") is None:
#             show_error("Load data first."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status("Computing Q/HKL/intensity…")
#         pump(50)

#         try:
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)
#             state["builder"] = b
#             state["grid"] = state["edges"] = None
#             set_progress(50, busy=False)
#             status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_progress(40, busy=False)
#             status(f"Build failed: {e}")
#         finally:
#             set_busy(False)

#     def on_regrid() -> None:
#         b = state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first."); return

#         try:
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         except Exception as e:
#             show_error(str(e)); return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#         pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state.get("loader") is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             state["grid"], state["edges"] = grid, edges
#             set_progress(75, busy=False)
#             status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_progress(60, busy=False)
#             status(f"Regrid failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first."); return

#         try:
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             status("Opening RSM viewer…")
#             pump(50)

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
#             viz.launch()
#             set_progress(100, busy=False)
#             status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_progress(80, busy=False)
#             status(f"View failed: {e}")

#     # NEW: Export to ParaView .vtr
#     def on_export() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first, then export."); return
#         out_path = as_path_str(export_vtr_w.value).strip()
#         if not out_path:
#             show_error("Choose an output .vtr file path."); return
#         if not out_path.lower().endswith(".vtr"):
#             out_path += ".vtr"

#         try:
#             set_busy(True)
#             set_progress(None, busy=True)
#             status(f"Exporting VTR → {out_path}")
#             pump(50)

#             rsm = state["grid"]
#             edges = state["edges"]  # (xax, yax, zax)
#             write_rsm_volume_to_vtr(rsm, edges, out_path, binary=False, compress=True)

#             set_progress(100, busy=False)
#             status(f"Exported: {out_path}")
#         except Exception as e:
#             show_error(f"Export error: {e}")
#             set_progress(0, busy=False)
#             status(f"Export failed: {e}")
#         finally:
#             set_busy(False)

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_progress(0, busy=False)
#             status("Running pipeline (Load → Build → Regrid → View)…")

#             on_load()
#             if state.get("loader") is None: return

#             on_build()
#             if state.get("builder") is None: return

#             on_regrid()
#             if state.get("grid") is None or state.get("edges") is None: return

#             on_view()
#             status("Run All completed.")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect buttons
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)
#     btn_export.clicked.connect(on_export)

#     # Keep saving YAML on changes
#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = (
#             None if (section == "ExperimentSetup" and key == "wavelength"
#                      and str(widget.value).strip().lower() in {"", "none", "null"})
#             else (float(widget.value) if isinstance(widget, FloatSpinBox)
#                   else int(widget.value) if isinstance(widget, SpinBox)
#                   else bool(widget.value) if isinstance(widget, CheckBox)
#                   else str(widget.value))
#         )
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # Run
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (simple, synchronous pipeline)

# Layout
# ------
# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid
# • Column 3: View + Status + Progress + (View RSM / Run All at bottom)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error
# from magicgui.widgets import (
#     CheckBox, ComboBox, Container, FileEdit, FloatSpinBox,
#     Label, LineEdit, PushButton, SpinBox, TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Theme / QSS
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;      /* larger titles */
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utils
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)   # inner gutter to avoid overlap with splitter
#     v.setSpacing(8)
#     v.addWidget(inner)
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# # ─────────────────────────────────────────────────────────────────────────────
# # App (synchronous step-by-step pipeline)
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)

#     # ── Build UI ────────────────────────────────────────────────────────────
#     # Column 1: Data & Setup
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment Setup</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = FloatSpinBox(label="wavelength (Å)", min=1e-6, max=1e6, step=1e-3)
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(), title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # Column 2: Build & Regrid
#     title_build = Label(value="<b>RSM Builder</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="🔧 Build RSM Map")

#     title_regrid = Label(value="<b>Grid Settings</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="🧮 Regrid")

#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # Column 3: View + Status/Progress + Bottom buttons
#     title_view   = Label(value="<b>Napari Viewer</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#     except Exception:
#         pass

#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0); progress.setMaximum(100); progress.setValue(0); progress.setTextVisible(True)

#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")
#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(8)
#     row3.addWidget(btn_view.native); row3.addWidget(btn_run_all.native); row3.addStretch(1)

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # Wrap columns in groups + scroll areas
#     g1 = make_group("Data", col1.native); g1.layout().addWidget(btn_row1)
#     g2 = make_group("Build", col2.native); g2.layout().addWidget(btn_row2)
#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)

#     s1, s2, s3 = make_scroll(g1), make_scroll(g2), make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1); splitter.addWidget(s2); splitter.addWidget(s3)
#     splitter.setHandleWidth(10); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar(); win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save)
#     ydoc = load_yaml(ypath)
#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "tiff_dir": tiff_dir_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers (synchronous pipeline) ─────────────────────
#     def pump(ms: int = 0):
#         """Let the GUI breathe before a heavy call."""
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, ms)

#     def status(msg: str) -> None:
#         try:
#             status_w.native.append(msg)
#         except Exception:
#             status_w.value = (status_w.value or "") + (("\n" if status_w.value else "") + msg)
#         status_bar.showMessage(msg, 3000)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     def set_busy(b: bool):
#         for btn in (btn_load, btn_build, btn_regrid, btn_view, btn_run_all):
#             try:
#                 btn.native.setEnabled(not b)
#             except Exception:
#                 pass

#     # ── App state
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     # ── Actions (synchronous, UI updates between steps) ─────────────────────
#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         status(f"Opening TIFFs in napari: {d}")
#         pump(50)
#         open_tiffs_in_napari(d)
#         status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         spec = as_path_str(spec_file_w.value).strip()
#         tdir = as_path_str(tiff_dir_w.value).strip()
#         try:
#             scans = parse_scan_list((scans_w.value or "").strip())
#         except Exception as e:
#             show_error(str(e)); return

#         if not spec or not os.path.isfile(spec):
#             show_error("Select a valid SPEC file."); return
#         if not tdir or not os.path.isdir(tdir):
#             show_error("Select a valid TIFF folder."); return
#         if not scans:
#             show_error("Enter at least one scan (e.g. '17, 18-22')."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Loading scans {scans}…")
#         pump(50)

#         try:
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),   # single YAML with ExperimentSetup
#                 tdir,
#                 selected_scans=scans,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             )
#             loader.load()  # synchronous
#             state["loader"] = loader
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_progress(25, busy=False)
#             status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_progress(0, busy=False)
#             status(f"Load failed: {e}")
#         finally:
#             set_busy(False)

#     def on_build() -> None:
#         if state.get("loader") is None:
#             show_error("Load data first."); return

#         set_busy(True)
#         set_progress(None, busy=True)
#         status("Computing Q/HKL/intensity…")
#         pump(50)

#         try:
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=False)  # synchronous
#             state["builder"] = b
#             state["grid"] = state["edges"] = None
#             set_progress(50, busy=False)
#             status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_progress(40, busy=False)
#             status(f"Build failed: {e}")
#         finally:
#             set_busy(False)

#     def on_regrid() -> None:
#         b = state.get("builder")
#         if b is None:
#             show_error("Build the RSM map first."); return

#         try:
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         except Exception as e:
#             show_error(str(e)); return
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*)."); return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_busy(True)
#         set_progress(None, busy=True)
#         status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#         pump(50)

#         try:
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state.get("loader") is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,  # sync path — returns when done
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)  # synchronous
#             state["grid"], state["edges"] = grid, edges
#             set_progress(75, busy=False)
#             status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_progress(60, busy=False)
#             status(f"Regrid failed: {e}")
#         finally:
#             set_busy(False)

#     def on_view() -> None:
#         if state.get("grid") is None or state.get("edges") is None:
#             show_error("Regrid first."); return

#         try:
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             status("Opening RSM viewer…")
#             pump(50)

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
#             viz.launch()
#             set_progress(100, busy=False)
#             status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_progress(80, busy=False)
#             status(f"View failed: {e}")

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_progress(0, busy=False)
#             status("Running pipeline (Load → Build → Regrid → View)…")

#             on_load()
#             if state.get("loader") is None: return

#             on_build()
#             if state.get("builder") is None: return

#             on_regrid()
#             if state.get("grid") is None or state.get("edges") is None: return

#             on_view()
#             status("Run All completed.")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect buttons
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)

#     # Load YAML into UI (already done above), keep saving on changes
#     def on_changed(section: str, key: str, widget: Any) -> None:
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = (
#             None if (section == "ExperimentSetup" and key == "wavelength"
#                      and str(widget.value).strip().lower() in {"", "none", "null"})
#             else (float(widget.value) if isinstance(widget, FloatSpinBox)
#                   else int(widget.value) if isinstance(widget, SpinBox)
#                   else bool(widget.value) if isinstance(widget, CheckBox)
#                   else str(widget.value))
#         )
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # Run
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app — single-YAML persistence (beautified UI + live logging + threaded progress)

# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid  (Build/Regrid run in background threads)
# • Column 3: View + Status + Progress + (View RSM / Run All at bottom)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple, Callable

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets, QtGui
# from napari.utils.notifications import show_error, show_info
# from magicgui.widgets import (
#     CheckBox,
#     ComboBox,
#     Container,
#     FileEdit,
#     FloatSpinBox,
#     Label,
#     LineEdit,
#     PushButton,
#     SpinBox,
#     TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Global style (better group titles, crisp splitter, polished inputs)
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 12px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 6px 10px;
#     color: #2c3e50;
#     font-size: 18px;
#     font-weight: 700;
#     letter-spacing: 0.2px;
# }
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utilities
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)   # inner gutter
#     v.setSpacing(8)
#     v.addWidget(inner)

#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# # ─────────────────────────────────────────────────────────────────────────────
# # Stdout/stderr capture → status TextEdit (thread-safe via Qt signals)
# # ─────────────────────────────────────────────────────────────────────────────

# class QtLogEmitter(QtCore.QObject):
#     message = QtCore.Signal(str, bool)  # text, is_stderr

# class QtLogWriter:
#     """File-like writer that forwards .write() to a Qt signal (works from threads)."""
#     def __init__(self, emitter: QtLogEmitter, is_stderr: bool = False):
#         self._emitter = emitter
#         self._is_err = is_stderr
#         self._buffer = ""

#     def write(self, s: str):
#         if not s:
#             return
#         self._buffer += s
#         while "\n" in self._buffer:
#             line, self._buffer = self._buffer.split("\n", 1)
#             self._emitter.message.emit(line + "\n", self._is_err)

#     def flush(self):
#         if self._buffer:
#             self._emitter.message.emit(self._buffer, self._is_err)
#             self._buffer = ""


# # ─────────────────────────────────────────────────────────────────────────────
# # Tiny threading helper for long tasks (keeps UI/Progress live)
# # ─────────────────────────────────────────────────────────────────────────────

# class Worker(QtCore.QObject):
#     finished = QtCore.Signal(object)            # result
#     error = QtCore.Signal(object)               # exception

#     def __init__(self, fn: Callable[[], Any]):
#         super().__init__()
#         self._fn = fn

#     @QtCore.Slot()
#     def run(self):
#         try:
#             res = self._fn()
#             self.finished.emit(res)
#         except Exception as e:
#             self.error.emit(e)


# # Registry of live threads to avoid premature destruction
# THREADS: set[QtCore.QThread] = set()

# def run_in_thread(
#     fn: Callable[[], Any],
#     on_done: Callable[[Any], None],
#     on_err: Callable[[Exception], None],
# ) -> QtCore.QThread:
#     """Run fn() in a QThread, call on_done/on_err in the GUI thread, keep strong ref."""
#     th = QtCore.QThread()
#     worker = Worker(fn)
#     worker.moveToThread(th)
#     th.started.connect(worker.run)
#     worker.finished.connect(on_done)
#     worker.error.connect(on_err)
#     # cleanup
#     worker.finished.connect(th.quit)
#     worker.error.connect(th.quit)
#     th.finished.connect(worker.deleteLater)
#     th.finished.connect(lambda: THREADS.discard(th))
#     th.finished.connect(th.deleteLater)
#     THREADS.add(th)
#     th.start()
#     return th


# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     setting_up = False  # avoid YAML writes during initial population

#     # ── Column 1: Data & Setup ──────────────────────────────────────────────
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     # Button bar (Column 1)
#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(),
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # ── Column 2: Build & Regrid ────────────────────────────────────────────
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="🔧 Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="🧮 Regrid")

#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # ── Column 3: View + Status + Progress (buttons at bottom) ──────────────
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(220)
#         # Use system fixed font to avoid missing-family warnings (Consolas etc.)
#         fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
#         fixed.setPointSize(fixed.pointSize() + 1)
#         status_w.native.setFont(fixed)
#     except Exception:
#         pass
#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0)
#     progress.setMaximum(100)
#     progress.setValue(0)
#     progress.setTextVisible(True)

#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")
#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(8)
#     row3.addWidget(btn_view.native); row3.addWidget(btn_run_all.native); row3.addStretch(1)

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # ── Wrap columns in group boxes + scroll areas ───────────────────────────
#     g1 = make_group("Data", col1.native)
#     g1.layout().addWidget(btn_row1)

#     g2 = make_group("Processing", col2.native)
#     g2.layout().addWidget(btn_row2)

#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)

#     s1 = make_scroll(g1)
#     s2 = make_scroll(g2)
#     s3 = make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1)
#     splitter.addWidget(s2)
#     splitter.addWidget(s3)
#     splitter.setHandleWidth(10)
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     # Main window + status bar
#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 820)
#     status_bar = QtWidgets.QStatusBar()
#     win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save) ─────────────────────────────────
#     ydoc = load_yaml(ypath)

#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {"spec_file": spec_file_w, "tiff_dir": tiff_dir_w, "scans": scans_w, "only_hkl": only_hkl_w},
#         "ExperimentSetup": {
#             "distance": distance_w, "pitch": pitch_w, "ycenter": ycenter_w, "xcenter": xcenter_w,
#             "xpixels": xpixels_w, "ypixels": ypixels_w, "energy": energy_w, "wavelength": wavelength_w,
#         },
#         "build": {"ub_includes_2pi": ub_2pi_w, "center_is_one_based": center_one_based_w},
#         "crop": {"enable": crop_enable_w, "y_min": y_min_w, "y_max": y_max_w, "x_min": x_min_w, "x_max": x_max_w},
#         "regrid": {"space": space_w, "grid_shape": grid_shape_w, "fuzzy": fuzzy_w, "fuzzy_width": fuzzy_width_w, "normalize": normalize_w},
#         "view": {"log_view": log_view_w, "cmap": cmap_w, "rendering": rendering_w, "contrast_lo": contrast_lo_w, "contrast_hi": contrast_hi_w},
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     setting_up = True
#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers ────────────────────────────────────────────
#     def scroll_to_end(te: QtWidgets.QTextEdit):
#         cursor = te.textCursor()
#         cursor.movePosition(QtGui.QTextCursor.End)
#         te.setTextCursor(cursor)
#         te.ensureCursorVisible()

#     def set_status(msg: str) -> None:
#         status_w.native.append(msg)
#         scroll_to_end(status_w.native)
#         status_bar.showMessage(msg, 3000)
#         try: show_info(msg)
#         except Exception: pass
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)

#     def set_progress(value: int | None, *, busy: bool = False):
#         if busy:
#             progress.setRange(0, 0)  # indeterminate
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))
#         QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)

#     # Hook Python stdout/stderr → status panel (thread-safe)
#     log_emitter = QtLogEmitter()

#     def _append_log(text: str, is_err: bool):
#         if not text:
#             return
#         te = status_w.native
#         if is_err:
#             te.setTextColor(QtGui.QColor("#b22222"))
#             te.append(text.rstrip("\n"))
#             te.setTextColor(QtGui.QColor("#000000"))
#         else:
#             te.append(text.rstrip("\n"))
#         scroll_to_end(te)

#     log_emitter.message.connect(_append_log)
#     stdout_writer = QtLogWriter(log_emitter, is_stderr=False)
#     stderr_writer = QtLogWriter(log_emitter, is_stderr=True)
#     sys_stdout_orig, sys_stderr_orig = sys.stdout, sys.stderr
#     sys.stdout, sys.stderr = stdout_writer, stderr_writer

#     # Ensure threads quit cleanly on app exit
#     def _shutdown_threads():
#         # Stop starting new work
#         btn_run_all.native.setEnabled(False)
#         btn_view.native.setEnabled(False)
#         # Ask all threads to finish and wait briefly
#         for th in list(THREADS):
#             try:
#                 th.quit()
#             except Exception:
#                 pass
#         for th in list(THREADS):
#             try:
#                 th.wait(5000)
#             except Exception:
#                 pass
#         THREADS.clear()
#         # restore stdio
#         sys.stdout, sys.stderr = sys_stdout_orig, sys_stderr_orig

#     app.aboutToQuit.connect(_shutdown_threads)

#     # ── Actions (Build/Regrid run in background threads) ─────────────────────
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         set_progress(None, busy=True)
#         set_status(f"Opening TIFFs in napari: {d}")
#         open_tiffs_in_napari(d)
#         set_progress(0, busy=False)
#         set_status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         try:
#             spec = as_path_str(spec_file_w.value).strip()
#             tdir = as_path_str(tiff_dir_w.value).strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = parse_scan_list((scans_w.value or "").strip())
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_progress(None, busy=True)
#             print(f"[LOAD] Scans: {scan_list}")
#             set_status(f"Loading scans {scan_list}…")

#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),
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
#         finally:
#             set_progress(25)

#     # Build (threaded)
#     def on_build() -> None:
#         if state["loader"] is None:
#             show_error("Load data first.")
#             return
#         set_status("Computing Q/HKL/intensity… (verbose on)")
#         set_progress(None, busy=True)

#         def task():
#             print("[BUILD] Starting compute_full(verbose=True)")
#             b = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             b.compute_full(verbose=True)
#             print("[BUILD] compute_full() done")
#             return b

#         def done(res):
#             state["builder"] = res
#             state["grid"] = state["edges"] = None
#             set_status("RSM map built.")
#             set_progress(50, busy=False)

#         def err(e: Exception):
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")
#             set_progress(40, busy=False)

#         th = run_in_thread(task, done, err)
#         # keep strong ref
#         THREADS.add(th)

#     # Regrid (threaded)
#     def on_regrid() -> None:
#         if state["builder"] is None:
#             show_error("Build the RSM map first.")
#             return

#         gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#         if gx is None:
#             show_error("Grid X (first value) is required (e.g., 200,*,*).")
#             return

#         do_crop = bool(crop_enable_w.value)
#         ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#         xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#         set_progress(None, busy=True)
#         msg = f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…"
#         print(f"[REGRID] {msg}")
#         set_status(msg)

#         def task():
#             b = state["builder"]
#             if do_crop:
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 print(f"[REGRID] Cropping to y=({ymin},{ymax}), x=({xmin},{xmax})")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             grid, edges = b.regrid_xu(**kwargs)
#             print("[REGRID] Done")
#             return grid, edges

#         def done(res):
#             grid, edges = res
#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#             set_progress(75, busy=False)

#         def err(e: Exception):
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")
#             set_progress(60, busy=False)

#         th = run_in_thread(task, done, err)
#         THREADS.add(th)

#     def on_view() -> None:
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")
#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             print(f"[VIEW] Opening viewer: log={bool(log_view_w.value)}, cmap={cmap_w.value}, render={rendering_w.value}, contrast=({lo},{hi})")
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
#             viz.launch()
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")
#         finally:
#             set_progress(100, busy=False)

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         set_status("Running pipeline (Load → Build → Regrid → View)…")
#         set_progress(0)
#         on_load()
#         if state["loader"] is None:
#             set_status("Load failed; aborting Run All.")
#             btn_run_all.native.setEnabled(True)
#             return

#         # Build → Regrid → View as chained background tasks
#         def after_view():
#             btn_run_all.native.setEnabled(True)
#             set_status("Run All completed.")

#         def after_regrid():
#             on_view(); after_view()

#         def after_build():
#             on_regrid_threaded_then(after_regrid)

#         # threaded build helper
#         def on_build_threaded_then(cb: Callable[[], None]):
#             def task():
#                 print("[BUILD] Starting compute_full(verbose=True)")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=True)
#                 print("[BUILD] compute_full() done")
#                 state["builder"] = b
#                 return True
#             def done(_):
#                 set_progress(50, busy=False); cb()
#             def err(e: Exception):
#                 show_error(f"Build error: {e}")
#                 set_status(f"Build failed: {e}")
#                 btn_run_all.native.setEnabled(True)
#                 set_progress(40, busy=False)
#             th = run_in_thread(task, done, err)
#             THREADS.add(th)

#         def on_regrid_threaded_then(cb: Callable[[], None]):
#             if state["builder"] is None:
#                 show_error("Build the RSM map first.")
#                 btn_run_all.native.setEnabled(True)
#                 return
#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 show_error("Grid X (first value) is required (e.g., 200,*,*).")
#                 btn_run_all.native.setEnabled(True)
#                 return

#             do_crop = bool(crop_enable_w.value)
#             ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#             xmin, xmax = int(x_min_w.value), int(x_max_w.value)

#             set_progress(None, busy=True)
#             set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")

#             def task():
#                 b = state["builder"]
#                 if do_crop:
#                     if ymin >= ymax or xmin >= xmax:
#                         raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                     if state["loader"] is None:
#                         raise RuntimeError("Internal error: loader missing; run Build again.")
#                     b = RSMBuilder(
#                         state["loader"],
#                         ub_includes_2pi=bool(ub_2pi_w.value),
#                         center_is_one_based=bool(center_one_based_w.value),
#                     )
#                     b.compute_full(verbose=False)
#                     b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))
#                 kwargs = dict(
#                     space=space_w.value,
#                     grid_shape=(gx, gy, gz),
#                     fuzzy=bool(fuzzy_w.value),
#                     normalize=normalize_w.value,
#                     stream=True,
#                 )
#                 if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                     kwargs["width"] = float(fuzzy_width_w.value)
#                 grid, edges = b.regrid_xu(**kwargs)
#                 state["grid"], state["edges"] = grid, edges
#                 return True
#             def done(_):
#                 set_status("Regrid completed.")
#                 set_progress(75, busy=False)
#                 cb()
#             def err(e: Exception):
#                 show_error(f"Regrid error: {e}")
#                 set_status(f"Regrid failed: {e}")
#                 btn_run_all.native.setEnabled(True)
#                 set_progress(60, busy=False)
#             th = run_in_thread(task, done, err)
#             THREADS.add(th)

#         on_build_threaded_then(after_build)

#     # Connect actions
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)

#     # Run (Qt5/Qt6 compatible)
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app: single-YAML persistence for all parameters (beautified UI + logging + progress)

# Layout
# ------
# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid
# • Column 3: View + Status + Progress + (View RSM / Run All at bottom)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error, show_info
# from magicgui.widgets import (
#     CheckBox,
#     ComboBox,
#     Container,
#     FileEdit,
#     FloatSpinBox,
#     Label,
#     LineEdit,
#     PushButton,
#     SpinBox,
#     TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder


# # ─────────────────────────────────────────────────────────────────────────────
# # Global style (fix overlapping dividers + polished look)
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 10px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 4px 8px;
#     color: #2c3e50;
# }

# /* Crisp vertical divider with its own gutters */
# QSplitter::handle {
#     background: #e9edf3;
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }

# /* Inputs / buttons */
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utilities
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     """Magicgui separator for use *inside* Container lists."""
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def q_hsep(height: int = 10) -> QtWidgets.QWidget:
#     """Pure-Qt separator for addWidget() in Qt layouts."""
#     line = QtWidgets.QFrame()
#     line.setFrameShape(QtWidgets.QFrame.HLine)
#     line.setFrameShadow(QtWidgets.QFrame.Sunken)
#     line.setLineWidth(1)
#     line.setFixedHeight(height)
#     return line

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(12, 12, 12, 12)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     """Scroll area with inner gutters so borders don't touch the splitter."""
#     wrapper = QtWidgets.QWidget()
#     v = QtWidgets.QVBoxLayout(wrapper)
#     v.setContentsMargins(8, 8, 8, 8)   # inner gutter
#     v.setSpacing(8)
#     v.addWidget(inner)

#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(wrapper)
#     return sc


# # ─────────────────────────────────────────────────────────────────────────────
# # Qt stdout/stderr capture → status TextEdit
# # ─────────────────────────────────────────────────────────────────────────────

# class QtLogEmitter(QtCore.QObject):
#     message = QtCore.Signal(str, bool)  # text, is_stderr

# class QtLogWriter:
#     """File-like writer that forwards .write() to a Qt signal."""
#     def __init__(self, emitter: QtLogEmitter, is_stderr: bool = False):
#         self._emitter = emitter
#         self._is_err = is_stderr
#         self._buffer = ""

#     def write(self, s: str):
#         if not s:
#             return
#         # Buffer until newline to avoid too many signals on partial writes
#         self._buffer += s
#         while "\n" in self._buffer:
#             line, self._buffer = self._buffer.split("\n", 1)
#             self._emitter.message.emit(line + "\n", self._is_err)

#     def flush(self):
#         if self._buffer:
#             self._emitter.message.emit(self._buffer, self._is_err)
#             self._buffer = ""


# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     setting_up = False  # avoid YAML writes during initial population

#     # ── Column 1: Data & Setup ──────────────────────────────────────────────
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is meters → converted to Å."

#     # Button bar (Column 1)
#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(),
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # ── Column 2: Build & Regrid ────────────────────────────────────────────
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="🔧 Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="🧮 Regrid")

#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # ── Column 3: View + Status + Progress (buttons moved to bottom) ────────
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     # Status + progress
#     status_label_w = Label(value="Status / Output")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(180)
#         status_w.native.setStyleSheet("font-family: Consolas, Menlo, monospace;")
#     except Exception:
#         pass
#     progress = QtWidgets.QProgressBar()
#     progress.setMinimum(0)
#     progress.setMaximum(100)
#     progress.setValue(0)
#     progress.setTextVisible(True)

#     # Buttons at the *bottom*
#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")
#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(8)
#     row3.addWidget(btn_view.native); row3.addWidget(btn_run_all.native); row3.addStretch(1)

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # ── Wrap columns in group boxes + scroll areas ───────────────────────────
#     g1 = make_group("Data", col1.native)
#     g1.layout().addWidget(btn_row1)

#     g2 = make_group("Processing", col2.native)
#     g2.layout().addWidget(btn_row2)

#     # Column 3 group layout (Qt additions: status, progress, bottom buttons)
#     g3 = make_group("View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status / Output</b>"))
#     g3_lay.addWidget(status_w.native)
#     g3_lay.addWidget(progress)
#     g3_lay.addWidget(q_hsep())
#     g3_lay.addWidget(btn_row3)

#     s1 = make_scroll(g1)
#     s2 = make_scroll(g2)
#     s3 = make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1)
#     splitter.addWidget(s2)
#     splitter.addWidget(s3)
#     splitter.setHandleWidth(10)  # wider so divider doesn't visually collide
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     splitter.setSizes([440, 440, 440])

#     # Main window + status bar
#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 800)
#     status_bar = QtWidgets.QStatusBar()
#     win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save) ─────────────────────────────────
#     ydoc = load_yaml(ypath)  # reload post-UI

#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {
#             "spec_file": spec_file_w,
#             "tiff_dir":  tiff_dir_w,
#             "scans":     scans_w,
#             "only_hkl":  only_hkl_w,
#         },
#         "ExperimentSetup": {
#             "distance":   distance_w,
#             "pitch":      pitch_w,
#             "ycenter":    ycenter_w,
#             "xcenter":    xcenter_w,
#             "xpixels":    xpixels_w,
#             "ypixels":    ypixels_w,
#             "energy":     energy_w,
#             "wavelength": wavelength_w,
#         },
#         "build": {
#             "ub_includes_2pi":    ub_2pi_w,
#             "center_is_one_based": center_one_based_w,
#         },
#         "crop": {
#             "enable": crop_enable_w,
#             "y_min":  y_min_w, "y_max": y_max_w,
#             "x_min":  x_min_w, "x_max": x_max_w,
#         },
#         "regrid": {
#             "space":      space_w,
#             "grid_shape": grid_shape_w,
#             "fuzzy":      fuzzy_w,
#             "fuzzy_width":fuzzy_width_w,
#             "normalize":  normalize_w,
#         },
#         "view": {
#             "log_view":   log_view_w,
#             "cmap":       cmap_w,
#             "rendering":  rendering_w,
#             "contrast_lo":contrast_lo_w,
#             "contrast_hi":contrast_hi_w,
#         },
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     setting_up = True
#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status / progress helpers ────────────────────────────────────────────
#     def set_status(msg: str) -> None:
#         # status TextEdit
#         status_w.native.append(msg)
#         # status bar (5s)
#         status_bar.showMessage(msg, 5000)
#         # napari toast (non-blocking)
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     def set_progress(value: int | None, *, busy: bool = False):
#         """Set progress bar state; busy=True puts it in 'infinite' mode."""
#         if busy:
#             progress.setRange(0, 0)  # indefinite
#         else:
#             progress.setRange(0, 100)
#             progress.setValue(int(value or 0))

#     # Hook Python stdout/stderr → status panel
#     log_emitter = QtLogEmitter()

#     def _append_log(text: str, is_err: bool):
#         if not text:
#             return
#         if is_err:
#             status_w.native.setTextColor(QtWidgets.QColor("#b22222"))  # firebrick
#             status_w.native.append(text.rstrip("\n"))
#             status_w.native.setTextColor(QtWidgets.QColor("#000000"))
#         else:
#             status_w.native.append(text.rstrip("\n"))

#     log_emitter.message.connect(_append_log)
#     stdout_writer = QtLogWriter(log_emitter, is_stderr=False)
#     stderr_writer = QtLogWriter(log_emitter, is_stderr=True)
#     # Keep originals so you could restore at exit if desired
#     sys_stdout_orig, sys_stderr_orig = sys.stdout, sys.stderr
#     sys.stdout, sys.stderr = stdout_writer, stderr_writer

#     # ── Actions ─────────────────────────────────────────────────────────────
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         set_progress(None, busy=True)
#         set_status(f"Opening TIFFs in napari: {d}")
#         open_tiffs_in_napari(d)
#         set_progress(0, busy=False)
#         set_status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         try:
#             spec = as_path_str(spec_file_w.value).strip()
#             tdir = as_path_str(tiff_dir_w.value).strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")

#             scan_list = parse_scan_list((scans_w.value or "").strip())
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_progress(None, busy=True)
#             print(f"[LOAD] Scans: {scan_list}")
#             set_status(f"Loading scans {scan_list}…")

#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),   # single YAML shared everywhere
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
#         finally:
#             set_progress(25)

#     def on_build() -> None:
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             set_progress(None, busy=True)
#             set_status("Computing Q/HKL/intensity… (verbose on)")
#             print("[BUILD] Starting compute_full(verbose=True)")
#             builder = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             # Turn on verbose to surface prints into the status panel
#             builder.compute_full(verbose=True)
#             print("[BUILD] compute_full() done")

#             state["builder"] = builder
#             state["grid"] = state["edges"] = None
#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")
#         finally:
#             set_progress(50)

#     def on_regrid() -> None:
#         try:
#             b = state["builder"]
#             if b is None:
#                 raise RuntimeError("Build the RSM map first.")

#             # Optional crop on a *fresh* builder to avoid cumulative cropping
#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 print(f"[REGRID] Cropping to y=({ymin},{ymax}), x=({xmin},{xmax})")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,  # allow streaming log messages from the gridder
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             set_progress(None, busy=True)
#             msg = f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…"
#             print(f"[REGRID] {msg}")
#             set_status(msg)

#             grid, edges = b.regrid_xu(**kwargs)

#             state["grid"], state["edges"] = grid, edges
#             print("[REGRID] Done")
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")
#         finally:
#             set_progress(75)

#     def on_view() -> None:
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")

#             lo = float(contrast_lo_w.value)
#             hi = float(contrast_hi_w.value)
#             if not (0 <= lo < hi <= 100):
#                 raise ValueError("Contrast % must satisfy 0 ≤ low < high ≤ 100")

#             set_progress(None, busy=True)
#             print(f"[VIEW] Opening viewer: log={bool(log_view_w.value)}, cmap={cmap_w.value}, render={rendering_w.value}, contrast=({lo},{hi})")
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
#             viz.launch()
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")
#         finally:
#             set_progress(100, busy=False)

#     def on_run_all() -> None:
#         btn_run_all.native.setEnabled(False)
#         try:
#             set_status("Running pipeline (Load → Build → Regrid → View)…")
#             set_progress(0)
#             on_load()
#             if state["loader"] is None:
#                 raise RuntimeError("Load failed; aborting.")
#             on_build()
#             if state["builder"] is None:
#                 raise RuntimeError("Build failed; aborting.")
#             on_regrid()
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid failed; aborting.")
#             on_view()
#             set_status("Run All completed.")
#         except Exception as e:
#             show_error(f"Run All error: {e}")
#             set_status(f"Run All failed: {e}")
#         finally:
#             btn_run_all.native.setEnabled(True)

#     # Connect actions
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)

#     # Run (Qt5/Qt6 compatible)
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app: single-YAML persistence for all parameters (beautified UI)

# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid
# • Column 3: View + Status (+ Run All)

# YAML: env RSM3D_DEFAULTS_YAML, else ~/.rsm3d_defaults.yaml
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error, show_info
# from magicgui.widgets import (
#     CheckBox,
#     ComboBox,
#     Container,
#     FileEdit,
#     FloatSpinBox,
#     Label,
#     LineEdit,
#     PushButton,
#     SpinBox,
#     TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder

# # ─────────────────────────────────────────────────────────────────────────────
# # Global style
# # ─────────────────────────────────────────────────────────────────────────────

# APP_QSS = """
# QMainWindow { background: #fafafa; }
# QGroupBox {
#     border: 1px solid #d9d9d9;
#     border-radius: 8px;
#     margin-top: 10px;
#     background: #ffffff;
#     font-weight: 600;
# }
# QGroupBox::title {
#     subcontrol-origin: margin;
#     subcontrol-position: top left;
#     padding: 4px 8px;
#     color: #2c3e50;
# }

# /* Crisp vertical divider with its own gutters */
# QSplitter::handle {
#     background: #e9edf3;        /* soft neutral */
#     border-left: 1px solid #d0d4db;
#     border-right: 1px solid #ffffff;
# }

# /* Inputs / buttons (unchanged) */
# QLabel { color: #34495e; }
# QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {
#     border: 1px solid #d4d7dd;
#     border-radius: 6px;
#     padding: 4px 6px;
#     background: #ffffff;
# }
# QPushButton {
#     background: #eef2f7;
#     border: 1px solid #d4d7dd;
#     border-radius: 8px;
#     padding: 6px 10px;
#     font-weight: 600;
# }
# QPushButton:hover { background: #e6ebf3; }
# QPushButton:pressed { background: #dfe5ee; }
# """

# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utilities
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str) -> None:
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {"spec_file": None, "tiff_dir": None, "scans": "", "only_hkl": None},
#         "ExperimentSetup": {
#             "distance": None, "pitch": None, "ycenter": None, "xcenter": None,
#             "xpixels": None, "ypixels": None, "energy": None, "wavelength": None,
#         },
#         "build": {"ub_includes_2pi": None, "center_is_one_based": None},
#         "crop": {"enable": None, "y_min": None, "y_max": None, "x_min": None, "x_max": None},
#         "regrid": {"space": None, "grid_shape": "", "fuzzy": None, "fuzzy_width": None, "normalize": None},
#         "view": {"log_view": None, "cmap": None, "rendering": None, "contrast_lo": None, "contrast_hi": None},
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")

# def as_path_str(v: Any) -> str:
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)
#     except TypeError:
#         return str(v)

# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w

# def parse_scan_list(text: str) -> List[int]:
#     if not text or not text.strip():
#         return []
#     out: set[int] = set()
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore[return-value]

# def open_tiffs_in_napari(tiff_dir: str):
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

# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────

# def make_group(title: str, inner_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
#     box = QtWidgets.QGroupBox(title)
#     lay = QtWidgets.QVBoxLayout(box)
#     lay.setContentsMargins(10, 10, 10, 10)
#     lay.setSpacing(8)
#     lay.addWidget(inner_widget)
#     return box

# def make_scroll(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
#     sc = QtWidgets.QScrollArea()
#     sc.setWidgetResizable(True)
#     sc.setFrameShape(QtWidgets.QFrame.NoFrame)
#     sc.setWidget(w)
#     return sc

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
#     app.setStyleSheet(APP_QSS)

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     setting_up = False  # avoid YAML writes during initial population

#     # ── Column 1: Data & Setup ──────────────────────────────────────────────
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")
#     wavelength_w.tooltip = "Leave empty to derive from energy. < 1e-3 is treated as meters and converted to Å."

#     # Button bar
#     btn_load = PushButton(text="📂 Load Data")
#     btn_tiff = PushButton(text="🖼️ View TIFFs in napari")
#     btn_row1 = QtWidgets.QWidget()
#     row1 = QtWidgets.QHBoxLayout(btn_row1); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
#     row1.addWidget(btn_load.native); row1.addWidget(btn_tiff.native); row1.addStretch(1)

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             hsep(),
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             hsep(),
#         ],
#     )

#     # ── Column 2: Build & Regrid ────────────────────────────────────────────
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="🔧 Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="🧮 Regrid")

#     # Button bar 2
#     btn_row2 = QtWidgets.QWidget()
#     row2 = QtWidgets.QHBoxLayout(btn_row2); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
#     row2.addWidget(btn_build.native); row2.addWidget(btn_regrid.native); row2.addStretch(1)

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w,
#             hsep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             hsep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             hsep(),
#         ],
#     )

#     # ── Column 3: View + Status ─────────────────────────────────────────────
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     btn_view     = PushButton(text="🌈 View RSM")
#     btn_run_all  = PushButton(text="▶️ Run All")

#     btn_row3 = QtWidgets.QWidget()
#     row3 = QtWidgets.QHBoxLayout(btn_row3); row3.setContentsMargins(0,0,0,0); row3.setSpacing(8)
#     row3.addWidget(btn_view.native); row3.addWidget(btn_run_all.native); row3.addStretch(1)

#     status_label_w = Label(value="Status")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(140)
#         status_w.native.setStyleSheet("font-family: Consolas, Menlo, monospace;")
#     except Exception:
#         pass

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w,
#             hsep(),
#         ],
#     )

#     # ── Wrap columns in group boxes + scroll areas ───────────────────────────
#     g1 = make_group("① Data & Setup", col1.native)
#     g1_layout = g1.layout(); g1_layout.addWidget(btn_row1)

#     g2 = make_group("② Build  &  ③ Regrid", col2.native)
#     g2.layout().addWidget(btn_row2)

#     g3 = make_group("④ View", col3.native)
#     g3_lay = g3.layout()
#     g3_lay.addWidget(btn_row3)
#     g3_lay.addWidget(hsep().native)
#     g3_lay.addWidget(QtWidgets.QLabel("<b>Status</b>"))
#     g3_lay.addWidget(status_w.native)

#     s1 = make_scroll(g1)
#     s2 = make_scroll(g2)
#     s3 = make_scroll(g3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(s1)
#     splitter.addWidget(s2)
#     splitter.addWidget(s3)
#     splitter.setHandleWidth(4)
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     splitter.setSizes([420, 420, 420])

#     # Main window + status bar mirroring the text panel
#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Single YAML")
#     win.setCentralWidget(splitter)
#     win.resize(1320, 780)
#     status_bar = QtWidgets.QStatusBar()
#     win.setStatusBar(status_bar)
#     win.show()

#     # ── YAML binding (load → UI; UI → save) ─────────────────────────────────
#     ydoc = load_yaml(ypath)  # reload post-UI

#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {
#             "spec_file": spec_file_w,
#             "tiff_dir":  tiff_dir_w,
#             "scans":     scans_w,
#             "only_hkl":  only_hkl_w,
#         },
#         "ExperimentSetup": {
#             "distance":   distance_w,
#             "pitch":      pitch_w,
#             "ycenter":    ycenter_w,
#             "xcenter":    xcenter_w,
#             "xpixels":    xpixels_w,
#             "ypixels":    ypixels_w,
#             "energy":     energy_w,
#             "wavelength": wavelength_w,
#         },
#         "build": {
#             "ub_includes_2pi":    ub_2pi_w,
#             "center_is_one_based": center_one_based_w,
#         },
#         "crop": {
#             "enable": crop_enable_w,
#             "y_min":  y_min_w, "y_max": y_max_w,
#             "x_min":  x_min_w, "x_max": x_max_w,
#         },
#         "regrid": {
#             "space":      space_w,
#             "grid_shape": grid_shape_w,
#             "fuzzy":      fuzzy_w,
#             "fuzzy_width":fuzzy_width_w,
#             "normalize":  normalize_w,
#         },
#         "view": {
#             "log_view":   log_view_w,
#             "cmap":       cmap_w,
#             "rendering":  rendering_w,
#             "contrast_lo":contrast_lo_w,
#             "contrast_hi":contrast_hi_w,
#         },
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     setting_up = True
#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ── Status helper ────────────────────────────────────────────────────────
#     def set_status(msg: str) -> None:
#         status_w.value = msg
#         status_bar.showMessage(msg, 5000)
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ── Actions ─────────────────────────────────────────────────────────────
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         try:
#             spec = as_path_str(spec_file_w.value).strip()
#             tdir = as_path_str(tiff_dir_w.value).strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")

#             scan_list = parse_scan_list((scans_w.value or "").strip())
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             loader = RSMDataLoader(
#                 spec,
#                 yaml_path(),  # single YAML shared everywhere
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

#     def on_build() -> None:
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             set_status("Computing Q/HKL/intensity…")
#             builder = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             builder.compute_full(verbose=False)
#             state["builder"] = builder
#             state["grid"] = state["edges"] = None
#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")

#     def on_regrid() -> None:
#         try:
#             b = state["builder"]
#             if b is None:
#                 raise RuntimeError("Build the RSM map first.")

#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min < y_max and x_min < x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#             grid, edges = b.regrid_xu(**kwargs)
#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")

#     def on_view() -> None:
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
#             viz.launch()
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     def on_run_all() -> None:
#         btn_run_all.enabled = False
#         try:
#             set_status("Running pipeline (Load → Build → Regrid → View)…")
#             on_load()
#             if state["loader"] is None:
#                 raise RuntimeError("Load failed; aborting.")
#             on_build()
#             if state["builder"] is None:
#                 raise RuntimeError("Build failed; aborting.")
#             on_regrid()
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid failed; aborting.")
#             on_view()
#             set_status("Run All completed.")
#         except Exception as e:
#             show_error(f"Run All error: {e}")
#             set_status(f"Run All failed: {e}")
#         finally:
#             btn_run_all.enabled = True

#     # Connect actions
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)

#     # Run (Qt5/Qt6 compatible)
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RSM3D app: single-YAML persistence for all parameters.

# Layout
# ------
# • Column 1: Data paths → Experiment/Detector params → Buttons (Load, View TIFFs)
# • Column 2: Build + optional Crop + Regrid
# • Column 3: View + Status (+ Run All)

# YAML
# ----
# • Path from env var RSM3D_DEFAULTS_YAML; else ~/.rsm3d_defaults.yaml
# • App loads at startup and writes back on any change.

# Requirements (example)
# ----------------------
# pip install napari magicgui qtpy pyyaml xrayutilities  # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import pathlib
# import re
# import sys
# from typing import Any, Dict, List, Tuple

# import yaml
# import napari
# from qtpy import QtCore, QtWidgets
# from napari.utils.notifications import show_error, show_info
# from magicgui.widgets import (
#     CheckBox,
#     ComboBox,
#     Container,
#     FileEdit,
#     FloatSpinBox,
#     Label,
#     LineEdit,
#     PushButton,
#     SpinBox,
#     TextEdit,
# )

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.data_viz import RSMNapariViewer
# from rsm3d.rsm3d import RSMBuilder

# # ─────────────────────────────────────────────────────────────────────────────
# # YAML utilities
# # ─────────────────────────────────────────────────────────────────────────────

# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
# )


# def yaml_path() -> str:
#     """Resolve the YAML path from env, else fall back to ~/.rsm3d_defaults.yaml."""
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")


# def ensure_yaml(path: str) -> None:
#     """Create minimal structure if missing (no hard-coded defaults)."""
#     if os.path.isfile(path):
#         return

#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {
#             "spec_file": None,
#             "tiff_dir": None,
#             "scans": "",
#             "only_hkl": None,
#         },
#         "ExperimentSetup": {
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
#             "y_min": None,
#             "y_max": None,
#             "x_min": None,
#             "x_max": None,
#         },
#         "regrid": {
#             "space": None,         # "hkl" | "q"
#             "grid_shape": "",      # e.g. "200,*,*"
#             "fuzzy": None,
#             "fuzzy_width": None,
#             "normalize": None,     # "mean" | "sum"
#         },
#         "view": {
#             "log_view": None,
#             "cmap": None,          # e.g. "inferno"
#             "rendering": None,     # "attenuated_mip" | "mip" | "translucent"
#             "contrast_lo": None,   # %
#             "contrast_hi": None,   # %
#         },
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)


# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}


# def save_yaml(path: str, doc: Dict[str, Any]) -> None:
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")


# def as_path_str(v: Any) -> str:
#     """Normalize Path/str/None from FileEdit to a clean string."""
#     if v is None:
#         return ""
#     try:
#         return os.fspath(v)  # supports Path and str (PEP 519)
#     except TypeError:
#         return str(v)


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def hsep(height: int = 10) -> Label:
#     """Horizontal separator using a styled label."""
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(height)
#     except Exception:
#         pass
#     return w


# def parse_scan_list(text: str) -> List[int]:
#     """Parse '17, 18-22, 30' → [17, 18, 19, 20, 21, 22, 30]."""
#     if not text or not text.strip():
#         return []

#     out: set[int] = set()
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


# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     """
#     Parse 'x,y,z' where y/z may be omitted or '*'.
#     Examples: '200,*,*' → (200, None, None), '256,256,256' → (256, 256, 256), '200' → (200, None, None)
#     """
#     if text is None:
#         return (None, None, None)

#     s = text.strip()
#     if not s:
#         return (None, None, None)

#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")

#     def one(p: str) -> int | None:
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v

#     return tuple(one(p) for p in parts)  # type: ignore[return-value]


# def open_tiffs_in_napari(tiff_dir: str):
#     """Open all TIFFs in a folder in a new napari Viewer."""
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


# # ─────────────────────────────────────────────────────────────────────────────
# # App
# # ─────────────────────────────────────────────────────────────────────────────

# def main() -> None:
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     # Optional: small UI polish for splitter handle visibility on some themes
#     app.setStyleSheet("QSplitter::handle { background: #d0d0d0; }")

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     # Suppress YAML writes during initial widget population
#     setting_up = False

#     # ── Column 1: Data paths → Experiment/Detector → Buttons ────────────────
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w = FileEdit(mode="d", label="TIFF folder")
#     scans_w = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl_w = CheckBox(label="Only HKL scans")

#     scans_w.tooltip = "Comma/range list. Examples: 17, 18-22, 30"

#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w = FloatSpinBox(label="pitch (m)", min=-1e9, max=1e9, step=1e-9)
#     ycenter_w = SpinBox(label="ycenter (px)", min=0, max=10_000_000, step=1)
#     xcenter_w = SpinBox(label="xcenter (px)", min=0, max=10_000_000, step=1)
#     xpixels_w = SpinBox(label="xpixels", min=0, max=10_000_000, step=1)
#     ypixels_w = SpinBox(label="ypixels", min=0, max=10_000_000, step=1)
#     energy_w = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")
#     wavelength_w.tooltip = "Leave empty to derive from energy. Values < 1e-3 assumed meters and converted to Å."

#     btn_load = PushButton(text="Load Data")
#     btn_tiff = PushButton(text="View TIFFs in napari")

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w,
#             tiff_dir_w,
#             scans_w,
#             only_hkl_w,
#             hsep(),
#             title_params,
#             distance_w,
#             pitch_w,
#             ycenter_w,
#             xcenter_w,
#             xpixels_w,
#             ypixels_w,
#             energy_w,
#             wavelength_w,
#             hsep(),
#             btn_load,
#             btn_tiff,
#         ],
#     )

#     # ── Column 2: Build + Crop + Regrid ─────────────────────────────────────
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build = PushButton(text="Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     grid_shape_w.tooltip = "Examples: 200,*,* or 256,256,256 or just 200"
#     fuzzy_w = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w = CheckBox(label="Crop before regrid")
#     y_min_w = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid = PushButton(text="Regrid")

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build,
#             ub_2pi_w,
#             center_one_based_w,
#             btn_build,
#             hsep(),
#             title_regrid,
#             space_w,
#             grid_shape_w,
#             fuzzy_w,
#             fuzzy_width_w,
#             normalize_w,
#             hsep(),
#             title_crop,
#             crop_enable_w,
#             y_min_w,
#             y_max_w,
#             x_min_w,
#             x_max_w,
#             btn_regrid,
#         ],
#     )

#     # ── Column 3: View ──────────────────────────────────────────────────────
#     title_view = Label(value="<b>View</b>")
#     log_view_w = CheckBox(label="Log view")
#     cmap_w = ComboBox(
#         label="Colormap",
#         choices=["viridis", "inferno", "magma", "plasma", "cividis"],
#     )
#     rendering_w = ComboBox(
#         label="Rendering",
#         choices=["attenuated_mip", "mip", "translucent"],
#     )
#     contrast_lo_w = FloatSpinBox(label="Contrast low (%)", min=0.0, max=100.0, step=0.1)
#     contrast_hi_w = FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)

#     btn_view = PushButton(text="View RSM")
#     btn_run_all = PushButton(text="Run All")

#     status_label_w = Label(value="Status")
#     status_w = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(120)
#         status_w.native.setStyleSheet("font-family: Consolas, Menlo, monospace;")
#     except Exception:
#         pass

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title_view,
#             log_view_w,
#             cmap_w,
#             rendering_w,
#             contrast_lo_w,
#             contrast_hi_w,
#             btn_view,
#             btn_run_all,
#             hsep(),
#             status_label_w,
#             status_w,
#         ],
#     )

#     # ── Wrap in a 3-way splitter ────────────────────────────────────────────
#     def wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         vbox = QtWidgets.QVBoxLayout(host)
#         vbox.setContentsMargins(8, 8, 8, 8)
#         vbox.setSpacing(6)
#         hdr = QtWidgets.QLabel(f"<b>{title}</b>")
#         vbox.addWidget(hdr)
#         vbox.addWidget(container.native, 1)
#         return host

#     w1 = wrap("① Data & Setup", col1)
#     w2 = wrap("② Build  &  ③ Regrid", col2)
#     w3 = wrap("④ View", col3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w1)
#     splitter.addWidget(w2)
#     splitter.addWidget(w3)
#     splitter.setHandleWidth(2)
#     splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1)
#     splitter.setStretchFactor(1, 1)
#     splitter.setStretchFactor(2, 1)
#     splitter.setSizes([400, 400, 400])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Single YAML")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ── Bind widgets to YAML (load → UI, UI → save) ─────────────────────────
#     ydoc = load_yaml(ypath)  # reload after UI built

#     widget_map: Dict[str, Dict[str, Any]] = {
#         "data": {
#             "spec_file": spec_file_w,
#             "tiff_dir": tiff_dir_w,
#             "scans": scans_w,
#             "only_hkl": only_hkl_w,
#         },
#         "ExperimentSetup": {
#             "distance": distance_w,
#             "pitch": pitch_w,
#             "ycenter": ycenter_w,
#             "xcenter": xcenter_w,
#             "xpixels": xpixels_w,
#             "ypixels": ypixels_w,
#             "energy": energy_w,
#             "wavelength": wavelength_w,
#         },
#         "build": {
#             "ub_includes_2pi": ub_2pi_w,
#             "center_is_one_based": center_one_based_w,
#         },
#         "crop": {
#             "enable": crop_enable_w,
#             "y_min": y_min_w,
#             "y_max": y_max_w,
#             "x_min": x_min_w,
#             "x_max": x_max_w,
#         },
#         "regrid": {
#             "space": space_w,
#             "grid_shape": grid_shape_w,
#             "fuzzy": fuzzy_w,
#             "fuzzy_width": fuzzy_width_w,
#             "normalize": normalize_w,
#         },
#         "view": {
#             "log_view": log_view_w,
#             "cmap": cmap_w,
#             "rendering": rendering_w,
#             "contrast_lo": contrast_lo_w,
#             "contrast_hi": contrast_hi_w,
#         },
#     }

#     def set_widget(widget: Any, value: Any) -> None:
#         """Populate widget with a value, skipping None."""
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 sval = str(value)
#                 if sval in widget.choices:
#                     widget.value = sval
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     # Populate UI from YAML
#     setting_up = True
#     for section, mapping in widget_map.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in widget_map:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     def val_for_yaml(widget: Any, section: str, key: str) -> Any:
#         """Extract a widget's value in a YAML-serializable form."""
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt

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

#     def on_changed(section: str, key: str, widget: Any) -> None:
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     # Connect all widget changes to YAML persistence
#     for section, mapping in widget_map.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(
#                 lambda *_, s=section, k=key, w=widget: on_changed(s, k, w)
#             )

#     # ── Status helper ────────────────────────────────────────────────────────
#     def set_status(msg: str) -> None:
#         status_w.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ── Actions ─────────────────────────────────────────────────────────────
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs() -> None:
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load() -> None:
#         try:
#             spec = as_path_str(spec_file_w.value).strip()
#             tdir = as_path_str(tiff_dir_w.value).strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")

#             scan_list = parse_scan_list((scans_w.value or "").strip())
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")

#             # Pass single YAML so loader reads ExperimentSetup from the same file
#             loader = RSMDataLoader(
#                 spec,
#                 ypath,
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

#     def on_build() -> None:
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             set_status("Computing Q/HKL/intensity…")

#             builder = RSMBuilder(
#                 state["loader"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             builder.compute_full(verbose=False)

#             state["builder"] = builder
#             state["grid"] = state["edges"] = None
#             set_status("RSM map built.")
#         except Exception as e:
#             show_error(f"Build error: {e}")
#             set_status(f"Build failed: {e}")

#     def on_regrid() -> None:
#         try:
#             b = state["builder"]
#             if b is None:
#                 raise RuntimeError("Build the RSM map first.")

#             # Optional crop: use a fresh builder to avoid cumulative cropping
#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError(
#                         "Crop bounds must satisfy y_min < y_max and x_min < x_max."
#                     )
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")

#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")

#             kwargs = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kwargs["width"] = float(fuzzy_width_w.value)

#             set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#             grid, edges = b.regrid_xu(**kwargs)

#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")

#     def on_view() -> None:
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
#             viz.launch()
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     def on_run_all() -> None:
#         """Execute: Load → Build → Regrid → View."""
#         btn_run_all.enabled = False
#         try:
#             set_status("Running pipeline (Load → Build → Regrid → View)…")
#             on_load()
#             if state["loader"] is None:
#                 raise RuntimeError("Load failed; aborting.")

#             on_build()
#             if state["builder"] is None:
#                 raise RuntimeError("Build failed; aborting.")

#             on_regrid()
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid failed; aborting.")

#             on_view()
#             set_status("Run All completed.")
#         except Exception as e:
#             show_error(f"Run All error: {e}")
#             set_status(f"Run All failed: {e}")
#         finally:
#             btn_run_all.enabled = True

#     # Connections
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)
#     btn_run_all.clicked.connect(on_run_all)

#     # Run (Qt5/Qt6 compatible)
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python3
# """
# RSM3D app: single-YAML persistence for all parameters.
# - Column 1: Data path boxes -> Setup parameters -> Buttons (Load Data, View TIFFs)
# - Column 2: Build + optional Crop + Regrid
# - Column 3: View + Status

# YAML path:
# - Use env var RSM3D_DEFAULTS_YAML if set, else ~/.rsm3d_defaults.yaml
# - App loads at startup and writes back immediately on any change.

# Requirements (example):
#   pip install napari magicgui qtpy pyyaml xrayutilities   # plus your rsm3d package
# """

# from __future__ import annotations

# import os, pathlib
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

# import yaml  # required

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ────────────────────────── YAML utils ──────────────────────────
# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(
#     DEFAULTS_ENV,
#     str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve())
# )

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str):
#     """Create minimal structure if missing (no hard-coded default values)."""
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {
#             "spec_file": None,
#             "tiff_dir": None,
#             "scans": "",
#             "only_hkl": None,
#         },
#         "ExperimentSetup": {
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
#             "space": None,        # "hkl"/"q"
#             "grid_shape": "",     # e.g. "200,*,*"
#             "fuzzy": None,
#             "fuzzy_width": None,
#             "normalize": None,    # "mean"/"sum"
#         },
#         "view": {
#             "log_view": None,
#             "cmap": None,         # e.g. "inferno"
#             "rendering": None,    # "attenuated_mip"/"mip"/"translucent"
#             "contrast_lo": None,  # %
#             "contrast_hi": None,  # %
#         },
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]):
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")


# def as_path_str(v) -> str:
#     """Return a normalized string path for FileEdit values (Path/str/None)."""
#     if v is None:
#         return ""
#     try:
#         # Works for Path and str (PEP 519)
#         return os.fspath(v)
#     except TypeError:
#         return str(v)

# # ────────────────────────── helpers ──────────────────────────
# def HSep(h: int = 10):
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(h)
#     except Exception:
#         pass
#     return w

# def parse_scan_list(text: str) -> List[int]:
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p):
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore

# def open_tiffs_in_napari(tiff_dir: str):
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


# # ────────────────────────── app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     # Flag to suppress save while populating from YAML
#     setting_up = False

#     # ---------- Column 1: Data paths -> Setup (ExperimentSetup) -> Buttons ----------
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")

#     btn_load = PushButton(text="Load Data")
#     btn_tiff = PushButton(text="View TIFFs in napari")

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             HSep(),
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             HSep(),
#             btn_load, btn_tiff,
#         ],
#     )

#     # ---------- Column 2: Build + Crop + Regrid ----------
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="Regrid")

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w, btn_build,
#             HSep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             HSep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             btn_regrid,
#         ],
#     )

#     # ---------- Column 3: View ----------
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)
#     btn_view     = PushButton(text="View RSM")
#     btn_run_all  = PushButton(text="Run All")
#     status_label_w = Label(value="Status")
#     status_w       = TextEdit(value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(120)
#     except Exception:
#         pass
    
    
#     col3 = Container(
#             layout="vertical",
#                 widgets=[
#                 title_view,
#                 log_view_w,
#                 cmap_w,
#                 rendering_w,
#                 contrast_lo_w,
#                 contrast_hi_w,
#                 btn_view,
#                 btn_run_all,
#                 HSep(),
#                 status_label_w,
#                 status_w,
#                 ],
#                 )
    

#     # ---------- Wrap in a 3-way splitter ----------
#     def wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
#         t = QtWidgets.QLabel(f"<b>{title}</b>")
#         v.addWidget(t)
#         v.addWidget(container.native, 1)
#         return host

#     w1 = wrap("① Data & Setup", col1)
#     w2 = wrap("② Build  &  ③ Regrid", col2)
#     w3 = wrap("④ View", col3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w1); splitter.addWidget(w2); splitter.addWidget(w3)
#     splitter.setHandleWidth(2); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([400, 400, 400])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Single YAML")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ---------- bind widgets to YAML (load → UI, UI → save) ----------
#     ydoc = load_yaml(ypath)  # reload after UI built

#     W: Dict[str, Dict[str, Any]] = {
#         "data": {
#             "spec_file": spec_file_w,
#             "tiff_dir":  tiff_dir_w,
#             "scans":     scans_w,
#             "only_hkl":  only_hkl_w,
#         },
#         "ExperimentSetup": {
#             "distance":   distance_w,
#             "pitch":      pitch_w,
#             "ycenter":    ycenter_w,
#             "xcenter":    xcenter_w,
#             "xpixels":    xpixels_w,
#             "ypixels":    ypixels_w,
#             "energy":     energy_w,
#             "wavelength": wavelength_w,
#         },
#         "build": {
#             "ub_includes_2pi":    ub_2pi_w,
#             "center_is_one_based": center_one_based_w,
#         },
#         "crop": {
#             "enable": crop_enable_w,
#             "y_min":  y_min_w, "y_max": y_max_w,
#             "x_min":  x_min_w, "x_max": x_max_w,
#         },
#         "regrid": {
#             "space":      space_w,
#             "grid_shape": grid_shape_w,
#             "fuzzy":      fuzzy_w,
#             "fuzzy_width":fuzzy_width_w,
#             "normalize":  normalize_w,
#         },
#         "view": {
#             "log_view":   log_view_w,
#             "cmap":       cmap_w,
#             "rendering":  rendering_w,
#             "contrast_lo":contrast_lo_w,
#             "contrast_hi":contrast_hi_w,
#         },
#     }

#     def set_widget(widget, value):
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 if str(value) in widget.choices:
#                     widget.value = str(value)
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     setting_up = True
#     for section, mapping in W.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     for s in W:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     def val_for_yaml(widget, section: str, key: str):
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget):
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in W.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ---------- status helper ----------
#     def set_status(msg: str):
#         status_w.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ---------- actions ----------
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs():
#         d = as_path_str(tiff_dir_w.value).strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")
    
#     def on_load():
#         try:
#             spec = as_path_str(spec_file_w.value).strip()
#             tdir = as_path_str(tiff_dir_w.value).strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = parse_scan_list((scans_w.value or "").strip())

#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             # Use the single YAML for ExperimentSetup (no separate setup file)
#             loader = RSMDataLoader(
#                 spec,
#                 ypath,         # pass our single YAML so ExperimentSetup is honored
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
#             # compute_full may return (Q_samp, hkl, intensity) or set internals; both are okay
#             builder.compute_full(verbose=False)

#             state["builder"] = builder
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

#             # Optional crop on a FRESH builder to avoid cumulative cropping
#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")
#             kw = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
#                 kw["width"] = float(fuzzy_width_w.value)

#             set_status(f"Regridding to {space_w.value.upper()} grid {(gx, gy, gz)}…")
#             grid, edges = b.regrid_xu(**kw)
#             state["grid"], state["edges"] = grid, edges
#             set_status("Regrid completed.")
#         except Exception as e:
#             show_error(f"Regrid error: {e}")
#             set_status(f"Regrid failed: {e}")
            
#     def on_run_all():
#         # Execute: load → build → regrid → view
#         btn_run_all.enabled = False
#         try:
#             set_status("Running full pipeline (Load → Build → Regrid → View)…")
#             on_load()
#             if state["loader"] is None:
#                 raise RuntimeError("Load failed; aborting.")
#             on_build()
#             if state["builder"] is None:
#                 raise RuntimeError("Build failed; aborting.")
#             on_regrid()
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid failed; aborting.")
#             on_view()
#             set_status("Run All completed.")
#         except Exception as e:
#             show_error(f"Run All error: {e}")
#             set_status(f"Run All failed: {e}")
#         finally:
#             btn_run_all.enabled = True

#     def on_view():
#         try:
#             if state["grid"] is None or state["edges"] is None:
#                 raise RuntimeError("Regrid first.")
#             lo = contrast_lo_w.value
#             hi = contrast_hi_w.value
#             if lo is None or hi is None:
#                 raise ValueError("Set contrast low/high percentages.")
#             lo = float(lo); hi = float(hi)
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
#             viz.launch()
#             set_status("RSM viewer opened.")
#         except Exception as e:
#             show_error(f"View error: {e}")
#             set_status(f"View failed: {e}")

#     # connect
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_run_all.clicked.connect(on_run_all)
#     btn_view.clicked.connect(on_view)

#     # run (Qt5/Qt6 compatible)
#     exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
#     sys.exit(exec_fn())


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# RSM3D app: single-YAML persistence for all parameters.
# - Column 1: Data path boxes -> Setup parameters -> Buttons (Load Data, View TIFFs)
# - Column 2: Build + optional Crop + Regrid
# - Column 3: View + Status

# YAML path:
# - Use env var RSM3D_DEFAULTS_YAML if set, else ~/.rsm3d_defaults.yaml
# - App loads at startup and writes back immediately on any change.

# Requirements (example):
#   pip install napari magicgui qtpy pyyaml xrayutilities   # plus your rsm3d package
# """

# from __future__ import annotations

# import os, pathlib
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

# import yaml  # required

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ────────────────────────── YAML utils ──────────────────────────
# # import os, pathlib
# DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# os.environ.setdefault(DEFAULTS_ENV, str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()))
# # DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"

# def yaml_path() -> str:
#     p = os.environ.get(DEFAULTS_ENV, "").strip()
#     if p:
#         return os.path.abspath(os.path.expanduser(p))
#     return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

# def ensure_yaml(path: str):
#     """Create minimal structure if missing (no hard-coded default values)."""
#     if os.path.isfile(path):
#         return
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     seed = {
#         "data": {
#             "spec_file": None,
#             "tiff_dir": None,
#             "scans": "",
#             "only_hkl": None,
#         },
#         "ExperimentSetup": {
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
#             "space": None,        # "hkl"/"q"
#             "grid_shape": "",     # e.g. "200,*,*"
#             "fuzzy": None,
#             "fuzzy_width": None,
#             "normalize": None,    # "mean"/"sum"
#         },
#         "view": {
#             "log_view": None,
#             "cmap": None,         # e.g. "inferno"
#             "rendering": None,    # "attenuated_mip"/"mip"/"translucent"
#             "contrast_lo": None,  # %
#             "contrast_hi": None,  # %
#         },
#     }
#     with open(path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(seed, f, sort_keys=False)

# def load_yaml(path: str) -> Dict[str, Any]:
#     try:
#         return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
#     except Exception:
#         return {}

# def save_yaml(path: str, doc: Dict[str, Any]):
#     try:
#         with open(path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(doc, f, sort_keys=False)
#     except Exception as e:
#         show_error(f"Failed to write YAML: {e}")


# # ────────────────────────── helpers ──────────────────────────
# def HSep(h: int = 10):
#     w = Label(value="")
#     try:
#         w.native.setFrameShape(QtWidgets.QFrame.HLine)
#         w.native.setFrameShadow(QtWidgets.QFrame.Sunken)
#         w.native.setLineWidth(1)
#         w.native.setFixedHeight(h)
#     except Exception:
#         pass
#     return w

# def parse_scan_list(text: str) -> List[int]:
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

# def parse_grid_shape(text: str) -> Tuple[int | None, int | None, int | None]:
#     if text is None:
#         return (None, None, None)
#     s = text.strip()
#     if not s:
#         return (None, None, None)
#     parts = [p.strip() for p in s.split(",")]
#     if len(parts) == 1:
#         parts += ["*", "*"]
#     if len(parts) != 3:
#         raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")
#     def one(p):
#         if p in ("*", "", None):
#             return None
#         if not p.isdigit():
#             raise ValueError(f"Grid size must be integer or '*', got '{p}'")
#         v = int(p)
#         if v <= 0:
#             raise ValueError("Grid sizes must be > 0")
#         return v
#     return tuple(one(p) for p in parts)  # type: ignore

# def open_tiffs_in_napari(tiff_dir: str):
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


# # ────────────────────────── app ──────────────────────────
# def main():
#     app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

#     ypath = yaml_path()
#     ensure_yaml(ypath)
#     ydoc = load_yaml(ypath)

#     # Flag to suppress save while populating from YAML
#     setting_up = False

#     # ---------- Column 1: Data paths -> Setup (experiment) -> Buttons ----------
#     # Data path boxes FIRST
#     spec_file_w = FileEdit(mode="r", label="SPEC file")
#     tiff_dir_w  = FileEdit(mode="d", label="TIFF folder")
#     scans_w     = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl_w  = CheckBox(label="Only HKL scans")

#     # Setup parameters (experiment/detector)
#     title_params = Label(value="<b>Experiment / Detector</b>")
#     distance_w   = FloatSpinBox(label="distance (m)", min=-1e9, max=1e9, step=1e-6)
#     pitch_w      = FloatSpinBox(label="pitch (m)",    min=-1e9, max=1e9, step=1e-9)
#     ycenter_w    = SpinBox(label="ycenter (px)",      min=0,    max=10_000_000, step=1)
#     xcenter_w    = SpinBox(label="xcenter (px)",      min=0,    max=10_000_000, step=1)
#     xpixels_w    = SpinBox(label="xpixels",           min=0,    max=10_000_000, step=1)
#     ypixels_w    = SpinBox(label="ypixels",           min=0,    max=10_000_000, step=1)
#     energy_w     = FloatSpinBox(label="energy (keV)", min=-1e9, max=1e9, step=1e-3)
#     wavelength_w = LineEdit(label="wavelength (Å or None)")

#     # Buttons at the BOTTOM
#     btn_load = PushButton(text="Load Data")
#     btn_tiff = PushButton(text="View TIFFs in napari")

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             # Data first
#             spec_file_w, tiff_dir_w, scans_w, only_hkl_w,
#             HSep(),
#             # Then setup parameters
#             title_params,
#             distance_w, pitch_w, ycenter_w, xcenter_w, xpixels_w, ypixels_w, energy_w, wavelength_w,
#             HSep(),
#             # Buttons last
#             btn_load, btn_tiff,
#         ],
#     )

#     # ---------- Column 2: Build + Crop + Regrid ----------
#     title_build = Label(value="<b>Build</b>")
#     ub_2pi_w    = CheckBox(label="UB includes 2π")
#     center_one_based_w = CheckBox(label="1-based center")
#     btn_build   = PushButton(text="Build RSM Map")

#     title_regrid = Label(value="<b>Regrid</b>")
#     space_w      = ComboBox(label="Space", choices=["hkl", "q"])
#     grid_shape_w = LineEdit(label="Grid (x,y,z), '*' allowed")
#     fuzzy_w      = CheckBox(label="Fuzzy gridder")
#     fuzzy_width_w= FloatSpinBox(label="Width (fuzzy)", min=0.0, max=1e9, step=0.01)
#     normalize_w  = ComboBox(label="Normalize", choices=["mean", "sum"])

#     title_crop   = Label(value="<b>Optional crop (pixel bounds)</b>")
#     crop_enable_w= CheckBox(label="Crop before regrid")
#     y_min_w      = SpinBox(label="y_min", min=0, max=10_000_000, step=1)
#     y_max_w      = SpinBox(label="y_max", min=0, max=10_000_000, step=1)
#     x_min_w      = SpinBox(label="x_min", min=0, max=10_000_000, step=1)
#     x_max_w      = SpinBox(label="x_max", min=0, max=10_000_000, step=1)

#     btn_regrid   = PushButton(text="Regrid")

#     col2 = Container(
#         layout="vertical",
#         widgets=[
#             title_build, ub_2pi_w, center_one_based_w, btn_build,
#             HSep(),
#             title_regrid, space_w, grid_shape_w, fuzzy_w, fuzzy_width_w, normalize_w,
#             HSep(),
#             title_crop, crop_enable_w, y_min_w, y_max_w, x_min_w, x_max_w,
#             btn_regrid,
#         ],
#     )

#     # ---------- Column 3: View ----------
#     title_view   = Label(value="<b>View</b>")
#     log_view_w   = CheckBox(label="Log view")
#     cmap_w       = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"])
#     rendering_w  = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"])
#     contrast_lo_w= FloatSpinBox(label="Contrast low (%)",  min=0.0, max=100.0, step=0.1)
#     contrast_hi_w= FloatSpinBox(label="Contrast high (%)", min=0.0, max=100.0, step=0.1)
#     btn_view     = PushButton(text="View RSM")
#     status_w     = TextEdit(label="Status", value="")
#     try:
#         status_w.native.setReadOnly(True)
#         status_w.native.setMinimumHeight(120)
#     except Exception:
#         pass

#     col3 = Container(
#         layout="vertical",
#         widgets=[title_view, log_view_w, cmap_w, rendering_w, contrast_lo_w, contrast_hi_w, btn_view, HSep(), status_w],
#     )

#     # ---------- Wrap in a 3-way splitter ----------
#     def wrap(title: str, container: Container) -> QtWidgets.QWidget:
#         host = QtWidgets.QWidget()
#         v = QtWidgets.QVBoxLayout(host); v.setContentsMargins(8, 8, 8, 8); v.setSpacing(6)
#         t = QtWidgets.QLabel(f"<b>{title}</b>")
#         v.addWidget(t)
#         v.addWidget(container.native, 1)
#         return host

#     w1 = wrap("① Data & Setup", col1)
#     w2 = wrap("② Build  &  ③ Regrid", col2)
#     w3 = wrap("④ View", col3)

#     splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
#     splitter.addWidget(w1); splitter.addWidget(w2); splitter.addWidget(w3)
#     splitter.setHandleWidth(2); splitter.setChildrenCollapsible(False)
#     splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 1)
#     splitter.setSizes([400, 400, 400])

#     win = QtWidgets.QMainWindow()
#     win.setWindowTitle("RSM3D (Qt + magicgui) — Single YAML")
#     win.setCentralWidget(splitter)
#     win.resize(1280, 720)
#     win.show()

#     # ---------- bind widgets to YAML (load → UI, UI → save) ----------
#     ydoc = load_yaml(ypath)  # reload after UI built

#     # Map: section -> { key: widget }
#     W: Dict[str, Dict[str, Any]] = {
#         "data": {
#             "spec_file": spec_file_w,
#             "tiff_dir":  tiff_dir_w,
#             "scans":     scans_w,
#             "only_hkl":  only_hkl_w,
#         },
#         "ExperimentSetup": {
#             "distance":   distance_w,
#             "pitch":      pitch_w,
#             "ycenter":    ycenter_w,
#             "xcenter":    xcenter_w,
#             "xpixels":    xpixels_w,
#             "ypixels":    ypixels_w,
#             "energy":     energy_w,
#             "wavelength": wavelength_w,
#         },
#         "build": {
#             "ub_includes_2pi":    ub_2pi_w,
#             "center_is_one_based": center_one_based_w,
#         },
#         "crop": {
#             "enable": crop_enable_w,
#             "y_min":  y_min_w, "y_max": y_max_w,
#             "x_min":  x_min_w, "x_max": x_max_w,
#         },
#         "regrid": {
#             "space":      space_w,
#             "grid_shape": grid_shape_w,
#             "fuzzy":      fuzzy_w,
#             "fuzzy_width":fuzzy_width_w,
#             "normalize":  normalize_w,
#         },
#         "view": {
#             "log_view":   log_view_w,
#             "cmap":       cmap_w,
#             "rendering":  rendering_w,
#             "contrast_lo":contrast_lo_w,
#             "contrast_hi":contrast_hi_w,
#         },
#     }

#     def set_widget(widget, value):
#         try:
#             if value is None:
#                 return
#             if isinstance(widget, (FloatSpinBox, SpinBox)):
#                 widget.value = float(value) if isinstance(widget, FloatSpinBox) else int(value)
#             elif isinstance(widget, CheckBox):
#                 widget.value = bool(value)
#             elif isinstance(widget, ComboBox):
#                 # only set if valid
#                 if str(value) in widget.choices:
#                     widget.value = str(value)
#             elif isinstance(widget, (LineEdit, TextEdit, FileEdit)):
#                 widget.value = str(value)
#         except Exception:
#             pass

#     # populate UI
#     setting_up = True
#     for section, mapping in W.items():
#         vals = ydoc.get(section, {})
#         for key, widget in mapping.items():
#             set_widget(widget, vals.get(key, None))
#     # ensure structure present
#     for s in W:
#         ydoc.setdefault(s, {})
#     save_yaml(ypath, ydoc)
#     setting_up = False

#     # save on change
#     def val_for_yaml(widget, section: str, key: str):
#         if section == "ExperimentSetup" and key == "wavelength":
#             txt = (widget.value or "").strip()
#             if txt.lower() in {"", "none", "null"}:
#                 return None
#             try:
#                 return float(txt)
#             except Exception:
#                 return txt
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

#     def on_changed(section: str, key: str, widget):
#         nonlocal ydoc
#         if setting_up:
#             return
#         ydoc.setdefault(section, {})
#         ydoc[section][key] = val_for_yaml(widget, section, key)
#         save_yaml(ypath, ydoc)

#     for section, mapping in W.items():
#         for key, widget in mapping.items():
#             widget.changed.connect(lambda *_, s=section, k=key, w=widget: on_changed(s, k, w))

#     # ---------- status helper ----------
#     def set_status(msg: str):
#         status_w.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ---------- actions ----------
#     state: Dict[str, Any] = dict(loader=None, builder=None, grid=None, edges=None)

#     def on_view_tiffs():
#         d = (tiff_dir_w.value or "").strip()
#         if not d or not os.path.isdir(d):
#             show_error("Please select a valid TIFF folder (column 1).")
#             return
#         open_tiffs_in_napari(d)
#         set_status("Opened TIFFs in napari.")

#     def on_load():
#         try:
#             spec = (spec_file_w.value or "").strip()
#             tdir = (tiff_dir_w.value or "").strip()
#             if not spec or not os.path.isfile(spec):
#                 raise FileNotFoundError("Select a valid SPEC file.")
#             if not tdir or not os.path.isdir(tdir):
#                 raise NotADirectoryError("Select a valid TIFF folder.")
#             scan_list = parse_scan_list(scans_w.value or "")
#             if not scan_list:
#                 raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#             set_status(f"Loading scans {scan_list}…")
#             # No setup YAML: pass None as setup file
#             print(ypath)
#             setup, UB, df = RSMDataLoader(
#                 spec,
#                 ypath,          # <-- no setup yaml
#                 tdir,
#                 selected_scans=scan_list,
#                 process_hklscan_only=bool(only_hkl_w.value),
#             ).load()
#             state["setup"] = setup
#             state["UB"] = UB
#             state["df"] = df
#             state["builder"] = None
#             state["grid"] = state["edges"] = None
#             set_status("Data loaded.")
#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_status(f"Load failed: {e}")

#     def on_build():
#         try:
#             if state["df"] is None:
#                 raise RuntimeError("Load data first.")
#             set_status("Computing Q/HKL/intensity…")
#             builder = RSMBuilder(
#                 state["setup"],
#                 state["UB"],
#                 state["df"],
#                 ub_includes_2pi=bool(ub_2pi_w.value),
#                 center_is_one_based=bool(center_one_based_w.value),
#             )
#             builder.compute_full(verbose=False)
#             state["builder"] = builder
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
#             # optional crop (fresh builder to avoid cumulative crops)
#             if bool(crop_enable_w.value):
#                 ymin, ymax = int(y_min_w.value), int(y_max_w.value)
#                 xmin, xmax = int(x_min_w.value), int(x_max_w.value)
#                 if ymin >= ymax or xmin >= xmax:
#                     raise ValueError("Crop bounds must satisfy y_min<y_max and x_min<x_max.")
#                 # rebuild from loader to avoid double crops
#                 if state["loader"] is None:
#                     raise RuntimeError("Internal error: loader missing; run Build again.")
#                 b = RSMBuilder(
#                     state["loader"],
#                     ub_includes_2pi=bool(ub_2pi_w.value),
#                     center_is_one_based=bool(center_one_based_w.value),
#                 )
#                 b.compute_full(verbose=False)
#                 b.crop_by_positions(y_bound=(ymin, ymax), x_bound=(xmin, xmax))

#             gx, gy, gz = parse_grid_shape(grid_shape_w.value)
#             if gx is None:
#                 raise ValueError("Grid X (first value) is required (e.g., 200,*,*).")
#             kw = dict(
#                 space=space_w.value,
#                 grid_shape=(gx, gy, gz),
#                 fuzzy=bool(fuzzy_w.value),
#                 normalize=normalize_w.value,
#                 stream=True,
#             )
#             if bool(fuzzy_w.value) and (fuzzy_width_w.value or 0) > 0:
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
#             lo = contrast_lo_w.value
#             hi = contrast_hi_w.value
#             if lo is None or hi is None:
#                 raise ValueError("Set contrast low/high percentages.")
#             lo = float(lo); hi = float(hi)
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
#             viz.launch()
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

#     # run
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()


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