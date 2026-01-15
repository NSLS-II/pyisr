#!/usr/bin/env python3
"""
RSM3D MagicGUI — 3 columns with vertical dividers:
  [1] Load Data  -> choose files/scans; editable ExperimentSetup; preview TIFFs
  [2] Build+Regrid -> compute RSM + regrid controls
  [3] View RSM   -> viewer controls + status

Requirements (example):
  pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Tuple

from qtpy.QtWidgets import QApplication, QFrame, QSizePolicy
import napari
from napari.utils.notifications import show_info, show_error

from magicgui.widgets import (
    Container, Label,
    FileEdit, TextEdit, LineEdit,
    CheckBox, ComboBox, FloatSpinBox, SpinBox, PushButton,
)

# Optional YAML for ExperimentSetup parsing
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from rsm3d.data_io import RSMDataLoader
from rsm3d.rsm3d     import RSMBuilder
from rsm3d.data_viz  import RSMNapariViewer


# ────────────────────────── small UI helpers ──────────────────────────
def HSeparator(height: int = 12):
    """A thin horizontal rule using QLabel's frame."""
    w = Label(value="")
    try:
        w.native.setFrameShape(QFrame.HLine)
        w.native.setFrameShadow(QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedHeight(height)
    except Exception:
        pass
    return w

def VSeparator(width: int = 2):
    """A thin vertical rule using QLabel's frame."""
    w = Label(value="")
    try:
        w.native.setFrameShape(QFrame.VLine)
        w.native.setFrameShadow(QFrame.Sunken)
        w.native.setLineWidth(1)
        w.native.setFixedWidth(width)
    except Exception:
        pass
    return w


# ────────────────────────── general helpers ──────────────────────────
def _parse_scan_list(text: str) -> List[int]:
    """Accepts: '17, 18-22,30' → [17,18,19,20,21,22,30]"""
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
    """
    'x,y,z' where y/z may be omitted or '*'.
    '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
    """
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
    """Open TIFF stack(s) in napari via glob patterns."""
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


# ────────────── dynamic ExperimentSetup editor (editable) ─────────────
def _widget_for_value(key: str, val: Any):
    """Return a magicgui widget appropriate for type(val)."""
    label = key
    if isinstance(val, bool):
        w = CheckBox(label=label, value=bool(val))
    elif isinstance(val, int):
        w = SpinBox(label=label, min=-10_000_000, max=10_000_000, step=1, value=int(val))
    elif isinstance(val, float):
        w = FloatSpinBox(label=label, min=-1e12, max=1e12, step=0.1, value=float(val))
    else:
        # serialize lists/dicts to YAML-ish text for edit
        if isinstance(val, (list, dict)):
            try:
                txt = yaml.safe_dump(val, sort_keys=False) if yaml else str(val)
            except Exception:
                txt = str(val)
            w = TextEdit(label=label, value=txt)
        else:
            w = LineEdit(label=label, value=str(val))
    return w

def _build_setup_editor(exp_setup: Dict[str, Any]) -> Tuple[Container, Dict[str, Any]]:
    """Build a vertical container of editable widgets for ExperimentSetup."""
    fields: Dict[str, Any] = {}
    widgets: List = [Label(value="ExperimentSetup (editable)"), HSeparator()]
    for k in sorted(exp_setup.keys()):
        w = _widget_for_value(k, exp_setup[k])
        widgets.append(w)
        fields[k] = exp_setup[k]
    col = Container(layout="vertical", widgets=widgets)
    return col, fields

def _collect_setup_overrides(editor_container: Container) -> Dict[str, Any]:
    """Read current values from editor_container widgets."""
    overrides: Dict[str, Any] = {}
    for w in editor_container:
        if isinstance(w, Label):  # skip labels/separators
            continue
        k = getattr(w, "label", None) or getattr(w, "name", None)
        if not k:
            continue
        if isinstance(w, CheckBox):
            overrides[k] = bool(w.value)
        elif isinstance(w, SpinBox):
            overrides[k] = int(w.value)
        elif isinstance(w, FloatSpinBox):
            overrides[k] = float(w.value)
        elif isinstance(w, LineEdit):
            overrides[k] = w.value
        elif isinstance(w, TextEdit):
            txt = w.value or ""
            if yaml:
                try:
                    parsed = yaml.safe_load(txt)
                    overrides[k] = parsed
                except Exception:
                    overrides[k] = txt
            else:
                overrides[k] = txt
        else:
            overrides[k] = w.value
    return overrides

def _load_experiment_setup(setup_file: str) -> Dict[str, Any]:
    """Return ExperimentSetup dict (or empty) from YAML file."""
    if not setup_file or not os.path.isfile(setup_file) or yaml is None:
        return {}
    try:
        data = yaml.safe_load(open(setup_file, "r", encoding="utf-8"))
        if isinstance(data, dict):
            exp = data.get("ExperimentSetup") or data.get("experiment_setup")
            if isinstance(exp, dict):
                return exp
    except Exception:
        pass
    return {}

def _write_edited_setup_yaml(original_yaml: str, overrides: Dict[str, Any]) -> str:
    """Write a temp YAML with ExperimentSetup merged with overrides; return path."""
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
        exp.update(overrides)
        base["ExperimentSetup"] = exp
        tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
        out_path = os.path.join(tmpdir, "edited_setup.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(base, f, sort_keys=False)
        return out_path
    except Exception as e:
        show_error(f"Failed to write edited setup YAML: {e}")
        return original_yaml


# ────────────────────────── the App (3 columns) ──────────────────────────
def main():
    app = QApplication.instance() or QApplication(sys.argv)

    # ── Column 1: Load Data + editable ExperimentSetup
    c1_title = Label(value="① Load Data")
    spec_file = FileEdit(mode="r", label="SPEC file")
    setup_file = FileEdit(mode="r", label="YAML setup (optional)")
    tiff_dir = FileEdit(mode="d", label="TIFF folder")
    scans = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
    only_hkl = CheckBox(label="Only HKL scans", value=True)
    btn_load = PushButton(text="Load Data")
    btn_tiff = PushButton(text="View TIFFs in napari")

    exp_editor_placeholder = Container(
        layout="vertical",
        widgets=[HSeparator(100), Label(value="ExperimentSetup editor appears here after Load")],
    )

    col1 = Container(
        layout="vertical",
        widgets=[
            c1_title, HSeparator(),
            spec_file, setup_file, tiff_dir, scans, only_hkl,
            btn_load, btn_tiff,
            HSeparator(), exp_editor_placeholder,
        ],
    )

    # ── Column 2: Build + Regrid (combined)
    c2_title = Label(value="② Build RSM Map  &  ③ Regrid")
    # Build
    ub_2pi = CheckBox(label="UB includes 2π", value=True)
    center_one_based = CheckBox(label="1-based center", value=False)
    btn_build = PushButton(text="Build RSM Map")
    # Regrid
    space = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
    grid_shape = LineEdit(label="Grid (x,y,z)", value="200,*,*")
    fuzzy = CheckBox(label="Fuzzy gridder", value=True)
    fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
    normalize = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
    btn_regrid = PushButton(text="Regrid")

    col2 = Container(
        layout="vertical",
        widgets=[
            c2_title, HSeparator(),
            Label(value="Build options"), ub_2pi, center_one_based, btn_build,
            HSeparator(),
            Label(value="Regrid options"), space, grid_shape, fuzzy, fuzzy_width, normalize, btn_regrid,
        ],
    )

    # ── Column 3: View RSM
    c3_title = Label(value="④ View RSM")
    log_view = CheckBox(label="Log view", value=True)
    cmap = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
    rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
    contrast = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
    btn_view = PushButton(text="View RSM")
    status = TextEdit(label="Status", value="")
    try:
        status.native.setReadOnly(True)
        status.native.setMinimumHeight(120)
    except Exception:
        pass

    col3 = Container(
        layout="vertical",
        widgets=[c3_title, HSeparator(), log_view, cmap, rendering, contrast, btn_view, HSeparator(), status],
    )

    # ── Main root: 3 columns with vertical dividers between them
    vline1 = VSeparator()
    vline2 = VSeparator()
    root = Container(layout="horizontal", widgets=[col1, vline1, col2, vline2, col3])

    # Make columns expand evenly
    for col in (col1, col2, col3):
        try:
            col.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
    try:
        lay = root.native.layout()
        # child order: 0 col1, 1 vline1, 2 col2, 3 vline2, 4 col3
        lay.setStretch(0, 1)
        lay.setStretch(1, 0)
        lay.setStretch(2, 1)
        lay.setStretch(3, 0)
        lay.setStretch(4, 1)
    except Exception:
        pass

    # ── App state
    state: Dict[str, Any] = dict(
        loader=None,               # RSMDataLoader
        builder=None,              # RSMBuilder
        Q=None, hkl=None, intensity=None,
        grid=None, edges=None,
        edited_setup_path=None,    # path to temp edited YAML (if any)
        exp_editor_container=exp_editor_placeholder,
        exp_editor_loaded=False,
    )

    # ── status helper
    def set_status(msg: str):
        status.value = msg
        try:
            show_info(msg)
        except Exception:
            pass

    # ── button handlers
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
            state["loader"] = loader
            state["builder"] = None
            state["Q"] = state["hkl"] = state["intensity"] = None
            state["grid"] = state["edges"] = None
            state["edited_setup_path"] = None

            # Build editable ExperimentSetup UI
            exp_dict = _load_experiment_setup(setup_file.value) if setup_file.value else {}
            state["exp_editor_container"].widgets = []  # clear area
            if exp_dict:
                editor_col, _snapshot = _build_setup_editor(exp_dict)
                state["exp_editor_container"].extend([editor_col])
                state["exp_editor_loaded"] = True
                set_status("Data loaded. ExperimentSetup ready to edit.")
            else:
                state["exp_editor_container"].extend([Label(value="(No ExperimentSetup found in YAML)")])
                state["exp_editor_loaded"] = False
                set_status("Data loaded. No ExperimentSetup found in YAML.")
        except Exception as e:
            show_error(f"Load error: {e}")
            set_status(f"Load failed: {e}")

    def on_build():
        try:
            if state["loader"] is None:
                raise RuntimeError("Load data first.")
            # If user edited ExperimentSetup, write a temp YAML and rebuild loader
            edited_yaml = setup_file.value
            if state["exp_editor_loaded"]:
                editor_cols = [w for w in state["exp_editor_container"] if isinstance(w, Container)]
                if editor_cols:
                    overrides = _collect_setup_overrides(editor_cols[0])
                    edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
                    state["edited_setup_path"] = edited_yaml

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

            state["loader"]   = loader2
            state["builder"]  = builder
            state["Q"]        = Q_samp
            state["hkl"]      = hkl_arr
            state["intensity"]= intensity_arr
            state["grid"]     = None
            state["edges"]    = None

            set_status("RSM map built.")
        except Exception as e:
            show_error(f"Build error: {e}")
            set_status(f"Build failed: {e}")

    def on_regrid():
        try:
            builder = state["builder"]
            if builder is None:
                raise RuntimeError("Build the RSM map first.")
            gx, gy, gz = _parse_grid_shape(grid_shape.value)
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
            grid, edges = builder.regrid_xu(**kw)
            state["grid"], state["edges"] = grid, edges
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

    # show window
    try:
        root.native.setWindowTitle("RSM3D Pipeline (3 columns)")
        root.native.resize(1280, 720)
    except Exception:
        pass

    root.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# RSM3D MagicGUI with 4-column layout:
#   [1] Load Data      -> choose files/scans, preview & EDIT ExperimentSetup
#   [2] Build RSM Map  -> compute Q/HKL/intensity using edited ExperimentSetup
#   [3] Regrid         -> build regular grid (HKL or Q)
#   [4] View RSM       -> launch RSMNapariViewer; status lives here too

# Also includes "View TIFFs in napari" (column 1).

# Requirements (example):
#   pip install napari magicgui qtpy xrayutilities pyyaml   # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import re
# import sys
# import tempfile
# from typing import Any, Dict, List, Tuple

# from qtpy.QtWidgets import QApplication
# from qtpy.QtWidgets import QFrame
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


# # ------------------------ helpers ------------------------

# def HSeparator():
#     sep = Label(value="")
#     try:
#         sep.native.setFrameShape(QFrame.HLine)   # draw a line
#         sep.native.setFrameShadow(QFrame.Sunken)
#         sep.native.setLineWidth(1)
#         sep.native.setFixedHeight(12)
#     except Exception:
#         pass
#     return sep


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


# # ------------------------ dynamic ExperimentSetup editor ------------------------

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
#     """
#     Build a column of editable widgets for ExperimentSetup.
#     Returns (container, snapshot_dict) where snapshot_dict mirrors the initial values.
#     """
#     fields: Dict[str, Any] = {}
#     widgets: List = []

#     # Stable ordering
#     for k in sorted(exp_setup.keys()):
#         w = _widget_for_value(k, exp_setup[k])
#         widgets.append(w)
#         fields[k] = exp_setup[k]

#     col = Container(layout="vertical", widgets=[Label(value="ExperimentSetup (editable)")] + widgets)
#     return col, fields


# def _collect_setup_overrides(editor_container: Container) -> Dict[str, Any]:
#     """
#     Read current values from editor_container and produce a dict suitable
#     to merge into YAML['ExperimentSetup'].
#     """
#     overrides: Dict[str, Any] = {}
#     for w in editor_container:
#         # skip the title Label
#         if isinstance(w, Label):
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
#             # try to parse YAML for lists/dicts
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
#     if not setup_file or not os.path.isfile(setup_file):
#         return {}
#     if not yaml:
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
#     """Write a temp YAML with ExperimentSetup replaced/merged with overrides; return path."""
#     if yaml is None:
#         # If YAML not available, just return original (edits won't apply)
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
#         # Prefer canonical key
#         base["ExperimentSetup"] = exp

#         tmpdir = tempfile.mkdtemp(prefix="rsm_setup_")
#         out_path = os.path.join(tmpdir, "edited_setup.yaml")
#         with open(out_path, "w", encoding="utf-8") as f:
#             yaml.safe_dump(base, f, sort_keys=False)
#         return out_path
#     except Exception as e:
#         show_error(f"Failed to write edited setup YAML: {e}")
#         return original_yaml


# # ------------------------ App: 4 columns ------------------------

# def main():
#     app = QApplication.instance() or QApplication(sys.argv)

#     # ----- Column 1: Load Data -----
#     title1 = Label(value="① Load Data")
#     spec_file = FileEdit(mode="r", label="SPEC file")
#     setup_file = FileEdit(mode="r", label="YAML setup (optional)")
#     tiff_dir = FileEdit(mode="d", label="TIFF folder")
#     scans = LineEdit(label="Scans (e.g. 17, 18-22, 30)")
#     only_hkl = CheckBox(label="Only HKL scans", value=True)
#     btn_load = PushButton(text="Load Data")
#     btn_tiff = PushButton(text="View TIFFs in napari")

#     # placeholder for dynamic ExperimentSetup editor
#     exp_editor_title = Label(value="ExperimentSetup editor appears here after Load")
#     exp_editor = Container(layout="vertical", widgets=[exp_editor_title])

#     col1 = Container(
#         layout="vertical",
#         widgets=[
#             title1, HSeparator(),
#             spec_file, setup_file, tiff_dir, scans, only_hkl,
#             btn_load, btn_tiff,
#             HSeparator(), exp_editor,
#         ],
#     )

#     # ----- Column 2: Build RSM Map -----
#     title2 = Label(value="② Build RSM Map")
#     ub_2pi = CheckBox(label="UB includes 2π", value=True)
#     center_one_based = CheckBox(label="1-based center", value=False)
#     btn_build = PushButton(text="Build RSM Map")

#     col2 = Container(
#         layout="vertical",
#         widgets=[title2, HSeparator(), ub_2pi, center_one_based, btn_build],
#     )

#     # ----- Column 3: Regrid -----
#     title3 = Label(value="③ Regrid")
#     space = ComboBox(label="Space", choices=["hkl", "q"], value="hkl")
#     grid_shape = LineEdit(label="Grid (x,y,z)", value="200,*,*")
#     fuzzy = CheckBox(label="Fuzzy gridder", value=True)
#     fuzzy_width = FloatSpinBox(label="Width (fuzzy)", min=0.0, max=5.0, step=0.1, value=0.0)
#     normalize = ComboBox(label="Normalize", choices=["mean", "sum"], value="mean")
#     btn_regrid = PushButton(text="Regrid")

#     col3 = Container(
#         layout="vertical",
#         widgets=[
#             title3, HSeparator(), space, grid_shape,
#             fuzzy, fuzzy_width, normalize,
#             btn_regrid,
#         ],
#     )

#     # ----- Column 4: View RSM -----
#     title4 = Label(value="④ View RSM")
#     log_view = CheckBox(label="Log view", value=True)
#     cmap = ComboBox(label="Colormap", choices=["viridis", "inferno", "magma", "plasma", "cividis"], value="inferno")
#     rendering = ComboBox(label="Rendering", choices=["attenuated_mip", "mip", "translucent"], value="attenuated_mip")
#     contrast = LineEdit(label="Contrast % (lo,hi)", value="1,99.8")
#     status = TextEdit(label="Status", value="")
#     btn_view = PushButton(text="View RSM")

#     # make status nicer
#     try:
#         status.native.setReadOnly(True)
#         status.native.setMinimumHeight(120)
#     except Exception:
#         pass

#     col4 = Container(
#         layout="vertical",
#         widgets=[title4, HSeparator(), log_view, cmap, rendering, contrast, btn_view, HSeparator(), status],
#     )

#     # ----- Main 4-column container -----
#     root = Container(layout="horizontal", widgets=[col1, col2, col3, col4])

#     # ----- App state -----
#     state: Dict[str, Any] = dict(
#         loader=None,               # RSMDataLoader
#         builder=None,              # RSMBuilder
#         Q=None, hkl=None, intensity=None,
#         grid=None, edges=None,
#         edited_setup_path=None,    # path to temp edited YAML
#         exp_editor_container=exp_editor,
#         exp_editor_loaded=False,
#     )

#     # ----- helpers for status -----
#     def set_status(msg: str):
#         status.value = msg
#         try:
#             show_info(msg)
#         except Exception:
#             pass

#     # ----- button handlers -----
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

#             # Build/editable ExperimentSetup UI
#             exp_dict = _load_experiment_setup(setup_file.value) if setup_file.value else {}
#             # Clear old editor content
#             state["exp_editor_container"].widgets = []
#             if exp_dict:
#                 editor_col, _snapshot = _build_setup_editor(exp_dict)
#                 # Embed the generated widgets into the placeholder container
#                 state["exp_editor_container"].extend([editor_col])
#                 state["exp_editor_loaded"] = True
#                 set_status("Data loaded. ExperimentSetup ready to edit.")
#             else:
#                 state["exp_editor_container"].extend([Label(value="(No ExperimentSetup in YAML)")])
#                 state["exp_editor_loaded"] = False
#                 set_status("Data loaded. No ExperimentSetup found in YAML.")

#         except Exception as e:
#             show_error(f"Load error: {e}")
#             set_status(f"Load failed: {e}")

#     def on_build():
#         try:
#             if state["loader"] is None:
#                 raise RuntimeError("Load data first.")
#             # If user edited ExperimentSetup, materialize to a temp YAML and rebuild loader
#             edited_yaml = setup_file.value
#             if state["exp_editor_loaded"]:
#                 # We embedded a Container inside exp_editor_container
#                 # The first child is the editor column with all parameter widgets.
#                 editor_cols = [w for w in state["exp_editor_container"] if isinstance(w, Container)]
#                 if editor_cols:
#                     overrides = _collect_setup_overrides(editor_cols[0])
#                     edited_yaml = _write_edited_setup_yaml(setup_file.value, overrides)
#                     state["edited_setup_path"] = edited_yaml

#             # Rebuild loader so builder uses edited YAML
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

#     # ----- connect buttons -----
#     btn_tiff.clicked.connect(on_view_tiffs)
#     btn_load.clicked.connect(on_load)
#     btn_build.clicked.connect(on_build)
#     btn_regrid.clicked.connect(on_regrid)
#     btn_view.clicked.connect(on_view)

#     # ----- show UI and start loop -----
#     try:
#         # prettify window
#         root.native.setWindowTitle("RSM3D Pipeline (4 columns)")
#         root.native.resize(1280, 720)
#     except Exception:
#         pass

#     root.show()
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# RSM3D MagicGUI pipeline with explicit steps:
#   1) Load Data      -> reads SPEC/YAML/TIFF; shows ExperimentSetup params
#   2) Build RSM Map  -> computes Q_samp / HKL / intensity
#   3) Regrid         -> builds regular grid in HKL or Q
#   4) View RSM       -> launches RSMNapariViewer

# Also includes a button to preview the raw TIFF frames in napari.

# Requirements (example):
#   pip install napari magicgui qtpy xrayutilities  # plus your rsm3d package
# """

# from __future__ import annotations

# import os
# import re
# import sys
# from typing import List, Tuple

# from magicgui import magicgui
# from magicgui.widgets import (
#     FileEdit, TextEdit, CheckBox, ComboBox, FloatSpinBox, PushButton
# )
# from qtpy.QtWidgets import QApplication
# import napari
# from napari.utils.notifications import show_info, show_error

# # Optional YAML parsing for ExperimentSetup display
# try:
#     import yaml  # type: ignore
# except Exception:  # pragma: no cover
#     yaml = None  # fallback to raw preview

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ------------------------ helpers ------------------------

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
#     Examples: '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
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


# def _format_experiment_setup(setup_file: str, loader: RSMDataLoader | None) -> str:
#     """Pretty-print ExperimentSetup after load."""
#     # 1) From YAML file
#     try:
#         if setup_file and os.path.isfile(setup_file):
#             if yaml is not None:
#                 data = yaml.safe_load(open(setup_file, "r", encoding="utf-8"))
#                 if isinstance(data, dict):
#                     exp = data.get("ExperimentSetup") or data.get("experiment_setup")
#                     if isinstance(exp, dict):
#                         lines = ["ExperimentSetup:"]
#                         for k, v in exp.items():
#                             lines.append(f"  {k}: {v}")
#                         return "\n".join(lines)
#                     keys = ", ".join(sorted(data.keys()))
#                     return f"YAML keys: {keys}"
#             with open(setup_file, "r", encoding="utf-8", errors="ignore") as f:
#                 txt = f.read(2000)
#             return f"(YAML preview, first 2000 chars)\n{txt}"
#     except Exception as e:
#         return f"(Could not parse YAML) {e}"

#     # 2) From loader attrs
#     try:
#         if loader is not None:
#             for attr in ("experiment_setup", "setup", "instrument", "config"):
#                 if hasattr(loader, attr):
#                     obj = getattr(loader, attr)
#                     if isinstance(obj, dict):
#                         lines = [f"{attr}:"]
#                         for k, v in obj.items():
#                             lines.append(f"  {k}: {v}")
#                         return "\n".join(lines)
#                     if hasattr(obj, "__dict__"):
#                         d = vars(obj)
#                         lines = [f"{attr}:"]
#                         for k, v in d.items():
#                             lines.append(f"  {k}: {v}")
#                         return "\n".join(lines)
#     except Exception:
#         pass

#     return "No ExperimentSetup information found."


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


# # ------------------------ UI (magicgui) ------------------------

# @magicgui(
#     spec_file     = {"widget_type": FileEdit, "mode": "r", "label": "SPEC file"},
#     setup_file    = {"widget_type": FileEdit, "mode": "r", "label": "YAML setup (optional)"},
#     tiff_dir      = {"widget_type": FileEdit, "mode": "d", "label": "TIFF folder"},
#     selected_scans= {"widget_type": TextEdit, "label": "Scans (e.g. 17, 18-22, 30)"},
#     process_hklscan_only={"widget_type": CheckBox, "label": "Only HKL scans", "value": True},
#     ub_includes_2pi    ={"widget_type": CheckBox, "label": "UB includes 2π", "value": True},
#     center_is_one_based={"widget_type": CheckBox, "label": "1-based center", "value": False},

#     space         = {"widget_type": ComboBox, "label": "Space", "choices": ["hkl", "q"], "value": "hkl"},
#     grid_shape    = {"widget_type": TextEdit, "label": "Grid (x,y,z)", "value": "200,*,*"},
#     fuzzy         = {"widget_type": CheckBox, "label": "Fuzzy gridder", "value": True},
#     fuzzy_width   = {"widget_type": FloatSpinBox, "label": "Width (fuzzy)", "min": 0.0, "max": 5.0, "step": 0.1, "value": 0.0},
#     normalize     = {"widget_type": ComboBox, "label": "Normalize", "choices": ["mean", "sum"], "value": "mean"},

#     log_view      = {"widget_type": CheckBox, "label": "Log view", "value": True},
#     cmap          = {"widget_type": ComboBox, "label": "Colormap", "choices": ["viridis", "inferno", "magma", "plasma", "cividis"], "value": "inferno"},
#     rendering     = {"widget_type": ComboBox, "label": "Rendering", "choices": ["attenuated_mip", "mip", "translucent"], "value": "attenuated_mip"},
#     contrast_pct  = {"widget_type": TextEdit, "label": "Contrast % (lo,hi)", "value": "1,99.8"},

#     # Pipeline buttons
#     load_button   = {"widget_type": PushButton, "text": "Load Data"},
#     build_button  = {"widget_type": PushButton, "text": "Build RSM Map"},
#     regrid_button = {"widget_type": PushButton, "text": "Regrid"},
#     view_button   = {"widget_type": PushButton, "text": "View RSM"},
#     tiff_button   = {"widget_type": PushButton, "text": "View TIFFs in napari"},

#     # Display areas (make read-only later via .native)
#     setup_info    = {"widget_type": TextEdit, "label": "ExperimentSetup", "value": ""},
#     status_text   = {"widget_type": TextEdit, "label": "Status", "value": ""},
# )
# def rsm_gui(
#     spec_file,
#     setup_file,
#     tiff_dir,
#     selected_scans,
#     process_hklscan_only,
#     ub_includes_2pi,
#     center_is_one_based,
#     space,
#     grid_shape,
#     fuzzy,
#     fuzzy_width,
#     normalize,
#     log_view,
#     cmap,
#     rendering,
#     contrast_pct,
#     load_button,
#     build_button,
#     regrid_button,
#     view_button,
#     tiff_button,
#     setup_info,
#     status_text,
# ):
#     """Use the buttons to run each step. Results cached on the widget object."""
#     return None


# # ------------------------ wiring (callbacks + state) ------------------------

# def _init_state(w):
#     w._state = dict(
#         loader=None,    # RSMDataLoader
#         builder=None,   # RSMBuilder
#         Q=None, hkl=None, intensity=None,
#         grid=None, edges=None,
#     )
#     # Button enablement
#     w.build_button.enabled  = False
#     w.regrid_button.enabled = False
#     w.view_button.enabled   = False

#     # Make info/status read-only via Qt
#     try:
#         w.setup_info.native.setReadOnly(True)
#         w.status_text.native.setReadOnly(True)
#     except Exception:
#         pass


# def _set_status(w, msg: str):
#     w.status_text.value = msg
#     try:
#         show_info(msg)
#     except Exception:
#         pass


# def _format_and_show_setup(w, loader: RSMDataLoader | None):
#     w.setup_info.value = _format_experiment_setup(w.setup_file.value, loader)


# def _on_load(w):
#     try:
#         # Validate
#         if not w.spec_file.value or not os.path.isfile(w.spec_file.value):
#             raise FileNotFoundError("Please select a valid SPEC file.")
#         if w.setup_file.value and not os.path.isfile(w.setup_file.value):
#             raise FileNotFoundError("YAML setup path does not exist.")
#         if not w.tiff_dir.value or not os.path.isdir(w.tiff_dir.value):
#             raise NotADirectoryError("Please select a valid TIFF folder.")
#         scans = _parse_scan_list(w.selected_scans.value)
#         if not scans:
#             raise ValueError("Enter at least one scan (e.g. '17, 18-22').")

#         # Load
#         _set_status(w, f"Loading scans {scans}…")
#         loader = RSMDataLoader(
#             w.spec_file.value,
#             w.setup_file.value,
#             w.tiff_dir.value,
#             selected_scans=scans,
#             process_hklscan_only=bool(w.process_hklscan_only.value),
#         )
#         loader.load()

#         # Show ExperimentSetup
#         _format_and_show_setup(w, loader)

#         # Update state + buttons
#         st = w._state
#         st["loader"] = loader
#         st["builder"] = None
#         st["Q"] = st["hkl"] = st["intensity"] = None
#         st["grid"] = st["edges"] = None

#         w.build_button.enabled  = True
#         w.regrid_button.enabled = False
#         w.view_button.enabled   = False

#         _set_status(w, "Data loaded.")
#     except Exception as e:
#         show_error(f"Load error: {e}")
#         _set_status(w, f"Load failed: {e}")


# def _on_build(w):
#     try:
#         st = w._state
#         loader = st.get("loader")
#         if loader is None:
#             raise RuntimeError("Load data first.")

#         _set_status(w, "Computing Q/HKL/intensity…")
#         builder = RSMBuilder(
#             loader,
#             ub_includes_2pi=bool(w.ub_includes_2pi.value),
#             center_is_one_based=bool(w.center_is_one_based.value),
#         )
#         Q_samp, hkl, intensity = builder.compute_full(verbose=False)

#         st["builder"]   = builder
#         st["Q"]         = Q_samp
#         st["hkl"]       = hkl
#         st["intensity"] = intensity
#         st["grid"]      = None
#         st["edges"]     = None

#         w.regrid_button.enabled = True
#         w.view_button.enabled   = False

#         _set_status(w, "RSM map built.")
#     except Exception as e:
#         show_error(f"Build error: {e}")
#         _set_status(w, f"Build failed: {e}")


# def _on_regrid(w):
#     try:
#         st = w._state
#         builder = st.get("builder")
#         if builder is None:
#             raise RuntimeError("Build the RSM map first.")

#         gx, gy, gz = _parse_grid_shape(w.grid_shape.value)
#         try:
#             lo_str, hi_str = [p.strip() for p in w.contrast_pct.value.split(",")]
#             pct_lo, pct_hi = float(lo_str), float(hi_str)
#             if not (0 <= pct_lo < pct_hi <= 100):
#                 raise ValueError
#         except Exception:
#             raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

#         kw = dict(
#             space=w.space.value,
#             grid_shape=(gx, gy, gz),
#             fuzzy=bool(w.fuzzy.value),
#             normalize=w.normalize.value,
#             stream=True,
#         )
#         if w.fuzzy.value and float(w.fuzzy_width.value) > 0:
#             kw["width"] = float(w.fuzzy_width.value)

#         _set_status(w, f"Regridding to {w.space.value.upper()} grid {(gx, gy, gz)}…")
#         grid, edges = builder.regrid_xu(**kw)

#         st["grid"]  = grid
#         st["edges"] = edges

#         w.view_button.enabled = True
#         _set_status(w, "Regrid completed.")
#     except Exception as e:
#         show_error(f"Regrid error: {e}")
#         _set_status(w, f"Regrid failed: {e}")


# def _on_view_rsm(w):
#     try:
#         st = w._state
#         if st.get("grid") is None or st.get("edges") is None:
#             raise RuntimeError("Regrid first.")

#         lo_str, hi_str = [p.strip() for p in w.contrast_pct.value.split(",")]
#         pct_lo, pct_hi = float(lo_str), float(hi_str)

#         viz = RSMNapariViewer(
#             st["grid"],
#             st["edges"],
#             space=w.space.value,
#             name="RSM3D",
#             log_view=bool(w.log_view.value),
#             contrast_percentiles=(pct_lo, pct_hi),
#             cmap=w.cmap.value,
#             rendering=w.rendering.value,
#         )
#         viz.launch()  # returns a napari.Viewer
#         _set_status(w, "RSM viewer opened.")
#     except Exception as e:
#         show_error(f"View error: {e}")
#         _set_status(w, f"View failed: {e}")


# def _on_view_tiffs(w):
#     try:
#         tiff_dir = w.tiff_dir.value
#         if not tiff_dir or not os.path.isdir(tiff_dir):
#             raise NotADirectoryError("Please select a valid TIFF folder first.")
#         _open_tiffs_in_napari(tiff_dir)
#         _set_status(w, "Opened TIFFs in napari.")
#     except Exception as e:
#         show_error(f"TIFF preview error: {e}")
#         _set_status(w, f"TIFF preview failed: {e}")


# # ------------------------ script entrypoint ------------------------

# def main():
#     app = QApplication.instance() or QApplication(sys.argv)

#     w = rsm_gui
#     # Make info/status read-only (constructor doesn’t accept read_only)
#     try:
#         w.setup_info.native.setReadOnly(True)
#         w.status_text.native.setReadOnly(True)
#     except Exception:
#         pass

#     # Initialize state + buttons
#     _init_state(w)

#     # Window tweak
#     try:
#         w.native.setWindowTitle("RSM3D Pipeline (Napari + MagicGUI)")
#         w.native.resize(560, 720)
#     except Exception:
#         pass

#     # Bind buttons
#     w.load_button.clicked.connect(lambda: _on_load(w))
#     w.build_button.clicked.connect(lambda: _on_build(w))
#     w.regrid_button.clicked.connect(lambda: _on_regrid(w))
#     w.view_button.clicked.connect(lambda: _on_view_rsm(w))
#     w.tiff_button.clicked.connect(lambda: _on_view_tiffs(w))

#     # Initial status
#     _set_status(w, "Ready. Choose files, then click 'Load Data'.")

#     # Show UI and start loop
#     w.show()
#     sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()


#-------------------------
#First runable napari magicgui example for RSM3D

# #!/usr/bin/env python3
# """
# RSM3D MagicGUI launcher (runs as a plain .py)

# Quick start:
#   pip install napari magicgui qtpy xrayutilities
#   # plus your rsm3d package (and its deps)

# Run:
#   python rsm_magicgui.py
# """

# from __future__ import annotations

# import os
# import re
# import sys
# import numpy as np

# from magicgui import magicgui
# from magicgui.widgets import (
#     FileEdit, TextEdit, CheckBox, ComboBox, FloatSpinBox, PushButton
# )

# from qtpy.QtWidgets import QApplication
# from napari.utils.notifications import show_info, show_error

# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz  import RSMNapariViewer


# # ------------------------ helpers ------------------------

# def _parse_scan_list(text: str) -> list[int]:
#     """
#     Accepts formats like: "17, 18-22,30" → [17,18,19,20,21,22,30]
#     """
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


# def _parse_grid_shape(text: str) -> tuple[int | None, int | None, int | None]:
#     """
#     'x,y,z' where y/z may be omitted or '*'.
#     Examples: '200,*,*' → (200, None, None), '256,256,256' → (256,256,256), '200' → (200,None,None)
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


# # ------------------------ UI ------------------------

# @magicgui(
#     spec_file     = {"widget_type": FileEdit, "mode": "r", "label": "SPEC file", "tooltip": "Path to SPEC file"},
#     setup_file    = {"widget_type": FileEdit, "mode": "r", "label": "YAML setup (optional)", "tooltip": "Instrument YAML (optional)"},
#     tiff_dir      = {"widget_type": FileEdit, "mode": "d", "label": "TIFF folder", "tooltip": "Directory of detector frames"},
#     selected_scans= {"widget_type": TextEdit, "label": "Scans (e.g. 17, 18-22, 30)"},
#     process_hklscan_only={"widget_type": CheckBox, "label": "Only HKL scans", "value": True},
#     ub_includes_2pi    ={"widget_type": CheckBox, "label": "UB includes 2π", "value": True, "tooltip": "Turn OFF if UB excludes 2π"},
#     center_is_one_based={"widget_type": CheckBox, "label": "1-based center", "value": False, "tooltip": "SPEC xcenter/ycenter are 1-based"},

#     space         = {"widget_type": ComboBox, "label": "Space", "choices": ["hkl", "q"], "value": "hkl", "tooltip": "Regrid space"},
#     grid_shape    = {"widget_type": TextEdit, "label": "Grid (x,y,z)", "value": "200,*,*", "tooltip": "x required; y/z '*'=auto"},
#     fuzzy         = {"widget_type": CheckBox, "label": "Fuzzy gridder", "value": True},
#     fuzzy_width   = {"widget_type": FloatSpinBox, "label": "Width (fuzzy)", "min": 0.0, "max": 5.0, "step": 0.1, "value": 0.0,
#                      "tooltip": "Optional kernel width; 0 = auto"},
#     normalize     = {"widget_type": ComboBox, "label": "Normalize", "choices": ["mean", "sum"], "value": "mean"},

#     log_view      = {"widget_type": CheckBox, "label": "Log view", "value": True},
#     cmap          = {"widget_type": ComboBox, "label": "Colormap", "choices": ["viridis", "inferno", "magma", "plasma", "cividis"], "value": "inferno"},
#     rendering     = {"widget_type": ComboBox, "label": "Rendering", "choices": ["attenuated_mip", "mip", "translucent"], "value": "attenuated_mip"},
#     contrast_pct  = {"widget_type": TextEdit, "label": "Contrast % (lo,hi)", "value": "1,99.8", "tooltip": "percentiles like '1,99.8'"},
#     run_button    = {"widget_type": PushButton, "text": "Run RSM3D"},
# )
# def run_rsm(
#     spec_file,
#     setup_file,
#     tiff_dir,
#     selected_scans,
#     process_hklscan_only,
#     ub_includes_2pi,
#     center_is_one_based,
#     space,
#     grid_shape,
#     fuzzy,
#     fuzzy_width,
#     normalize,
#     log_view,
#     cmap,
#     rendering,
#     contrast_pct,
#     run_button,  # trigger only
# ):
#     """
#     1) Load SPEC + TIFF
#     2) Build Q, HKL, intensity
#     3) Regrid to HKL/Q space using xrayutilities
#     4) Launch Napari viewer
#     """
#     try:
#         # --- Validate ---
#         if not spec_file or not os.path.isfile(spec_file):
#             raise FileNotFoundError("Please select a valid SPEC file.")
#         if setup_file and not os.path.isfile(setup_file):
#             raise FileNotFoundError("YAML setup path does not exist.")
#         if not tiff_dir or not os.path.isdir(tiff_dir):
#             raise NotADirectoryError("Please select a valid TIFF folder.")

#         scans = _parse_scan_list(selected_scans)
#         if not scans:
#             raise ValueError("Please enter at least one scan (e.g. '17, 18-22').")

#         gx, gy, gz = _parse_grid_shape(grid_shape)

#         try:
#             lo_str, hi_str = [p.strip() for p in contrast_pct.split(",")]
#             pct_lo, pct_hi = float(lo_str), float(hi_str)
#             if not (0 <= pct_lo < pct_hi <= 100):
#                 raise ValueError
#         except Exception:
#             raise ValueError("Contrast percentiles must be like '1,99.8' with 0<=lo<hi<=100")

#         # --- Load ---
#         loader = RSMDataLoader(
#             spec_file,
#             setup_file,
#             tiff_dir,
#             selected_scans=scans,
#             process_hklscan_only=bool(process_hklscan_only),
#         )
#         show_info(f"Loading scans {scans}…")
#         loader.load()

#         # --- Build ---
#         builder = RSMBuilder(
#             loader,
#             ub_includes_2pi=bool(ub_includes_2pi),
#             center_is_one_based=bool(center_is_one_based),
#         )
#         show_info("Computing Q/HKL/intensity…")
#         Q_samp, hkl, intensity = builder.compute_full(verbose=False)

#         # --- Regrid ---
#         shape = (gx, gy, gz)
#         kw = dict(
#             space=space,
#             grid_shape=shape,
#             fuzzy=bool(fuzzy),
#             normalize=normalize,
#             stream=True,
#         )
#         if fuzzy and (fuzzy_width is not None) and (float(fuzzy_width) > 0):
#             kw["width"] = float(fuzzy_width)

#         show_info(f"Regridding to {space.upper()} grid {shape}…")
#         grid, edges = builder.regrid_xu(**kw)

#         # --- View ---
#         viz = RSMNapariViewer(
#             grid, edges,
#             space=space,
#             name="RSM3D",
#             log_view=bool(log_view),
#             contrast_percentiles=(pct_lo, pct_hi),
#             cmap=cmap,
#             rendering=rendering,
#         )
#         viewer = viz.launch()
#         show_info("RSM viewing ready.")
#         return viewer

#     except Exception as e:
#         show_error(f"RSM3D error: {e}")
#         raise


# # ------------------------ script entrypoint ------------------------

# def main():
#     # Create (or reuse) a Qt app and start the loop explicitly.
#     app = QApplication.instance() or QApplication(sys.argv)

#     w = run_rsm
#     # Optional: tweak window title/size
#     try:
#         w.native.setWindowTitle("RSM3D Builder (Napari)")
#         w.native.resize(520, 580)
#     except Exception:
#         pass

#     w.show()

#     print("[RSM] Qt event loop starting… (close all windows to exit)")
#     exit_code = app.exec_()
#     print("[RSM] Qt event loop finished.")
#     sys.exit(exit_code)


# if __name__ == "__main__":
#     main()


# from magicgui import magicgui
# from magicgui.widgets import FileEdit, TextEdit, CheckBox, PushButton
# import numpy as np
# from napari.utils import gui_qt
# from rsm3d.data_io import RSMDataLoader
# from rsm3d.rsm3d     import RSMBuilder
# from rsm3d.data_viz   import RSMNapariViewer

# @magicgui(
#     spec_file={    "widget_type": FileEdit, "mode": "r", "label": "SPEC file"   },
#     setup_file={   "widget_type": FileEdit, "mode": "r", "label": "YAML setup" },
#     tiff_dir={     "widget_type": FileEdit, "mode": "d", "label": "TIFF folder" },
#     selected_scans={"widget_type": TextEdit, "label": "scan list (e.g. 17,18)"},
#     process_hklscan_only={"widget_type": CheckBox, "label": "Only HKL scans", "value": True},
#     ub_includes_2pi={"widget_type": CheckBox, "label": "UB includes 2π", "value": True},
#     center_is_one_based={"widget_type": CheckBox, "label": "1-based center", "value": False},
#     run_button={   "widget_type": PushButton, "text": "Run RSM3D" }
# )
# def run_rsm(
#     spec_file,
#     setup_file,
#     tiff_dir,
#     selected_scans,
#     process_hklscan_only,
#     ub_includes_2pi,
#     center_is_one_based,
#     run_button,
# ):
#     """
#     1) loads SPEC + TIFF
#     2) builds Q, HKL, intensity
#     3) regrids to (200,*,*) in HKL space
#     4) launches napari viewer
#     """
#     scans = [int(s) for s in selected_scans.split(",") if s.strip()]
#     # 1) load
#     loader = RSMDataLoader(
#         spec_file,
#         setup_file,
#         tiff_dir,
#         selected_scans=scans,
#         process_hklscan_only=process_hklscan_only,
#     )
#     loader.load()
#     # 2) build
#     builder = RSMBuilder(
#         loader,
#         ub_includes_2pi=ub_includes_2pi,
#         center_is_one_based=center_is_one_based,
#     )
#     Q_samp, hkl, intensity = builder.compute_full(verbose=False)
#     # 3) regrid
#     grid, edges = builder.regrid_xu(
#         space="hkl",
#         grid_shape=(200, None, None),
#         fuzzy=True,
#         normalize="mean",
#         stream=True,
#     )
#     # 4) view
#     with gui_qt():
#         viewer = RSMNapariViewer(
#             grid, edges,
#             space="hkl",
#             name="RSM3D",
#             log_view=True,
#             contrast_percentiles=(1, 99.8),
#             cmap="inferno",
#             rendering="attenuated_mip",
#         ).launch()
#     return viewer

# # if you want a single widget to .show() from a notebook:
# gui = run_rsm


     