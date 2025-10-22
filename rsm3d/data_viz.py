from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Iterable, Dict, Any, Sequence, Union

try:
    import napari  # type: ignore
except Exception as e:
    raise ImportError("napari must be installed to use RSMNapariViewer. `pip install napari`") from e


class RSMNapariViewer:
    """
    Robust napari viewer for 3D reciprocal-space maps (HKL or Q).

    Parameters
    ----------
    grid : (nx, ny, nz) ndarray (float or int)
        Gridded intensity volume in X-Y-Z order (not napari order).
    axes : tuple(list/ndarray, list/ndarray, list/ndarray)
        (xax, yax, zax) 1D arrays of coordinates for each axis. Can be non-uniform
        and can be ascending or descending (will be normalized).
    space : {"hkl","q"}
        Labeling/units convenience. "hkl" -> (H,K,L), unit ""; "q" -> (Qx,Qy,Qz), unit "Å⁻¹".
    name : str
        Base layer name for the napari image.
    log_view : bool
        Apply log1p for visualization (does not modify source grid).
    contrast_percentiles : tuple(float, float)
        Percentile bounds for initial contrast limits.
    cmap : str
        Colormap name (e.g., "viridis", "magma").
    rendering : str
        3D rendering mode; one of {"attenuated_mip","mip","translucent"} (fallback if unavailable).
    viewer_kwargs : dict
        Extra kwargs passed to napari.Viewer(...).

    Notes
    -----
    - Ensures full 3D display: sets `viewer.dims.ndisplay=3`, `layer.depiction='volume'` when available.
    - Computes a linear world transform (scale, translate) using average spacing even if axes are non-uniform.
      Exact coordinate readouts use the *actual* axes via interpolation.
    """

    # ------------------------------ construction ------------------------------
    def __init__(
        self,
        grid: np.ndarray,
        axes: Tuple[Iterable[float], Iterable[float], Iterable[float]],
        *,
        space: str = "hkl",
        name: str = "RSM",
        log_view: bool = True,
        contrast_percentiles: Tuple[float, float] = (1.0, 99.8),
        cmap: str = "viridis",
        rendering: str = "attenuated_mip",
        viewer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._validate_grid_axes(grid, axes)
        self.grid, (self.xax, self.yax, self.zax) = self._ensure_ascending(grid, axes)

        # store input meta
        self.space = space.lower()
        if self.space not in {"hkl", "q"}:
            raise ValueError("space must be 'hkl' or 'q'")
        self.name = name
        self.log_view = bool(log_view)
        self.contrast_percentiles = contrast_percentiles
        self.cmap = cmap
        self.rendering = rendering
        self.viewer_kwargs = viewer_kwargs or {}

        # build napari volume + transform
        self.volume, self.scale, self.translate, self.is_uniform = self._volume_from_grid_axes(
            self.grid, (self.xax, self.yax, self.zax)
        )

        # napari objects (populated in launch)
        self.viewer: Optional["napari.Viewer"] = None
        self.img_layer = None
        self._hud_enabled = True

    # ------------------------------ public API --------------------------------
    def launch(self) -> "napari.Viewer":
        """Create the napari viewer and show the full volume with UI niceties."""
        v = napari.Viewer(title=f"{self.name} viewer", **self.viewer_kwargs)
        v.dims.ndisplay = 3  # force 3D

        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        lo, hi = self._robust_percentiles(data, self.contrast_percentiles)

        layer = v.add_image(
            data,
            name=f"{self.name} ({'log1p' if self.log_view else 'linear'})",
            colormap=self.cmap,
            scale=self.scale,          # (Z,Y,X)
            translate=self.translate,  # (Z,Y,X)
            contrast_limits=(float(lo), float(hi)),
        )

        # Force volume depiction + 3D renderer if available
        self._force_volume(layer)

        # Nice camera pose
        # try:
        #     v.reset_view()
        #     if hasattr(v, "camera"):
        #         # Provide an oblique angle so it's clearly 3D
        #         v.camera.angles = (30, 30, 0)  # yaw, pitch, roll (deg)
        #         v.camera.zoom = 1.0
        # except Exception:
        #     pass
        try:
            v.reset_view()
            if hasattr(v, "camera"):
                # Provide an oblique angle
                v.camera.angles = (30, 30, 0)
                v.camera.zoom = 1.0

                # --- AUTO-ZOOM TO DATA EXTENTS ------------------------
                try:
                    # extent.data is ((zmin,zmax),(ymin,ymax),(xmin,xmax))
                    z_range, y_range, x_range = layer.extent.data
                    v.camera.set_range(
                        range_z=z_range,
                        range_y=y_range,
                        range_x=x_range,
                    )
                except Exception:
                    # fallback if set_range not available
                    v.reset_view()
        except Exception:
            pass
        # UI niceties
        v.axes.visible = True
        v.axes.colored = True
        v.axes.arrows = True
        v.scale_bar.visible = True
        if self.space == "q":
            v.scale_bar.unit = "Å⁻¹"
            v.dims.axis_labels = ("Qz", "Qy", "Qx")  # (Z,Y,X)
        else:
            v.scale_bar.unit = ""
            v.dims.axis_labels = ("L", "K", "H")

        # Outline & corners; very light overlays
        self._add_outline_and_corners(v)

        # Axes vectors
        self._add_axes_vectors(v)

        # HUD (coords + intensity)
        self._install_hud(v, layer)

        self.viewer = v
        self.img_layer = layer
        return self

    def add_grid_overlay(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        *,
        thickness_vox: int = 1,
        opacity: float = 0.25,
        color: str = "white",
        name: str = "grid-planes",
        max_planes_per_axis: int = 200,
    ) -> None:
        """
        Add faint grid planes every `spacing` in world coordinates.

        Parameters
        ----------
        spacing : (dx, dy, dz) in world units (HKL or Å⁻¹)
        thickness_vox : int
            Thickness in voxels (approx; planes are snapped to nearest indices).
        opacity : float
        color : str
        name : str
        max_planes_per_axis : int
            Safety cap to avoid accidental huge overlays.
        """
        self._require_viewer()

        # (nx, ny, nz) from the *grid*, but volume is (nz,ny,nx) -> keep index logic on grid
        nx, ny, nz = self.grid.shape
        xax, yax, zax = self.xax, self.yax, self.zax

        # Pick index positions where axis crosses multiples of spacing
        def nearest_indices(ax: np.ndarray, step: float) -> np.ndarray:
            if step <= 0 or ax.size < 2:
                return np.array([], dtype=int)
            start = np.ceil(ax[0] / step) * step
            coords = np.arange(start, ax[-1] + 0.5 * step, step)
            idx = np.searchsorted(ax, coords)
            idx = idx[(idx >= 0) & (idx < ax.size)]
            return np.unique(idx)

        ix = nearest_indices(xax, spacing[0])
        iy = nearest_indices(yax, spacing[1])
        iz = nearest_indices(zax, spacing[2])

        # Safety caps
        if ix.size > max_planes_per_axis or iy.size > max_planes_per_axis or iz.size > max_planes_per_axis:
            raise ValueError("Too many grid planes — reduce spacing or increase max_planes_per_axis.")

        # Build a sparse binary overlay (nx,ny,nz)
        mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        half = max(int(thickness_vox) // 2, 0)

        for k in ix:
            k0, k1 = max(0, k - half), min(nx, k + half + 1)
            mask[k0:k1, :, :] = 1
        for k in iy:
            k0, k1 = max(0, k - half), min(ny, k + half + 1)
            mask[:, k0:k1, :] = 1
        for k in iz:
            k0, k1 = max(0, k - half), min(nz, k + half + 1)
            mask[:, :, k0:k1] = 1

        # Convert to napari order (Z,Y,X) and add as another image layer
        mask_vol = np.ascontiguousarray(mask.transpose(2, 1, 0))
        
        layer = self.viewer.add_image(
            mask_vol,
            name=name,
            opacity=float(opacity),
            blending="additive",
            colormap=color,
            scale=self.scale,
            translate=self.translate,
            contrast_limits=(0, 1),
            rendering="translucent",
        )
        # set interpolation on the layer if supported
        try:
            layer.interpolation = "nearest"
        except Exception:
            pass


    def flip_axis(self, axis: str) -> None:
        """
        Flip one axis ('x','y','z') in both grid and axes, and update the view.
        """
        self._require_viewer()
        axis = axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        # Flip grid + axis arrays (grid is in (nx,ny,nz))
        if axis == "x":
            self.grid = self.grid[::-1, :, :]
            self.xax = self.xax[::-1].copy()
        elif axis == "y":
            self.grid = self.grid[:, ::-1, :]
            self.yax = self.yax[::-1].copy()
        else:
            self.grid = self.grid[:, :, ::-1]
            self.zax = self.zax[::-1].copy()

        # Rebuild volume + transform and refresh the image layer
        self.volume, self.scale, self.translate, self.is_uniform = self._volume_from_grid_axes(
            self.grid, (self.xax, self.yax, self.zax)
        )
        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        if self.img_layer is not None:
            self.img_layer.data = data
            self.img_layer.scale = self.scale
            self.img_layer.translate = self.translate
            self._force_volume(self.img_layer)

    # ------------------------------ internals ---------------------------------
    @staticmethod
    def _validate_grid_axes(grid: np.ndarray, axes) -> None:
        if not isinstance(grid, np.ndarray) or grid.ndim != 3:
            raise ValueError("grid must be a 3D numpy array with shape (nx, ny, nz).")
        xax, yax, zax = axes
        xax = np.asarray(xax); yax = np.asarray(yax); zax = np.asarray(zax)
        if xax.ndim != 1 or yax.ndim != 1 or zax.ndim != 1:
            raise ValueError("Each axis must be 1D.")
        nx, ny, nz = grid.shape
        if len(xax) != nx or len(yax) != ny or len(zax) != nz:
            raise ValueError(f"Axis lengths must match grid shape: {(nx,ny,nz)} vs ({len(xax)},{len(yax)},{len(zax)}).")

    @staticmethod
    def _ensure_ascending(grid: np.ndarray, axes) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """If any axis is descending, flip both that axis and the corresponding grid dimension."""
        xax, yax, zax = [np.asarray(a) for a in axes]
        G = grid
        if xax.size > 1 and xax[1] < xax[0]:
            xax = xax[::-1].copy()
            G = G[::-1, :, :]
        if yax.size > 1 and yax[1] < yax[0]:
            yax = yax[::-1].copy()
            G = G[:, ::-1, :]
        if zax.size > 1 and zax[1] < zax[0]:
            zax = zax[::-1].copy()
            G = G[:, :, ::-1]
        return G, (xax, yax, zax)

    @staticmethod
    def _volume_from_grid_axes(
        grid: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], bool]:
        """
        Convert (nx,ny,nz) + axes -> (nz,ny,nx) for napari with (scale, translate) in (Z,Y,X).
        Uses average spacing for linear transform; returns is_uniform flag.
        """
        xax, yax, zax = axes
        nx, ny, nz = grid.shape
        vol = np.ascontiguousarray(grid.transpose(2, 1, 0))  # (Z,Y,X)

        def avg_step(a: np.ndarray) -> float:
            return float(np.diff(a).mean()) if a.size > 1 else 1.0

        dx, dy, dz = avg_step(xax), avg_step(yax), avg_step(zax)
        translate = (float(zax[0]), float(yax[0]), float(xax[0]))
        scale = (dz, dy, dx)
        is_uniform = (
            (zax.size < 2 or np.allclose(np.diff(zax), dz, rtol=1e-5, atol=1e-8)) and
            (yax.size < 2 or np.allclose(np.diff(yax), dy, rtol=1e-5, atol=1e-8)) and
            (xax.size < 2 or np.allclose(np.diff(xax), dx, rtol=1e-5, atol=1e-8))
        )
        return vol, scale, translate, is_uniform

    @staticmethod
    def _robust_percentiles(a: np.ndarray, prc: Tuple[float, float]) -> Tuple[float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(a, prc)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                hi = lo + 1.0
        return float(lo), float(hi)

    @staticmethod
    def _log1p_clip(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        return np.log1p(np.maximum(a, 0.0))

    @staticmethod
    def _index_to_axis_value(ax: np.ndarray, idx: float) -> float:
        """Map fractional index -> axis value (linear interp between samples)."""
        n = ax.size
        if n == 0:
            return np.nan
        if idx <= 0:
            return float(ax[0])
        if idx >= n - 1:
            return float(ax[-1])
        i0 = int(np.floor(idx))
        t = float(idx - i0)
        return float((1.0 - t) * ax[i0] + t * ax[i0 + 1])

    def _force_volume(self, layer) -> None:
        """Ensure 3D volume depiction and a 3D renderer, across napari versions."""
        # force 3D
        if self.viewer is not None:
            self.viewer.dims.ndisplay = 3
        # set depiction if available
        try:
            if hasattr(layer, "depiction"):
                layer.depiction = "volume"
        except Exception:
            pass
        # set rendering if available
        try:
            if hasattr(layer, "rendering"):
                # If chosen rendering unsupported, napari will fallback; we try preferred order.
                preferred = [self.rendering, "attenuated_mip", "mip", "translucent"]
                for r in preferred:
                    try:
                        layer.rendering = r
                        break
                    except Exception:
                        continue
        except Exception:
            pass

    def _add_outline_and_corners(self, v: "napari.Viewer") -> None:
        # world bounds from axes (already ascending)
        zmin, zmax = float(self.zax[0]), float(self.zax[-1])
        ymin, ymax = float(self.yax[0]), float(self.yax[-1])
        xmin, xmax = float(self.xax[0]), float(self.xax[-1])

        corners_world = np.array([
            [zmin, ymin, xmin],
            [zmin, ymin, xmax],
            [zmin, ymax, xmin],
            [zmin, ymax, xmax],
            [zmax, ymin, xmin],
            [zmax, ymin, xmax],
            [zmax, ymax, xmin],
            [zmax, ymax, xmax],
        ], dtype=float)

        # tiny corner markers: take 15% of the *mean* voxel size (isotropic)
        voxel = np.array(self.scale, dtype=float)  # (dz, dy, dx)
        mean_vox = float(np.mean(voxel))
        corner_size = mean_vox * 0.15
        v.add_points(
            corners_world,
            name="Outline corners",
            size=np.full(8, corner_size),
            face_color="red",
            # edge_width=0,
            opacity=0.9,
            blending="additive",
        )

        box_edges = np.array([
            [0,1],[0,2],[0,4],
            [1,3],[1,5],
            [2,3],[2,6],
            [3,7],
            [4,5],[4,6],
            [5,7],
            [6,7],
        ], dtype=int)
        edge_segments = [corners_world[e] for e in box_edges]

       # frame edges: 5% of mean voxel size
        edge_width = mean_vox * 0.05
        kwargs = dict(
               shape_type="line",
               edge_color="yellow",
               edge_width=edge_width,
               opacity=0.9,
               blending="additive",
               name="Outline box",
           )
        v.add_shapes(edge_segments, **kwargs)
        
         # ----------------------------------------------------------------
        # annotate each corner with its (H,K,L) world‐coords via a Points‐layer text
        labels = [f"H={h:.3f}, K={k:.3f}, L={l:.3f}" for l, k, h in corners_world]
        pts = v.add_points(
            corners_world,
            name="Corner labels",
            text=labels,                    # per‐point label
            face_color="white",             # text color
            # text_color="white",
            # text_size=mean_vox * 0.1,       # scale text by voxel size
            size=0.0,                       # hide the marker itself
            blending="additive",
        )

    def _add_axes_vectors(self, v: "napari.Viewer") -> None:
        # use 10% of largest world extent for vector length
        Lx = float(self.xax[-1] - self.xax[0])
        Ly = float(self.yax[-1] - self.yax[0])
        Lz = float(self.zax[-1] - self.zax[0])
        axes_len = 0.10 * max(Lx, Ly, Lz)

        origin = np.array([float(self.zax[0]), float(self.yax[0]), float(self.xax[0])], dtype=float)
        vectors = np.stack([
            np.vstack([origin, origin + np.array([axes_len, 0, 0])]),  # +Z
            np.vstack([origin, origin + np.array([0, axes_len, 0])]),  # +Y
            np.vstack([origin, origin + np.array([0, 0, axes_len])]),  # +X
        ], axis=0)

       # line width ~5% of mean voxel size
        mean_vox = float(np.mean(self.scale))
        vec_width = mean_vox * 0.05
        kwargs = dict(
           name="World axes",
           edge_color=["cyan", "lime", "magenta"],
           edge_width=vec_width,
           blending="translucent_no_depth",
        )
        try:
            v.add_vectors(vectors, **kwargs)
        except TypeError:
            # Some versions use `width` instead of edge_width
            kwargs.pop("edge_width", None)
            kwargs["width"] = 0.75
            v.add_vectors(vectors, **kwargs)

    def _install_hud(self, v: "napari.Viewer", layer) -> None:
        # Closure captures self, layer, and axes
        def on_mouse_move(viewer, event):
            if not self._hud_enabled:
                return
            pos_world = viewer.cursor.position
            if pos_world is None:
                return
            # data indices (Z,Y,X)
            zi, yi, xi = layer.world_to_data(pos_world)
            # Intensity from *linear* volume (not log)
            I = np.nan
            zi_i, yi_i, xi_i = int(np.round(zi)), int(np.round(yi)), int(np.round(xi))
            if (0 <= zi_i < self.volume.shape[0]) and (0 <= yi_i < self.volume.shape[1]) and (0 <= xi_i < self.volume.shape[2]):
                I = float(self.volume[zi_i, yi_i, xi_i])

            # exact coords from axes (HKL or Q)
            H = self._index_to_axis_value(self.xax, xi)
            K = self._index_to_axis_value(self.yax, yi)
            L = self._index_to_axis_value(self.zax, zi)
            if self.space == "q":
                text = f"Qx={H:.4f} Å⁻¹   Qy={K:.4f} Å⁻¹   Qz={L:.4f} Å⁻¹    I={I:.3g}"
            else:
                text = f"H={H:.4f}   K={K:.4f}   L={L:.4f}    I={I:.3g}"

            # text overlay if available, else print
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = True
                overlay.position = 'top_left'
                overlay.color = 'white'
                overlay.font_size = 12
                overlay.text = text
            else:
                print(text, end="\r")

        v.mouse_move_callbacks.append(on_mouse_move)

        @v.bind_key('C')
        def _toggle_coords(viewer):
            """Toggle HUD overlay with 'C' key."""
            self._hud_enabled = not self._hud_enabled
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = self._hud_enabled

    def _require_viewer(self) -> None:
        if self.viewer is None:
            raise RuntimeError("Call launch() before adding overlays or flipping axes.")
        
        
Array2D = np.ndarray  # (Y, X)

def _robust_percentiles(a: np.ndarray, lo=1.0, hi=99.8) -> Tuple[float, float]:
    a = np.asarray(a)
    m = np.isfinite(a)
    if not m.any():
        return 0.0, 1.0
    v = np.percentile(a[m], [lo, hi])
    if v[0] == v[1]:
        v[1] = v[0] + 1e-6
    return float(v[0]), float(v[1])

def _stack_list_of_2d(
    frames: Sequence[Array2D],
    *,
    pad_value: float = np.nan,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("Empty intensity list.")

    shapes = []
    clean_frames: list[np.ndarray] = []
    for i, f in enumerate(frames):
        if f is None:
            continue
        a = np.asarray(f)
        if a.ndim != 2:
            raise ValueError(f"Frame {i} is not 2D (shape={a.shape})")
        shapes.append(a.shape)
        clean_frames.append(a)

    if not clean_frames:
        raise ValueError("All frames were None/invalid.")

    max_y = max(s[0] for s in shapes)
    max_x = max(s[1] for s in shapes)
    if all(s == (max_y, max_x) for s in shapes):
        # Allow copy when needed by dropping copy=False
        return np.stack([np.asarray(f, dtype=dtype) for f in clean_frames], axis=0)
    stacked = np.full((len(clean_frames), max_y, max_x), pad_value, dtype=dtype)
    for t, a in enumerate(clean_frames):
        y, x = a.shape
        stacked[t, :y, :x] = a
    return stacked

def _maybe_series_to_list(obj):
    # Avoid hard dep on pandas; detect lightly
    if hasattr(obj, "to_numpy") and hasattr(obj, "values") and hasattr(obj, "iloc"):
        try:
            return list(obj.to_numpy())
        except Exception:
            try:
                return list(obj.values)
            except Exception:
                return list(obj)
    return obj

def _to_tyx_any(intensity: Union[np.ndarray, Sequence[Array2D]]) -> np.ndarray:
    """
    Accept:
      • list/tuple of 2D arrays
      • pandas Series of 2D arrays
      • 1D object ndarray of 2D arrays
      • 3D ndarray (T,Y,X)
      • 2D ndarray (Y,X)
    Returns (T, Y, X).
    """
    intensity = _maybe_series_to_list(intensity)

    # List/tuple → stack
    if isinstance(intensity, (list, tuple)):
        return _stack_list_of_2d(intensity)

    a = np.asarray(intensity)
    # 3D numeric array already
    if a.ndim == 3 and a.dtype != object:
        return a
    # 2D numeric array
    if a.ndim == 2 and a.dtype != object:
        return a[None, ...]

    # 1D object array → elements should be 2D arrays
    if a.ndim == 1 and a.dtype == object:
        return _stack_list_of_2d(list(a))

    raise ValueError(
        f"Unsupported intensity shape {a.shape}; expected a list/Series/1D-object array "
        f"of 2D frames, a 2D array, or a 3D (T,Y,X) array."
    )

class IntensityNapariViewer:
    """
    Napari viewer for raw intensity frames.

    Parameters
    ----------
    intensity : list/Series/1D-object-ndarray of 2D frames OR 2D/3D ndarray
    name : str
    log_view : bool
    contrast_percentiles : (float, float)
    cmap : str
    rendering : str  ('attenuated_mip', 'mip', 'translucent', 'additive', 'minip')
    add_timeseries : bool   # (T,Y,X) with time slider
    add_volume : bool       # treat T as Z for 3D
    scale_tzyx : (float, float, float)  # spacing for (T/Z, Y, X)
    pad_value : float       # pad value for mismatched frames
    """
    def __init__(
        self,
        intensity: Union[np.ndarray, Sequence[Array2D]],
        *,
        name: str = "Intensity",
        log_view: bool = True,
        contrast_percentiles: Tuple[float, float] = (1.0, 99.8),
        cmap: str = "inferno",
        rendering: str = "attenuated_mip",
        add_timeseries: bool = True,
        add_volume: bool = True,
        scale_tzyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pad_value: float = np.nan,
    ):
        self._name = name
        self._log = bool(log_view)
        self._p_lo, self._p_hi = map(float, contrast_percentiles)
        self._cmap = cmap
        self._rendering = rendering
        self._add_ts = bool(add_timeseries)
        self._add_vol = bool(add_volume)
        self._scale = tuple(map(float, scale_tzyx))
        self._pad_value = float(pad_value)

        # Coerce to (T,Y,X)
        tyx = _to_tyx_any(intensity)
        # If we got here through an object array path, ensure dtype float32
        self._raw_tyx = tyx.astype(np.float32, copy=False)

        self._viewer: Optional[napari.Viewer] = None
        self._layer_ts = None
        self._layer_vol = None

    @classmethod
    def from_loader(cls, loader, **kwargs) -> "IntensityNapariViewer":
        setup, UB, df = loader.load()
        intensity = getattr(df, "intensity", None)
        if intensity is None:
            raise ValueError("Loader returned df without 'intensity'")
        return cls(intensity, **kwargs)

    def launch(self) -> napari.Viewer:
        """Show only the intensity frames as 2D slices with a draggable ROI."""
        v = napari.Viewer(title=self._name)
        self._viewer = v

        # Prepare data (F, H, W)
        data = self._prepare_data(self._raw_tyx)
        lo, hi = _robust_percentiles(data, self._p_lo, self._p_hi)

        # Single image layer renamed to Intensity(F,H,W)
        self._layer_ts = v.add_image(
            data,
            name=f"{self._name} (F,H,W)",
            contrast_limits=(lo, hi),
            colormap=self._cmap,
            blending="translucent",
            scale=self._scale,
        )
        v.dims.ndisplay = 2

        # Hide any accidental extra layers
        for layer in list(v.layers):
            if layer is not self._layer_ts:
                layer.visible = False

        # Add a centered, half-size rectangle ROI on the H-W plane
        _, H, W = data.shape
        half_h = H / 4.0
        half_w = W / 4.0
        y0 = H / 2.0 - half_h
        x0 = W / 2.0 - half_w
        y1 = H / 2.0 + half_h
        x1 = W / 2.0 + half_w
        rect = np.array([
            [y0, x0],
            [y0, x1],
            [y1, x1],
            [y1, x0],
        ])
        shapes = v.add_shapes(
            [rect],
            shape_type="rectangle",
            edge_color="red",
            face_color="transparent",
            name="ROI",
        )
        shapes.editable = True
        shapes.mode = "transform"

        return v

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def _prepare_data(self, tyx: np.ndarray) -> np.ndarray:
        a = tyx.astype(np.float32, copy=False)
        if self._log:
            a = np.log1p(np.maximum(a, 0.0))
        return a