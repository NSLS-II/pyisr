import napari
import numpy as np

def volume_from_grid_axes(grid, axes):
    xax, yax, zax = [np.asarray(a) for a in axes]
    nx, ny, nz = len(xax), len(yax), len(zax)
    if grid.shape != (nx, ny, nz):
        raise ValueError(f"grid shape {grid.shape} != ({nx},{ny},{nz})")
    vol = grid.transpose(2,1,0).copy()  # (Z,Y,X)
    def _avg_step(a):
        return float(np.diff(a).mean()) if len(a) > 1 else 1.0
    dx, dy, dz = _avg_step(xax), _avg_step(yax), _avg_step(zax)
    translate = (float(zax[0]), float(yax[0]), float(xax[0]))
    scale     = (dz, dy, dx)
    is_uniform = (
        np.allclose(np.diff(xax), dx) and
        np.allclose(np.diff(yax), dy) and
        np.allclose(np.diff(zax), dz)
    )
    return vol, scale, translate, is_uniform

def log1p_clip(a):
    a = np.asarray(a)
    return np.log1p(np.maximum(a, 0.0))

def index_to_axis_value(ax, idx):
    n = len(ax)
    if n == 0:
        return np.nan
    if idx <= 0:
        return float(ax[0])
    if idx >= n - 1:
        return float(ax[-1])
    i0 = int(np.floor(idx))
    t  = float(idx - i0)
    return float((1 - t) * ax[i0] + t * ax[i0 + 1])

class RSMNapariViewer:
    """
    Callable viewer for RSM volumes, suitable for Jupyter.

    Parameters
    ----------
    grid : 3D numpy array, shape (nx,ny,nz)
    axes : tuple of 3 arrays (xax, yax, zax)
    raw_intensity : optional 3D or 4D array for raw frames (time,z,y,x)
    """
    def __init__(self, grid, axes, raw_intensity=None):
        self.grid = grid
        self.axes = axes
        self.raw  = raw_intensity
        self.viewer = None

    def __call__(self, display_3d=True, use_log=True):
        """
        Launch the Napari viewer (in-process) from Jupyter.

        Parameters
        ----------
        display_3d : bool
            True for 3D volume rendering, False for 2D slice view.
        use_log : bool
            Apply log1p to the RSM volume.
        """
        vol, scale, translate, is_uniform = volume_from_grid_axes(self.grid, self.axes)
        data = log1p_clip(vol) if use_log else vol

        # create the viewer
        ndisp = 3 if display_3d else 2
        v = napari.Viewer(ndisplay=ndisp, title="RSM viewer")

        # add the RSM volume
        img = v.add_image(
            data,
            name="RSM (log1p)" if use_log else "RSM",
            scale=scale,
            translate=translate,
            rendering="attenuated_mip" if display_3d else None,
            blending="translucent" if display_3d else "additive"
        )

        # optionally add raw‐frame stack as time series

        # draw thin outline of the volume in world coords
        zax, yax, xax = self.axes
        corners = np.array([
            [float(zax[0]), float(yax[0]), float(xax[0])],
            [float(zax[0]), float(yax[0]), float(xax[-1])],
            [float(zax[0]), float(yax[-1]), float(xax[0])],
            [float(zax[0]), float(yax[-1]), float(xax[-1])],
            [float(zax[-1]), float(yax[0]), float(xax[0])],
            [float(zax[-1]), float(yax[0]), float(xax[-1])],
            [float(zax[-1]), float(yax[-1]), float(xax[0])],
            [float(zax[-1]), float(yax[-1]), float(xax[-1])]
        ])
        edges = [
            (0,1),(0,2),(0,4),(1,3),(1,5),
            (2,3),(2,6),(3,7),(4,5),(4,6),
            (5,7),(6,7)
        ]
          # need to index rows by list, not tuple
        segs = [corners[list(e)] for e in edges]

        v.add_shapes(segs, shape_type="line", edge_color="yellow", edge_width=0.1)
        # live‐cursor callback: show H,K,L and I under cursor
        def on_move(viewer, event):
            pos = viewer.cursor.position
            if pos is None:
                return
            zi, yi, xi = img.world_to_data(pos)
            zi_i, yi_i, xi_i = map(int, np.round([zi, yi, xi]))
            I = float(vol[zi_i, yi_i, xi_i]) if (0 <= zi_i < vol.shape[0]) else np.nan
            H = index_to_axis_value(xax, xi)
            K = index_to_axis_value(yax, yi)
            L = index_to_axis_value(zax, zi)
            text = f"H={H:.4f} K={K:.4f} L={L:.4f} I={I:.3g}"
            if hasattr(viewer, "text_overlay") and viewer.text_overlay is not None:
                ov = viewer.text_overlay
                ov.text = text
                ov.visible = True
                ov.position = "top_left"
            else:
                print(text, end="\r")

        v.mouse_move_callbacks.append(on_move)
        self.viewer = v

        # actually start the Qt event loop so the Napari window pops up
        napari.run()

        return v