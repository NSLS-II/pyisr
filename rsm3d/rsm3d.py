import numpy as np
import pandas as pd
from typing import Any, Optional, Sequence, Tuple, Union
from scipy.interpolate import griddata
import xrayutilities as xu


_TWO_PI = 2.0 * np.pi

class RSMBuilder:
    @staticmethod
    def _coerce_loader_payload(payload: Any):
        if isinstance(payload, tuple):
            if len(payload) == 3:
                return payload
            if len(payload) == 2:
                setup, df = payload
                return setup, None, df
            raise ValueError("loader.load() tuple must be (setup, df) or (setup, UB, df).")
        if isinstance(payload, dict):
            setup = payload.get("setup")
            df = payload.get("df")
            ub = payload.get("UB", payload.get("ub"))
            if setup is not None and df is not None:
                return setup, ub, df
        if all(hasattr(payload, attr) for attr in ("setup", "df")):
            setup = getattr(payload, "setup")
            df = getattr(payload, "df")
            ub = getattr(payload, "UB", getattr(payload, "ub", None))
            return setup, ub, df
        raise TypeError(
            "loader.load() must return (setup, df) or (setup, UB, df), or expose setup/df[/UB] attributes."
        )

    def __init__(
        self,
        loader,
        *,
        motor_map: dict | None = None,
        ub_includes_2pi: bool = True,
        center_is_one_based: bool = False,
        dtype=np.float32,
        sample_axes: Optional[Sequence[str]] = None,
        detector_axes: Optional[Sequence[str]] = None,
        
    ):
        loaded = loader.load()
        # print(loaded)
        self.setup, self.UB, self.df = self._coerce_loader_payload(loaded)
        # print('Data loaded:')
        self.dtype = np.dtype(dtype)

        self.UB = None if self.UB is None else np.asarray(self.UB, dtype=np.float64)
        self._has_global_ub = self.UB is not None
        self._row_has_ub = "ub" in self.df.columns and self.df["ub"].notna().any()
        self._compute_hkl = self._has_global_ub or self._row_has_ub
        
        self.ub_includes_2pi = bool(ub_includes_2pi)
        # Image shape
        ny, nx = self.df["intensity"].iat[0].shape
        self.img_shape = (ny, nx)
        # Wavelength (Å)
        lam_A = float(self.setup.wavelength)
        # Geometry
        dist_m = float(self.setup.distance)
        pitch_m = float(self.setup.pitch)

        x0 = float(self.setup.xcenter) - (1.0 if center_is_one_based else 0.0)
        y0 = float(self.setup.ycenter) - (1.0 if center_is_one_based else 0.0)
        x0 = np.clip(x0, 0, nx - 1)
        y0 = np.clip(y0, 0, ny - 1)

        # xrayutilities QConversion
        self.sample_angle_names = ('omega', 'chi', 'phi')
        self.detector_angle_names = ('theta',)
        
        print('Initializing QConversion area...')

        default_sample_axes = ('z-', 'y-', 'x+')
        default_detector_axes = ('z+',)

        def _coerce_axes(user_axes, default):
            if user_axes is None:
                return list(default)
            if isinstance(user_axes, str):
                return [user_axes]
            return list(user_axes)

        def _validate_axes(name, axes, expected_len, *, allow_empty=False):
            if not axes:
                if allow_empty:
                    return []
                raise ValueError(f"{name} must contain {expected_len} entries.")
            if expected_len is not None and len(axes) != expected_len:
                raise ValueError(
                    f"{name} must contain {expected_len} entries; got {len(axes)}."
                )
            if any(not isinstance(axis, str) for axis in axes):
                raise TypeError(f"All entries in {name} must be strings.")
            return axes

        sampleAxis = _validate_axes(
            "sample_axes",
            _coerce_axes(sample_axes, default_sample_axes),
            len(self.sample_angle_names),
        )
        detectorAxis = _validate_axes(
            "detector_axes",
            _coerce_axes(detector_axes, default_detector_axes),
            len(self.detector_angle_names),
            allow_empty=True,
        )

        self.sample_axes = tuple(sampleAxis)
        self.detector_axes = tuple(detectorAxis)

        # beam direction: along +Y
        r_i = (0, 1, 0)

        # QConversion (sample first, then detector)
        self.qconv = xu.experiment.QConversion(sampleAxis, detectorAxis, r_i, wl=lam_A)

        # detector mapping so returned arrays are (ny, nx)
        # dir1 (rows) along Z (use 'z-' to keep +Z up with row index increasing downward)
        # dir2 (cols) along +X
        self.qconv.init_area(
            'z-', 'x+',
            cch1=y0, cch2=x0,
            Nch1=ny, Nch2=nx,
            distance=dist_m,
            pwidth1=pitch_m, pwidth2=pitch_m,
            detrot=0.0, tiltazimuth=0.0, tilt=0.0
        )
        print('Initialized QConversion area with:')
        print(f"  Sample Axis: {self.sample_axes}")
        print(f"  Detector Axis: {self.detector_axes}")
        print(f"  Beam Direction: {r_i}")
        print(f"  Wavelength: {lam_A:.6f} Å")
        print(f"  Distance: {dist_m:.6f} m")
        print(f"  Pixel Width: {pitch_m:.6f} m")

        # motor names (include tth)
        default_motor_map = {"omega": "th", "chi": "chi", "phi": "phi", "tth": "tth"}
        self.motor_map = {**default_motor_map, **(motor_map or {})}

        # remember if a tth column is actually present
        self._has_tth = self.motor_map["tth"] in self.df.columns

    # ───────────────────────────────────────────────────────────────────────────
    # Core mapping
    # ───────────────────────────────────────────────────────────────────────────
    def compute_full(self, verbose: bool = True):
        """
        Compute per-pixel Q (Å⁻¹) and HKL for each frame using xrayutilities.

        Returns
        -------
        Q_samp : (Nf, ny, nx, 3) float32  (Å⁻¹)
        hkl    : (Nf, ny, nx, 3) float32
        intensity : (Nf, ny, nx) float32
        """
        df = self.df
        Nf = len(df)
        ny, nx = self.img_shape

        Q_samp = np.empty((Nf, ny, nx, 3), dtype=self.dtype)
        HKL    = np.empty_like(Q_samp)
        Icube  = np.empty((Nf, ny, nx), dtype=self.dtype)

        UB2pi_default = (self.UB if self.ub_includes_2pi else (_TWO_PI * self.UB))
        

        for idx, row in enumerate(df.itertuples(index=False)):
            # intensity array
            I = np.asarray(row.intensity, dtype=self.dtype)
            if I.shape != (ny, nx):
                raise ValueError(f"Frame shape {I.shape} != expected {(ny, nx)}")

            # pull motors with mapping (logical names -> df columns)
            omega = float(getattr(row, self.motor_map["omega"]))
            chi   = float(getattr(row, self.motor_map["chi"]))
            phi   = float(getattr(row, self.motor_map["phi"]))
            tth   = float(getattr(row, self.motor_map["tth"])) if self._has_tth else 0.0

            # Assemble angle tuple in the order XU expects:
            #   (*sample_angles outer→inner, *detector_angles)
            # → (phi, chi, omega, tth) if a detectorAxis was given, else (phi, chi, omega)
            if len(self.qconv.detectorAxis):
                angs = (omega, chi, phi, tth)
            else:
                angs = (omega, chi, phi)
            # print("angles:", angs)
            # Q in Å^-1
            qx, qy, qz = self.qconv.area(*angs, wl=self.qconv.wavelength, deg=True)
            Qf = np.stack((qx, qy, qz), axis=-1).astype(self.dtype, copy=False)

            # HKL via UB (2π convention for XU). Allow per-frame UB override.
            UB_row = getattr(row, "ub", None)
            # print(UB_row)
            UB2pi = np.asarray(UB_row, dtype=np.float64) if UB_row is not None else UB2pi_default
            if not self.ub_includes_2pi and UB_row is not None:
                UB2pi = _TWO_PI * UB2pi
            

            h, k, l = self.qconv.area(*angs, wl=self.qconv.wavelength, deg=True, UB=UB2pi)
            # manually apply -1 to h to convert from XU to HKL convention
            HKLf = np.stack((h, k, l), axis=-1).astype(self.dtype, copy=False)

            Q_samp[idx] = Qf
            HKL[idx]    = HKLf
            Icube[idx]  = I

            if verbose and (idx % 10 == 0 or idx == Nf - 1):
                print(f"Processed {idx+1}/{Nf} frames", end="\r")

        self.Q_samp   = Q_samp
        self.hkl      = HKL
        self.intensity = Icube
        return Q_samp, HKL, Icube

    # ───────────────────────────────────────────────────────────────────────────
    # Regridding with xrayutilities (3D)
    # ───────────────────────────────────────────────────────────────────────────
    def regrid_xu(
        self,
        *,
        space: str = "q",                # "q" or "hkl"
        grid_shape: Union[int, Tuple[int,int,int]] = (200, 200, 200),
        ranges: Optional[Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]]] = None,
        fuzzy: bool = False,
        width: Optional[float] = None,
        normalize: str = "mean",         # "mean", "sum", or None
        stream: bool = False,
    ):
        """
        Scatter‐to‐grid re‐binning using xrayutilities Gridder3D (or FuzzyGridder3D).

        Parameters
        ----------
        space
            "q" to grid Q_samp or "hkl" to grid self.hkl.
        grid_shape
            int → base nx; ny,nz auto‐scaled by data extents ratios
            (nx, ny, nz) → fixed shape
            (nx, None, None) or (nx, -1, -1) → nx fixed; ny,nz auto‐scaled
        ranges
            ((minx,maxx),(miny,maxy),(minz,maxz)); if None, auto‐computed from data
        fuzzy
            if True use xu.FuzzyGridder3D; else xu.Gridder3D
        width
            fuzzy width (if fuzzy=True)
        normalize
            "mean" (default) → normalize by point‐counts; "sum" → sum weighting
        stream
            if True accumulate frame‐by‐frame (lower peak RAM, retains raw points)

        Returns
        -------
        grid : ndarray
            3D volume array of shape (nx, ny, nz)
        (xaxis, yaxis, zaxis) : tuple of 1D arrays
            bin centers along each dimension
        """
        # select data array
        arr = self.Q_samp if space.lower() == "q" else self.hkl

        # auto‐compute axis ranges if not provided
        if ranges is None:
            ranges = tuple(
                (
                    float(np.nanmin(arr[..., i])),
                    float(np.nanmax(arr[..., i])),
                )
                for i in range(3)
            )
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = ranges
        # spans for aspect ratios
        Lx = max(1e-12, xmax - xmin)
        Ly = max(1e-12, ymax - ymin)
        Lz = max(1e-12, zmax - zmin)

        # helper to auto‐scale ny,nz from nx
        def _auto_shape(nx_val: int) -> Tuple[int,int,int]:
            nxv = max(2, int(nx_val))
            nyv = max(2, int(round(nxv * (Ly / Lx))))
            nzv = max(2, int(round(nxv * (Lz / Lx))))
            return nxv, nyv, nzv

        # interpret grid_shape
        if isinstance(grid_shape, int):
            nx, ny, nz = _auto_shape(grid_shape)
        else:
            gx = list(grid_shape)
            if len(gx) != 3:
                raise ValueError("grid_shape must be int or length‐3 tuple.")
            nx = gx[0]
            # auto‐compute missing dims
            if gx[1] in (None, -1):
                nx, ny, nz = _auto_shape(nx)
            else:
                ny = gx[1]
                nz = gx[2] if gx[2] not in (None, -1) else _auto_shape(nx)[2]
        nx, ny, nz = int(nx), int(ny), int(nz)

        # build the gridder
        Gridder = xu.FuzzyGridder3D if fuzzy else xu.Gridder3D
        G = Gridder(nx, ny, nz)
        if stream:
            G.KeepData(True)

        # set data range (fixed=True if supported)
        try:
            G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax, fixed=True)
        except TypeError:
            G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax)

        # feed scattered points
        if stream:
            # per‐frame loop
            for i in range(self.intensity.shape[0]):
                Xi = arr[i, ..., 0].ravel()
                Yi = arr[i, ..., 1].ravel()
                Zi = arr[i, ..., 2].ravel()
                Wi = self.intensity[i].ravel()
                if fuzzy and width is not None:
                    G(Xi, Yi, Zi, Wi, width=width)
                else:
                    G(Xi, Yi, Zi, Wi)
        else:
            # all‐points at once
            X = arr[..., 0].ravel()
            Y = arr[..., 1].ravel()
            Z = arr[..., 2].ravel()
            W = self.intensity.ravel()
            if fuzzy and width is not None:
                G(X, Y, Z, W, width=width)
            else:
                G(X, Y, Z, W)

        # normalize
        do_norm = False if normalize and normalize.lower() == "sum" else True
        G.Normalize(do_norm)

        # extract results
        grid = G.data.astype(self.dtype, copy=False)
        xax, yax, zax = G.xaxis, G.yaxis, G.zaxis
        return grid, (xax, yax, zax)
    # def regrid_xu(
    #     self,
    #     *,
    #     space: str = "q",                 # "q" or "hkl"
    #     grid_shape=(200, 200, 200),       # (nx, ny, nz)
    #     ranges=None,                      # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
    #     fuzzy: bool = False,              # use FuzzyGridder3D
    #     width=None,                       # scalar or (wx,wy,wz) for fuzzy (same units as axes)
    #     normalize: str = "mean",          # "mean" → averaged; "sum" → accumulated
    #     stream: bool = False              # iterate frame-by-frame to save RAM
    # ):
    #     assert space.lower() in ("q", "hkl")
    #     nx, ny, nz = map(int, grid_shape)
    #     # arr = self.Q_samp if space.lower() == "q" else self.hkl
    #     arr = self.Q_samp if space.lower() == "q" else self.hkl
        
    #     #+        # ensure we have a (x,y,z) range tuple
    #     if ranges is None:
    #        ranges = tuple(
    #            (float(np.nanmin(arr[..., i])), float(np.nanmax(arr[..., i])))
    #            for i in range(3)
    #        )

    #    # build the gridder
    #     G = (xu.FuzzyGridder3D if fuzzy else xu.Gridder3D)(nx, ny, nz)
    #     if stream:
    #        G.KeepData(True)

    #    # apply the ranges (try fixed=True if supported)
    #     (xmin, xmax), (ymin, ymax), (zmin, zmax) = ranges
    #     try:
    #        G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax, fixed=True)
    #     except TypeError:
    #        G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax)
    #     print(ranges)

    #     if stream:
    #         for i in range(self.intensity.shape[0]):
    #             Xi = arr[i, ..., 0].ravel()
    #             Yi = arr[i, ..., 1].ravel()
    #             Zi = arr[i, ..., 2].ravel()
    #             Wi = self.intensity[i].ravel()
    #             if fuzzy and width is not None:
    #                 G(Xi, Yi, Zi, Wi, width=width)
    #             else:
    #                 G(Xi, Yi, Zi, Wi)
    #     else:
    #         X = arr[..., 0].ravel()
    #         Y = arr[..., 1].ravel()
    #         Z = arr[..., 2].ravel()
    #         W = self.intensity.ravel()
    #         if fuzzy and width is not None:
    #             G(X, Y, Z, W, width=width)
    #         else:
    #             G(X, Y, Z, W)

    #     G.Normalize(False if normalize.lower() == "sum" else True)
    #     grid = G.data.astype(self.dtype, copy=False)
    #     xax, yax, zax = G.xaxis, G.yaxis, G.zaxis
    #     return grid, (xax, yax, zax)
    # ───────────────────────────────────────────────────────────────────────────
    # Regridding with xrayutilities (3D)
    # # ───────────────────────────────────────────────────────────────────────────
    # def regrid_xu(
    #     self,
    #     *,
    #     space: str = "q",                 # "q" or "hkl"
    #     grid_shape=(200, 200, 200),       # (nx, ny, nz)
    #     ranges=None,                      # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None→auto
    #     fuzzy: bool = False,              # FuzzyGridder3D if True
    #     width=None,                       # scalar or (wx,wy,wz) for fuzzy
    #     normalize: str = "mean",          # "mean" or "sum"
    #     stream: bool = False              # iterate frame-by-frame to save RAM
    # ):
    #     """
    #     Regrid scattered points with xrayutilities Gridder3D/FuzzyGridder3D.
    #     Returns (grid, (xaxis, yaxis, zaxis))
    #     """
    #     assert space.lower() in ("q", "hkl")
    #     nx, ny, nz = map(int, grid_shape)
    #     arr = self.Q_samp if space.lower() == "q" else self.hkl

    #     G = (xu.FuzzyGridder3D if fuzzy else xu.Gridder3D)(nx, ny, nz)
    #     if ranges is not None:
    #         (xmin, xmax), (ymin, ymax), (zmin, zmax) = ranges
    #         try:
    #             G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax, fixed=True)
    #         except TypeError:
    #             G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax)

    #     if stream:
    #         for i in range(self.intensity.shape[0]):
    #             Xi = arr[i, ..., 0].ravel()
    #             Yi = arr[i, ..., 1].ravel()
    #             Zi = arr[i, ..., 2].ravel()
    #             Wi = self.intensity[i].ravel()
    #             if fuzzy and width is not None:
    #                 G(Xi, Yi, Zi, Wi, width=width)
    #             else:
    #                 G(Xi, Yi, Zi, Wi)
    #     else:
    #         X = arr[..., 0].ravel(); Y = arr[..., 1].ravel(); Z = arr[..., 2].ravel()
    #         W = self.intensity.ravel()
    #         if fuzzy and width is not None:
    #             G(X, Y, Z, W, width=width)
    #         else:
    #             G(X, Y, Z, W)

    #     # normalization
    #     grid = None
    #     if normalize.lower() == "sum":
    #         if hasattr(G, "Normalize"):
    #             G.Normalize(False)
    #         grid = np.array(getattr(G, "gdata", getattr(G, "data")), copy=False)
    #     else:
    #         if hasattr(G, "Normalize"):
    #             G.Normalize(True)
    #         if hasattr(G, "normalize"):
    #             try: G.normalize()
    #             except Exception: pass
    #         grid = np.array(getattr(G, "data", getattr(G, "gdata")), copy=False)

    #     xax = getattr(G, "xaxis", getattr(G, "x", None))
    #     yax = getattr(G, "yaxis", getattr(G, "y", None))
    #     zax = getattr(G, "zaxis", getattr(G, "z", None))
    #     return grid.astype(self.dtype, copy=False), (xax, yax, zax)

    # ───────────────────────────────────────────────────────────────────────────
    # Optional NumPy-based regridders (back-compat)
    # ───────────────────────────────────────────────────────────────────────────
    # def setup_grid(self, grid_ranges, grid_shape):
    #     self.grid_ranges = grid_ranges
    #     self.grid_shape = grid_shape
    #     self.edges = [
    #         np.linspace(r[0], r[1], grid_shape[i] + 1)
    #         for i, r in enumerate(grid_ranges)
    #     ]

    # def regrid_intensity(self, method='sum', space='q'):
    #     if space == 'q':
    #         if not hasattr(self, "edges"):
    #             raise RuntimeError("Call setup_grid() or regrid_auto(space='q') first.")
    #         pts, edges = self.Q_samp.reshape(-1,3), self.edges
    #     else:
    #         pts, edges = self.hkl.reshape(-1,3), self.hkl_edges
    #     vals = self.intensity.ravel().astype(np.float64, copy=False)
    #     H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
    #     if method=='sum':
    #         return H_sum.astype(self.dtype, copy=False), edges
    #     H_cnt, _ = np.histogramdd(pts, bins=edges)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         Hm = H_sum / H_cnt
    #         Hm[~np.isfinite(Hm)] = 0
    #     return Hm.astype(self.dtype, copy=False), edges

    # def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
    #     arr = self.Q_samp if space=='q' else self.hkl
    #     ranges = tuple((arr[...,k].min(), arr[...,k].max()) for k in range(3))
    #     if space=='q':
    #         self.setup_grid(ranges, grid_shape)
    #     else:
    #         self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i]+1)
    #                           for i,r in enumerate(ranges)]
    #     return self.regrid_intensity(method=method, space=space)

    # def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
    #     pts = (self.Q_samp if space=='q' else self.hkl).reshape(-1,3)
    #     vals = self.intensity.ravel()
    #     mask = vals>0
    #     pts, vals = pts[mask], vals[mask]
    #     mins, maxs = pts.min(axis=0), pts.max(axis=0)
    #     axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
    #     XI, YI, ZI = np.meshgrid(*axes, indexing='ij')
    #     G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0)
    #     return G.astype(self.dtype, copy=False), axes

    # def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
    #     Nf, ny, nx = self.intensity.shape
    #     z0,z1 = (0,Nf-1) if z_bound is None else z_bound
    #     y0,y1 = (0,ny-1) if y_bound is None else y_bound
    #     x0,x1 = (0,nx-1) if x_bound is None else x_bound
    #     Qc = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
    #     Hc = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
    #     Ic = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
    #     if in_place:
    #         self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
    #         return None
    #     return Qc, Hc, Ic

