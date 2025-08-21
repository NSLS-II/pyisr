# import os
# import numpy as np
# import pandas as pd
# import xrayutilities as xu
# import tifffile
# import vtk
# from vtk.util import numpy_support
# from scipy.interpolate import griddata


# def read_data(spec_path, data_dir, filename_pattern):
#     """
#     Reads a SPEC file and corresponding TIFF images, returning a DataFrame with columns ['Intensity', 'h', 'k', 'l'].
    
#     Parameters
#     ----------
#     spec_path : str
#         Path to the SPEC file.
#     data_dir : str
#         Directory containing TIFF images.
#     filename_pattern : str
#         Pattern for TIFF filenames, e.g., 'setup_6oct23_{scan:03d}_{data:03d}_data_000001.tiff'.
#         Use '{scan}' and '{data}' as placeholders.
    
#     Returns
#     -------
#     df : pd.DataFrame
#         DataFrame with columns ['Intensity', 'h', 'k', 'l'].
#     """
#     s = xu.io.SPECFile(spec_path)
#     records = []
#     for scan in s:
#         scan.ReadData()
#         scan_number = scan.nr
#         for i in range(len(scan.data)):
#             data_number = i
#             h = scan.data[i][1]
#             k = scan.data[i][2]
#             l = scan.data[i][3]
#             filename = filename_pattern.format(scan=int(scan_number), data=int(data_number))
#             tiff_path = os.path.join(data_dir, filename)
#             data = tifffile.imread(tiff_path)
#             record = {'Intensity': data, 'h': h, 'k': k, 'l': l}
#             records.append(record)
#     df = pd.DataFrame(records)
#     return df

# def hkl2q(lattice_params, hkl):
#     """
#     Convert HKL (fractional coordinates) into Cartesian q (Å⁻¹) using the reciprocal lattice.
    
#     The direct lattice vectors (in Å) are computed as:
#       a_vec = [a, 0, 0]
#       b_vec = [b*cos(gamma), b*sin(gamma), 0]
#       c_vec = [c*cos(beta),
#                c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma),
#                c*sqrt(1 - cos(beta)**2 - ((cos(alpha)-cos(beta)*cos(gamma))/sin(gamma))**2)]
    
#     The reciprocal lattice vectors are:
#       a* = 2π (b_vec x c_vec) / V, etc.
    
#     Parameters
#     ----------
#     lattice_params : dict
#         Dictionary with keys: 'a', 'b', 'c', 'alpha', 'beta', 'gamma'
#         (angles in degrees).
#     hkl : ndarray, shape (N,3)
#         Array of fractional coordinates.
    
#     Returns
#     -------
#     q : ndarray, shape (N,3)
#         Cartesian reciprocal space coordinates (Å⁻¹).
#     """
#     a = lattice_params['a']
#     b = lattice_params['b']
#     c = lattice_params['c']
#     alpha = np.deg2rad(lattice_params['alpha'])
#     beta  = np.deg2rad(lattice_params['beta'])
#     gamma = np.deg2rad(lattice_params['gamma'])
    
#     # Direct lattice vectors
#     a_vec = np.array([a, 0, 0])
#     b_vec = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
#     c_x = c*np.cos(beta)
#     c_y = c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
#     c_z = c*np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma))**2)
#     c_vec = np.array([c_x, c_y, c_z])
    
#     # Volume of the unit cell
#     V = np.dot(a_vec, np.cross(b_vec, c_vec))
    
#     # Reciprocal lattice vectors
#     a_star = 2*np.pi * np.cross(b_vec, c_vec) / V
#     b_star = 2*np.pi * np.cross(c_vec, a_vec) / V
#     c_star = 2*np.pi * np.cross(a_vec, b_vec) / V

#     # For each h,k,l compute q = h*a* + k*b* + l*c* (hkl is of shape (N,3))
#     q = hkl[:,0][:,None]*a_star + hkl[:,1][:,None]*b_star + hkl[:,2][:,None]*c_star
#     return q

# def build_3d_rsm_from_hkl(
#     df,
#     lattice_params,
#     grid_shape=(200, 200, 200),
#     method='histogram'
# ):
#     """
#     Build a 3D RSM directly from fractional (h,k,l) coordinates using a DataFrame input.
    
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame with columns ['Intensity', 'h', 'k', 'l'].
#         For each row, if 'Intensity' is multi-dimensional, it is summed.
#     lattice_params : dict
#         Dictionary with keys 'a', 'b', 'c', 'alpha', 'beta', 'gamma'.
#     grid_shape : tuple of int
#         Number of voxels in qx, qy, qz (default: (200,200,200)).
#     method : {'histogram','linear','nearest','cubic'}
#         'histogram' uses direct binning in q-space;
#         others use scipy.interpolate.griddata.
    
#     Returns
#     -------
#     I_grid : ndarray
#         3D volume in q-space of shape `grid_shape`.
#     """
#     # 1) Extract columns and convert Intensities to scalars
#     h_arr = df['h'].to_numpy(dtype=float)
#     k_arr = df['k'].to_numpy(dtype=float)
#     l_arr = df['l'].to_numpy(dtype=float)
#     intensities = np.array([np.sum(img) if hasattr(img, 'sum') else float(img)
#                             for img in df['Intensity']])
    
#     # 2) Convert HKL → Cartesian q (Å⁻¹) using our own conversion function
#     hkl = np.vstack((h_arr, k_arr, l_arr)).T  # shape (N,3)
#     q_xyz = hkl2q(lattice_params, hkl)          # shape (N,3)
#     qx, qy, qz = q_xyz[:,0], q_xyz[:,1], q_xyz[:,2]
    
#     # 3) Bin or interpolate in q-space
#     if method == 'histogram':
#         qx_edges = np.linspace(qx.min(), qx.max(), grid_shape[0]+1)
#         qy_edges = np.linspace(qy.min(), qy.max(), grid_shape[1]+1)
#         qz_edges = np.linspace(qz.min(), qz.max(), grid_shape[2]+1)
#         sum_I, _ = np.histogramdd((qx, qy, qz),
#                                   bins=(qx_edges, qy_edges, qz_edges),
#                                   weights=intensities)
#         count, _ = np.histogramdd((qx, qy, qz),
#                                   bins=(qx_edges, qy_edges, qz_edges))
#         I_grid = np.divide(sum_I, count, out=np.zeros_like(sum_I), where=count>0)
#     else:
#         # Define regular q-space grid for interpolation
#         qx_lin = np.linspace(qx.min(), qx.max(), grid_shape[0])
#         qy_lin = np.linspace(qy.min(), qy.max(), grid_shape[1])
#         qz_lin = np.linspace(qz.min(), qz.max(), grid_shape[2])
#         QX, QY, QZ = np.meshgrid(qx_lin, qy_lin, qz_lin, indexing='ij')
#         I_grid = griddata((qx, qy, qz),
#                           intensities,
#                           (QX, QY, QZ),
#                           method=method,
#                           fill_value=0)
#     return I_grid



# def build_3d_rsm(df: pd.DataFrame,
#                  nh: int = 100, nk: int = 100, nl: int = 100,
#                  method: str = 'linear'):
#     """
#     Interpolate scattered (h, k, l, Intensity) → regular 3D HKL volume.
#     The output grid is centered at (0, 0, 0), e.g. for nh = nk = nl = 100,
#     the grid ranges from -50 to 50 for each coordinate.
#     """
#     if method not in ('linear', 'nearest'):
#         raise ValueError("method must be 'linear' or 'nearest' for 3D")

#     h = df['h'].to_numpy()
#     k = df['k'].to_numpy()
#     l = df['l'].to_numpy()
#     I = np.array([np.sum(img) if hasattr(img, 'sum') else float(img)
#                   for img in df['Intensity']])

#     # Build symmetric grid centered at 0
#     hl = np.linspace(-nh/2, nh/2, nh)
#     kl = np.linspace(-nk/2, nk/2, nk)
#     ll = np.linspace(-nl/2, nl/2, nl)
#     Hg, Kg, Lg = np.meshgrid(hl, kl, ll, indexing='ij')

#     # Interpolate scattered measurements onto regular grid
#     sample_pts = np.column_stack((h, k, l))
#     grid_pts   = np.column_stack((Hg.ravel(), Kg.ravel(), Lg.ravel()))
#     RSM_flat = griddata(sample_pts, I, grid_pts, method=method, fill_value=0)
#     RSM = RSM_flat.reshape((nh, nk, nl))

#     return Hg, Kg, Lg, RSM

# # ──────────────────────────────────────────────────────────────────────────────
# # 2) Convert HKL grid → Q-space grid (orthogonal cell)
# # ──────────────────────────────────────────────────────────────────────────────
# def convert_hkl_to_Q(Hg, Kg, Lg, a: float, b: float, c: float):
#     """
#     Qx = 2π·H/a, Qy = 2π·K/b, Qz = 2π·L/c
#     """
#     factor = 2 * np.pi
#     Qx = factor * Hg / a
#     Qy = factor * Kg / b
#     Qz = factor * Lg / c
#     return Qx, Qy, Qz

# # ──────────────────────────────────────────────────────────────────────────────
# # 3) Convert 2D detector pixels → Q-space map for a single frame
# # ──────────────────────────────────────────────────────────────────────────────
# def pixel_to_Q_map(D: float,
#                    p: float,
#                    m0: float, n0: float,
#                    Mx: int, Ny: int,
#                    wl: float = None,
#                    energy_ev: float = None,
#                    phi_d: float = 0.0,
#                    th0: float = 0.0,
#                    dth: float = 0.0,
#                    scan_index: int = 0):
#     """
#     Returns Qx,Qy,Qz,Qmag arrays of shape (Ny, Mx).
#     """
#     # wavelength
#     if wl is None:
#         if energy_ev is None:
#             raise ValueError("Supply wl or energy_ev")
#         wl = 12.3984196 / (energy_ev/1000.0)

#     # pixel coords in detector plane
#     xs = (np.arange(Mx) - n0) * p
#     ys = (np.arange(Ny) - m0) * p
#     X, Y = np.meshgrid(xs, ys)

#     k0 = 2*np.pi / wl
#     R = np.sqrt(X**2 + Y**2 + D**2)

#     # k_out in detector frame
#     kout_x = k0 * (X / R)
#     kout_y = k0 * (Y / R)
#     kout_z = k0 * (D / R)
#     # Q in detector frame
#     Qx_det = kout_x
#     Qy_det = kout_y
#     Qz_det = kout_z - k0

#     # rotations: φ_d then θ = th0 + scan_index*dth, about detector Y-axis
#     phi   = np.deg2rad(phi_d)
#     theta = np.deg2rad(th0 + scan_index*dth)
#     R_phi = np.array([[ np.cos(phi), 0, -np.sin(phi)],
#                       [          0., 1,           0.],
#                       [ np.sin(phi), 0,  np.cos(phi)]])
#     R_th  = np.array([[ np.cos(theta), 0, -np.sin(theta)],
#                       [            0., 1,            0.],
#                       [ np.sin(theta), 0,  np.cos(theta)]])
#     R_comb = R_th.dot(R_phi)

#     # apply rotation
#     pts = np.stack([Qx_det.ravel(), Qy_det.ravel(), Qz_det.ravel()], axis=0)
#     Qs  = R_comb.dot(pts)
#     Qx = Qs[0].reshape((Ny, Mx))
#     Qy = Qs[1].reshape((Ny, Mx))
#     Qz = Qs[2].reshape((Ny, Mx))
#     Qmag = np.sqrt(Qx**2 + Qy**2 + Qz**2)

#     return Qx, Qy, Qz, Qmag

# # ──────────────────────────────────────────────────────────────────────────────
# # 4) Save a 3D volume in Q-space to a legacy VTK file (ParaView)
# # ──────────────────────────────────────────────────────────────────────────────
# def save_to_vtk(Qx, Qy, Qz, RSM, filename="rsm_Q.vtk"):
#     """
#     Writes RSM (nh×nk×nl) sampled at Qx,Qy,Qz into a .vtk
#     """
#     # assume RSM.shape = (nx, ny, nz) = Qx.shape
#     nx, ny, nz = RSM.shape

#     origin = (Qx.min(), Qy.min(), Qz.min())
#     spacing = (Qx[1,0,0] - Qx[0,0,0],
#                Qy[0,1,0] - Qy[0,0,0],
#                Qz[0,0,1] - Qz[0,0,0])

#     img = vtk.vtkImageData()
#     img.SetDimensions(nx, ny, nz)
#     img.SetOrigin(*origin)
#     img.SetSpacing(*spacing)

#     flat = RSM.flatten(order='F').astype(np.float32)
#     vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
#     vtk_arr.SetName("Intensity")
#     img.GetPointData().SetScalars(vtk_arr)

#     writer = vtk.vtkStructuredPointsWriter()
#     writer.SetFileName(filename)
#     writer.SetInputData(img)
#     writer.Write()
#     print(f"Saved VTK → {filename}")

# # ──────────────────────────────────────────────────────────────────────────────
# # 2) Compute per‐pixel Q_det in the sample frame
# # ──────────────────────────────────────────────────────────────────────────────
# def compute_Qdet(Ny, Mx, D, p, m0, n0, wl, phi_d, two_theta):
#     xs = (np.arange(Mx) - n0) * p
#     ys = (np.arange(Ny) - m0) * p
#     X, Y = np.meshgrid(xs, ys)           # (Ny, Mx)
#     k0 = 2*np.pi / wl
#     R  = np.sqrt(X**2 + Y**2 + D**2)
#     kout = np.stack([k0*(X/R), k0*(Y/R), k0*(D/R)], axis=-1)  # (Ny, Mx, 3)
#     Qdet = kout - np.array([0.0, 0.0, k0])                    # subtract k_in
#     φ = np.deg2rad(phi_d)
#     θ = np.deg2rad(two_theta)
#     Rφ = np.array([[ np.cos(φ), 0, -np.sin(φ)],
#                    [        0., 1,         0. ],
#                    [ np.sin(φ), 0,  np.cos(φ)]])
#     Rθ = np.array([[ np.cos(θ), 0, -np.sin(θ)],
#                    [        0., 1,         0. ],
#                    [ np.sin(θ), 0,  np.cos(θ)]])
#     Rrot = Rθ.dot(Rφ)
#     flat = Qdet.reshape(-1, 3).T
#     Qs = (Rrot @ flat).T
#     return Qs.reshape(Ny, Mx, 3)

# # ──────────────────────────────────────────────────────────────────────────────
# # 3) Collect all scattered Q + I samples
# # ──────────────────────────────────────────────────────────────────────────────
# def collect_QI(frames, HKLs, UB, geom):
#     D, p, m0, n0 = geom['D'], geom['p'], geom['m0'], geom['n0']
#     wl           = geom['wl']
#     phi_d        = geom.get('phi_d', 0.0)
#     th0          = geom.get('th0', 0.0)
#     dth          = geom.get('dth', 0.0)
#     Ny, Mx = frames[0].shape
#     Q_list, I_list = [], []
#     for i, (frame, (H_i, K_i, L_i)) in enumerate(zip(frames, HKLs)):
#         two_theta = th0 + i*dth
#         Qdet = compute_Qdet(Ny, Mx, D, p, m0, n0, wl, phi_d, two_theta)
#         Q0 = UB.dot(np.array([H_i, K_i, L_i]))
#         Qtot = Qdet + Q0[np.newaxis, np.newaxis, :]
#         Q_list.append(Qtot.reshape(-1, 3))
#         I_list.append(frame.ravel())
#     Q_all = np.vstack(Q_list)
#     I_all = np.hstack(I_list)
#     return Q_all, I_all

# # ──────────────────────────────────────────────────────────────────────────────
# # 4) Grid into a 3D RSM via nearest‐neighbor interpolation
# # ──────────────────────────────────────────────────────────────────────────────
# def grid_RSM(Q_all, I_all, nq=(256,256,256)):
#     nx, ny, nz = nq
#     qx = np.linspace(Q_all[:,0].min(), Q_all[:,0].max(), nx)
#     qy = np.linspace(Q_all[:,1].min(), Q_all[:,1].max(), ny)
#     qz = np.linspace(Q_all[:,2].min(), Q_all[:,2].max(), nz)
#     Qxg, Qyg, Qzg = np.meshgrid(qx, qy, qz, indexing='ij')
#     pts_grid = np.column_stack((Qxg.ravel(), Qyg.ravel(), Qzg.ravel()))
#     interp = NearestNDInterpolator(Q_all, I_all)
#     R_flat = interp(pts_grid)
#     RSM    = R_flat.reshape(nq)
#     return Qxg, Qyg, Qzg, RSM

# # ──────────────────────────────────────────────────────────────────────────────


# #!/usr/bin/env python3
# """
# read_rsm_pipeline.py

# Combines SPEC metadata parsing with TIFF intensity loading to build
# 3D Reciprocal Space Maps (RSM).

# Usage:
#     python read_rsm_pipeline.py spec_file.spec tiff_directory [--dask]
# """
# import sys
# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import griddata

# # import existing pipeline and data loader
# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData



# def pixel2q(i, j, setup):
#     """
#     Convert detector pixel indices (i, j) to lab‐frame Q vectors,
#     automatically subtracting the beam‐center and using the
#     setup’s distance and wavelength.

#     Parameters:
#         i, j   : 2D arrays of pixel column (x) and row (y) indices
#         setup  : ExperimentSetup instance with attributes
#                  xcenter, ycenter, pitch (m), distance (m), wavelength (m)

#     Returns:
#         Q_lab  : array of shape (Ny, Mx, 3) giving (Qx, Qy, Qz) at each pixel
#     """
#     # physical coordinates relative to beam center
#     x = (i - setup.xcenter) * setup.pitch
#     y = (j - setup.ycenter) * setup.pitch

#     D  = setup.distance
#     wl = setup.wavelength
#     k0 = 2 * np.pi / wl

#     two_theta = np.arctan2(np.hypot(x, y), D)
#     eta       = np.arctan2(y, x)

#     sin2t = np.sin(2 * two_theta)
#     kfx   = k0 * sin2t * np.cos(eta)
#     kfy   = k0 * sin2t * np.sin(eta)
#     kfz   = k0 * np.cos(2 * two_theta)

#     # Q_lab = kf - ki, with ki = [0,0,k0]
#     Qx = kfx
#     Qy = kfy
#     Qz = kfz - k0

#     return np.stack((Qx, Qy, Qz), axis=-1)


# def q2hkl(Q_samp, UB):
#     """
#     Convert sample-frame Q vectors to fractional (h, k, l) indices via the UB matrix.

#     Parameters:
#         Q_samp : array (...,3) of Q vectors in the sample frame
#         UB     : 3×3 orientation matrix

#     Returns:
#         hkl    : array of same shape as Q_samp giving the (h, k, l) coordinates
#     """
#     UB_inv = np.linalg.inv(UB)
#     flat   = Q_samp.reshape(-1, 3)
#     hkl    = flat @ UB_inv.T
#     return hkl.reshape(Q_samp.shape)


# class RSMBuilder:
#     """
#     Builds per-pixel Q, HKL, and intensity arrays from SPEC and TIFF data.

#     Attributes:
#         exp      : ExperimentData instance
#         df       : merged DataFrame of metadata + intensity arrays
#         Q_lab    : (Ny, Mx, 3) lab-frame q-grid
#         UB       : (3,3) orientation matrix
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False, process_hklscan_only=False):
#         # load metadata
#         self.exp = SpecParser(spec_file)
#         df_meta = self.exp.to_pandas()
#         # ensure scan_number and data_number are ints for merging
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)
#         # load intensity frames
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()
#         # merge on scan_number, data_number
#         self.df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
#         # prep q-grid
#            # If processing only hklscan data, filter by type column (case-insensitive)
#         if process_hklscan_only:
#             self.df = self.df[self.df['type'].str.lower() == 'hklscan']
#         s = self.exp.setup
#         n = np.arange(s.xpixels)
#         m = np.arange(s.ypixels)
#         N, M = np.meshgrid(n, m)
#         x = (N - s.xcenter) * s.pitch
#         y = (M - s.ycenter) * s.pitch
#         self.Q_lab = pixel2q(x, y, s)
#         self.UB = self.exp.crystal.UB

#     def compute_full(self):
#         """
#         Compute per-frame Q_samp, HKL indices, and intensities for each pixel.

#         Returns:
#           Q_samp    : (Nf, Ny, Mx, 3) sample-frame q-vectors
#           hkl       : (Nf, Ny, Mx, 3) fractional indices
#           intensity : (Nf, Ny, Mx) pixel intensities
#         """
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'][0].shape
#         Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl       = np.zeros_like(Q_samp)
#         intensity = np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             I = row['intensity']
#             # get all angles per frame
#             phi   = row['phi']     # detector azimuth (horizontal 2θ offset)
#             chi   = row['chi']     # detector tilt
#             tth   = row['tth']     # detector two-theta
#             th    = row['th']      # sample Bragg theta
#             # include any global offsets from setup
#             phi_tot = phi + self.exp.setup.phi  # global horizontal offset
#             tth_tot = tth + self.exp.setup.theta  # global two-theta offset
#             th_tot  = th  # could include sample theta0 offset if defined

#             # build detector rotation: z (phi_tot), x (chi), then y (tth_tot)
#             rot_det = R.from_euler('zxy', [phi_tot, chi, tth_tot], degrees=True).as_matrix()
#             # build sample tilt rotation about x-axis by theta
#             rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
#             # full rotation: sample tilt then detector orientation
#             rot_full = rot_samp @ rot_det

#             # rotate lab-frame Q into sample frame
#             Qs = self.Q_lab @ rot_full.T
#             Q_samp[i]    = Qs
#             hkl[i]       = q2hkl(Qs, self.UB)
#             intensity[i] = I
#         self.Q_samp = Q_samp
#         self.hkl = hkl
#         self.intensity = intensity
        

#         return Q_samp, hkl, intensity
    
#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Define uniform Q-space grid for regridding intensities.

#         grid_ranges: tuple of 3 (min, max) ranges for qx, qy, qz
#         grid_shape : tuple of 3 ints (nx, ny, nz)
#         """
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def regrid_intensity(self, method='sum'):
#         """
#         Bin intensities onto the defined uniform Q-space grid.

#         method: 'sum' (default) or 'mean'
#         """
#         if not hasattr(self, 'Q_samp') or not hasattr(self, 'intensity'):
#             raise RuntimeError("Must call compute_full() before regrid_intensity()")
#         if not hasattr(self, 'edges'):
#             raise RuntimeError("Must define grid with setup_grid() before regrid_intensity()")

#         points = self.Q_samp.reshape(-1, 3)
#         values = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(points, bins=self.edges, weights=values)

#         if method == 'sum':
#             rsm = H_sum
#         elif method == 'mean':
#             H_count, _ = np.histogramdd(points, bins=self.edges)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 rsm = H_sum / H_count
#                 rsm[np.isnan(rsm)] = 0
#         else:
#             raise ValueError("method must be 'sum' or 'mean'")

#         return rsm, self.edges


# def crop_by_positions(Q_samp, hkl, intensity, y_bound, x_bound):
#     """
#     Crop Q_samp, hkl, and intensity arrays given explicit y and x boundaries.

#     Parameters:
#         Q_samp    : ndarray, shape (Nframes, Ny, Mx, 3)
#         hkl       : ndarray, shape (Nframes, Ny, Mx, 3)
#         intensity : ndarray, shape (Nframes, Ny, Mx)
#         y_bound   : tuple (y0, y1) inclusive row boundaries
#         x_bound   : tuple (x0, x1) inclusive column boundaries

#     Returns:
#         Q_crop    : ndarray, cropped Q_samp (Nframes, y1-y0+1, x1-x0+1, 3)
#         hkl_crop  : ndarray, cropped hkl
#         I_crop    : ndarray, cropped intensity
#         bounds    : (y_bound, x_bound)
#     """
#     y0, y1 = y_bound
#     x0, x1 = x_bound

#     # validate boundaries
#     Ny = intensity.shape[1]
#     Mx = intensity.shape[2]
#     if not (0 <= y0 <= y1 < Ny and 0 <= x0 <= x1 < Mx):
#         raise ValueError(f"Bounds out of range: y[{y0},{y1}] x[{x0},{x1}], array shape {intensity.shape}")

#     Q_crop   = Q_samp[:, y0:y1+1, x0:x1+1, :]
#     hkl_crop = hkl[:, y0:y1+1, x0:x1+1, :]
#     I_crop   = intensity[:, y0:y1+1, x0:x1+1]

#     return Q_crop, hkl_crop, I_crop

def crop_by_positions(Q_samp, hkl, intensity, z_bound, y_bound, x_bound):
    """
    Crop Q_samp, hkl, and intensity arrays given explicit z, y, and x boundaries.

    Parameters:
        Q_samp    : ndarray, shape (Nframes, Ny, Mx, 3)
        hkl       : ndarray, shape (Nframes, Ny, Mx, 3)
        intensity : ndarray, shape (Nframes, Ny, Mx)
        z_bound   : tuple (z0, z1) inclusive frame boundaries
        y_bound   : tuple (y0, y1) inclusive row boundaries
        x_bound   : tuple (x0, x1) inclusive column boundaries

    Returns:
        Q_crop    : ndarray, cropped Q_samp of shape (z1-z0+1, y1-y0+1, x1-x0+1, 3)
        hkl_crop  : ndarray, cropped hkl of same shape
        I_crop    : ndarray, cropped intensity of shape (z1-z0+1, y1-y0+1, x1-x0+1)
        bounds    : (z_bound, y_bound, x_bound)
    """
    z0, z1 = z_bound
    y0, y1 = y_bound
    x0, x1 = x_bound

    # Validate boundaries for each dimension
    Nframes, Ny, Mx = intensity.shape
    if not (0 <= z0 <= z1 < Nframes):
        raise ValueError(f"z_bound out of range: z[{z0},{z1}], number of frames {Nframes}")
    if not (0 <= y0 <= y1 < Ny):
        raise ValueError(f"y_bound out of range: y[{y0},{y1}], array shape {intensity.shape}")
    if not (0 <= x0 <= x1 < Mx):
        raise ValueError(f"x_bound out of range: x[{x0},{x1}], array shape {intensity.shape}")

    Q_crop   = Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
    hkl_crop = hkl[z0:z1+1, y0:y1+1, x0:x1+1, :]
    I_crop   = intensity[z0:z1+1, y0:y1+1, x0:x1+1]

    return Q_crop, hkl_crop, I_crop



# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import griddata

# # Import your spec parser and data loader modules
# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# def pixel2q(i, j, setup):
#     # ... existing function code ...
#     x = (i - setup.xcenter) * setup.pitch
#     y = (j - setup.ycenter) * setup.pitch
#     D  = setup.distance
#     wl = setup.wavelength
#     k0 = 2 * np.pi / wl
#     two_theta = np.arctan2(np.hypot(x, y), D)
#     eta       = np.arctan2(y, x)
#     sin2t = np.sin(2 * two_theta)
#     kfx   = k0 * sin2t * np.cos(eta)
#     kfy   = k0 * sin2t * np.sin(eta)
#     kfz   = k0 * np.cos(2 * two_theta)
#     Qx = kfx
#     Qy = kfy
#     Qz = kfz - k0
#     return np.stack((Qx, Qy, Qz), axis=-1)

# def q2hkl(Q_samp, UB):
#     UB_inv = np.linalg.inv(UB)
#     flat   = Q_samp.reshape(-1, 3)
#     hkl    = flat @ UB_inv.T
#     return hkl.reshape(Q_samp.shape)

# class RSMBuilder:
#     """
#     Builds per-pixel Q, HKL, and intensity arrays from SPEC and TIFF data.

#     Attributes:
#         exp      : ExperimentData instance
#         df       : Merged DataFrame of metadata and intensity arrays.
#         Q_lab    : (Ny, Mx, 3) Lab-frame q-grid.
#         UB       : (3,3) Orientation matrix.
    
#     New Parameter:
#         process_hklscan_only: bool - if True, only process hklscan rows.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False, process_hklscan_only=False):
#         # load metadata
#         self.exp = SpecParser(spec_file)
#         df_meta = self.exp.to_pandas()
#         # ensure scan_number and data_number are ints for merging
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)
#         # load intensity frames
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()
#         # merge on scan_number, data_number
#         self.df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
        
#         # If processing only hklscan data, filter by type column (case-insensitive)
#         if process_hklscan_only:
#             self.df = self.df[self.df['type'].str.lower() == 'hklscan']
        
#         # prepare Q-grid: 
#         s = self.exp.setup
#         n = np.arange(s.xpixels)
#         m = np.arange(s.ypixels)
#         N, M = np.meshgrid(n, m)
#         x = (N - s.xcenter) * s.pitch
#         y = (M - s.ycenter) * s.pitch
#         self.Q_lab = pixel2q(x, y, s)
#         self.UB = self.exp.crystal.UB

#     def compute_full(self):
#         """
#         Compute per-frame Q_samp, HKL indices, and intensities for each pixel.

#         Returns:
#           Q_samp    : (Nf, Ny, Mx, 3) Sample-frame q-vectors.
#           hkl       : (Nf, Ny, Mx, 3) Fractional indices.
#           intensity : (Nf, Ny, Mx) Pixel intensities.
#         """
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'].iloc[0].shape
#         Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl       = np.zeros_like(Q_samp)
#         intensity = np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             I = row['intensity']
#             phi   = row['phi']     # detector azimuth (horizontal 2θ offset)
#             chi   = row['chi']     # detector tilt
#             tth   = row['tth']     # detector two-theta
#             th    = row['th']      # sample Bragg theta

#             phi_tot = phi + self.exp.setup.phi  # global horizontal offset
#             tth_tot = tth + self.exp.setup.theta  # global two-theta offset
#             th_tot  = th

#             rot_det = R.from_euler('zxy', [phi_tot, chi, tth_tot], degrees=True).as_matrix()
#             rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
#             rot_full = rot_samp @ rot_det

#             Qs = self.Q_lab @ rot_full.T
#             Q_samp[i]    = Qs
#             hkl[i]       = q2hkl(Qs, self.UB)
#             intensity[i] = I

#         self.Q_samp = Q_samp
#         self.hkl = hkl
#         self.intensity = intensity

#         return Q_samp, hkl, intensity

#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Define a uniform Q-space grid for regridding intensities.

#         Parameters:
#           grid_ranges: Tuple of 3 (min, max) ranges for qx, qy, qz.
#           grid_shape : Tuple of 3 ints (nx, ny, nz).
#         """
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def regrid_intensity(self, method='sum'):
#         """
#         Bin intensities onto the defined uniform Q-space grid.

#         method: 'sum' (default) or 'mean'.
#         """
#         if not hasattr(self, 'Q_samp') or not hasattr(self, 'intensity'):
#             raise RuntimeError("Must call compute_full() before regrid_intensity()")
#         if not hasattr(self, 'edges'):
#             raise RuntimeError("Must define grid with setup_grid() before regrid_intensity()")

#         points = self.Q_samp.reshape(-1, 3)
#         values = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(points, bins=self.edges, weights=values)

#         if method == 'sum':
#             rsm = H_sum
#         elif method == 'mean':
#             H_count, _ = np.histogramdd(points, bins=self.edges)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 rsm = H_sum / H_count
#                 rsm[np.isnan(rsm)] = 0
#         else:
#             raise ValueError("method must be 'sum' or 'mean'")

#         return rsm, self.edges


# class RSMBuilder:
#     """
#     Builds per-pixel Q_samp, HKL, and intensity arrays from SPEC + TIFF data.

#     New Parameter:
#         process_hklscan_only: bool - if True, only process 'hklscan' entries.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None,
#                  use_dask=False, process_hklscan_only=False):
#         # 1) load metadata + setup + crystal
#         exp = SpecParser(spec_file)
#         self.setup   = exp.setup
#         self.UB      = exp.crystal.UB

#         # 2) get scan metadata
#         df_meta = exp.to_pandas()
#         # force ints
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         # 3) load TIFF frames
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()
#         # df_int has columns scan_number, data_number, intensity (2D array)

#         # 4) merge
#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])

#         # 5) optionally filter only hklscans
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan'].reset_index(drop=True)

#         self.df = df

#         # 6) build static Q_lab grid from pixel indices
#         s = self.setup
#         cols = np.arange(s.xpixels)
#         rows = np.arange(s.ypixels)
#         N, M = np.meshgrid(cols, rows)           # N: x indices, M: y indices
#         self.Q_lab = pixel2q(N, M, s)            # shape (Ny, Mx, 3)

#     def compute_full(self):
#         """
#         Compute per-frame Q_samp, hkl, and intensity arrays.

#         Returns:
#           Q_samp    : ndarray (Nf, Ny, Mx, 3)
#           hkl       : ndarray (Nf, Ny, Mx, 3)
#           intensity : ndarray (Nf, Ny, Mx)
#         """
#         df = self.df
#         Nf = len(df)
#         print(Nf, "frames to process")
#         Ny, Mx = df['intensity'].iat[0].shape

#         Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr   = np.zeros_like(Q_samp)
#         intens_arr= np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             print(df['scan_number'].iat[i], df['data_number'].iat[i], end='\r')
#             I   = row['intensity']
#             phi = row['phi']   # detector azimuth
#             chi = row['chi']   # tilt
#             tth = row['tth']   # two-theta
#             th  = row['th']    # sample theta

#             # incorporate global offsets if any (e.g. phi_d, th0)
#             phi_tot = phi      + self.setup.phi
#             tth_tot = tth      + self.setup.theta
#             th_tot  = th       # add sample-theta0 here if defined

#             # build rotations
#             rot_det  = R.from_euler('zxy', [phi_tot, chi, tth_tot],
#                                     degrees=True).as_matrix()
#             rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
#             rot_full = rot_samp @ rot_det

#             # rotate Q_lab into sample frame
#             Qs = self.Q_lab @ rot_full.T

#             Q_samp[i]     = Qs
#             hkl_arr[i]    = q2hkl(Qs, self.UB)
#             intens_arr[i] = I

#         # store for later
#         self.Q_samp    = Q_samp
#         self.hkl       = hkl_arr
#         self.intensity = intens_arr

#         return Q_samp, hkl_arr, intens_arr

#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Define uniform Q-space grid for regridding.
#         """
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def setup_hkl_grid(self, hkl_ranges, grid_shape):
#         """
#         Define uniform HKL-space grid for regridding.
#         """
#         self.hkl_ranges = hkl_ranges
#         self.hkl_shape  = grid_shape
#         self.hkl_edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(hkl_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         """
#         Bin intensities onto the defined uniform grid ('q' or 'hkl').

#         Returns:
#             rsm   : 3D array of binned intensities
#             edges : list of bin-edge arrays
#         """
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid() first.")
#             pts   = self.Q_samp.reshape(-1,3)
#             edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid() first.")
#             pts   = self.hkl.reshape(-1,3)
#             edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

#         if method == 'sum':
#             return H_sum, edges
#         # for mean:
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_rsm = H_sum / H_cnt
#             mean_rsm[np.isnan(mean_rsm)] = 0
#         return mean_rsm, edges


import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata

from rsm3d.spec_parser import SpecParser
from rsm3d.data_io import ReadData

def pixel2q(i, j, setup):
    x = (i - setup.xcenter) * setup.pitch
    y = (j - setup.ycenter) * setup.pitch
    D  = setup.distance
    wl = setup.wavelength
    k0 = 2 * np.pi / wl
    two_theta = np.arctan2(np.hypot(x, y), D)
    eta       = np.arctan2(y, x)
    sin2t = np.sin(2 * two_theta)
    kfx   = k0 * sin2t * np.cos(eta)
    kfy   = k0 * sin2t * np.sin(eta)
    kfz   = k0 * np.cos(2 * two_theta)
    Qx = kfx
    Qy = kfy
    Qz = kfz - k0
    return np.stack((Qx, Qy, Qz), axis=-1)

def q2hkl(Q_samp, UB):
    UB_inv = np.linalg.inv(UB)
    flat   = Q_samp.reshape(-1, 3)
    hkl    = flat @ UB_inv.T
    return hkl.reshape(Q_samp.shape)

class RSMBuilder:
    """
    Builds per-pixel Q_samp, HKL, and intensity arrays from SPEC + TIFF data.

    Parameters:
        spec_file         : Path to the SPEC file.
        tiff_dir          : Directory containing TIFF images.
        pattern           : Optional regex pattern to match TIFF filenames.
        use_dask          : If True, use Dask for loading data.
        process_hklscan_only : If True, process only rows with type 'hklscan'.
        selected_scans    : Optional iterable of scan_numbers to process.
                            Only entries whose 'scan_number' is in this list will be kept.
    """
    def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
                 process_hklscan_only=False, selected_scans=None):
        # 1) load metadata + setup + crystal
        exp = SpecParser(spec_file)
        self.setup   = exp.setup
        self.UB      = exp.crystal.UB

        # 2) get scan metadata and force ints for merging
        df_meta = exp.to_pandas()
        df_meta['scan_number'] = df_meta['scan_number'].astype(int)
        df_meta['data_number'] = df_meta['data_number'].astype(int)

        # 3) load TIFF frames
        rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
        df_int = rd.load_data()

        # 4) merge metadata and intensity on ['scan_number','data_number']
        df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
        
        # 5) optionally filter only hklscan data
        if process_hklscan_only:
            df = df[df['type'].str.lower() == 'hklscan']
        
        # 6) optionally filter to only selected scans
        if selected_scans is not None:
            df = df[df['scan_number'].isin(selected_scans)].reset_index(drop=True)

        self.df = df

        # 7) build static Q_lab grid from pixel indices
        s = self.setup
        cols = np.arange(s.xpixels)
        rows = np.arange(s.ypixels)
        N, M = np.meshgrid(cols, rows)  # N: x indices, M: y indices
        self.Q_lab = pixel2q(N, M, s)    # shape (Ny, Mx, 3)

    def compute_full(self):
        """
        Compute per-frame Q_samp, hkl, and intensity arrays.

        Returns:
          Q_samp    : ndarray (Nf, Ny, Mx, 3) sample-frame Q vectors.
          hkl       : ndarray (Nf, Ny, Mx, 3) fractional indices.
          intensity : ndarray (Nf, Ny, Mx) pixel intensities.
        """
        df = self.df
        Nf = len(df)
        # print(Nf, "frames to process")
        Ny, Mx = df['intensity'].iat[0].shape

        Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
        hkl_arr   = np.zeros_like(Q_samp)
        intens_arr= np.zeros((Nf, Ny, Mx), dtype=float)

        for i, row in df.iterrows():
            print(df['scan_number'].iat[i], df['data_number'].iat[i], end='\r')
            I   = row['intensity']
            phi = row['phi']
            chi = row['chi']
            tth = row['tth']
            th  = row['th']

            phi_tot = phi + self.setup.phi
            tth_tot = tth + self.setup.theta
            th_tot  = th

            rot_det  = R.from_euler('zxy', [phi_tot, chi, tth_tot], degrees=True).as_matrix()
            rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
            rot_full = rot_samp @ rot_det

            Qs = self.Q_lab @ rot_full.T

            Q_samp[i]     = Qs
            UB = row['ub'] if 'ub' in row and row['ub'] is not None else self.UB
            # print(UB)
            hkl_arr[i]    = q2hkl(Qs, UB)
            intens_arr[i] = I

        self.Q_samp    = Q_samp
        self.hkl       = hkl_arr
        self.intensity = intens_arr

        return Q_samp, hkl_arr, intens_arr

    def setup_grid(self, grid_ranges, grid_shape):
        """
        Define a uniform Q-space grid for regridding.

        Parameters:
            grid_ranges: Tuple of 3 (min, max) ranges for Qx, Qy, Qz.
            grid_shape : Tuple of 3 ints (nx, ny, nz).
        """
        self.grid_ranges = grid_ranges
        self.grid_shape  = grid_shape
        self.edges = [
            np.linspace(r[0], r[1], grid_shape[i] + 1)
            for i, r in enumerate(grid_ranges)
        ]

    def setup_hkl_grid(self, hkl_ranges, grid_shape):
        """
        Define a uniform HKL-space grid for regridding.

        Parameters:
            hkl_ranges: Tuple of 3 (min, max) ranges for h, k, l.
            grid_shape : Tuple of 3 ints (nx, ny, nz).
        """
        self.hkl_ranges = hkl_ranges
        self.hkl_shape  = grid_shape
        self.hkl_edges = [
            np.linspace(r[0], r[1], grid_shape[i] + 1)
            for i, r in enumerate(hkl_ranges)
        ]

    def regrid_intensity(self, method='sum', space='q'):
        """
        Bin intensities onto the defined uniform grid (either 'q' or 'hkl').

        Parameters:
            method: 'sum' (default) or 'mean'
            space : 'q' to use Q_samp or 'hkl' to use hkl

        Returns:
            rsm   : 3D array of binned intensities.
            edges : List of bin-edge arrays.
        """
        if not hasattr(self, 'Q_samp'):
            raise RuntimeError("Call compute_full() first.")
        if space == 'q':
            if not hasattr(self, 'edges'):
                raise RuntimeError("Call setup_grid() first.")
            pts   = self.Q_samp.reshape(-1, 3)
            edges = self.edges
        elif space == 'hkl':
            if not hasattr(self, 'hkl_edges'):
                raise RuntimeError("Call setup_hkl_grid() first.")
            pts   = self.hkl.reshape(-1, 3)
            edges = self.hkl_edges
        else:
            raise ValueError("space must be 'q' or 'hkl'")

        vals = self.intensity.ravel()
        H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

        if method == 'sum':
            return H_sum, edges
        H_cnt, _ = np.histogramdd(pts, bins=edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_rsm = H_sum / H_cnt
            mean_rsm[np.isnan(mean_rsm)] = 0
        return mean_rsm, edges
    def regrid_auto(self, space='q', grid_shape=(200, 200, 200), method='mean'):
        """
        Auto-compute grid ranges from Q_samp or hkl and regrid.

        Parameters:
            space: 'q' or 'hkl'
            grid_shape: (nx, ny, nz)
            method: 'sum' or 'mean'
        """
        if not hasattr(self, 'Q_samp'):
            raise RuntimeError("Call compute_full() first.")

        if space == 'q':
            arr = self.Q_samp
            ranges = (
                (arr[..., 0].min(), arr[..., 0].max()),
                (arr[..., 1].min(), arr[..., 1].max()),
                (arr[..., 2].min(), arr[..., 2].max()),
            )
            self.setup_grid(ranges, grid_shape)
        elif space == 'hkl':
            if not hasattr(self, 'hkl'):
                raise RuntimeError("HKL not available. Call compute_full().")
            arr = self.hkl
            ranges = (
                (arr[..., 0].min(), arr[..., 0].max()),
                (arr[..., 1].min(), arr[..., 1].max()),
                (arr[..., 2].min(), arr[..., 2].max()),
            )
            self.setup_hkl_grid(ranges, grid_shape)
        else:
            raise ValueError("space must be 'q' or 'hkl'")

        return self.regrid_intensity(method=method, space=space)
