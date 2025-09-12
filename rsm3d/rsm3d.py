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

# def crop_by_positions(Q_samp, hkl, intensity, z_bound, y_bound, x_bound):
    # """
    # Crop Q_samp, hkl, and intensity arrays given explicit z, y, and x boundaries.

    # Parameters:
    #     Q_samp    : ndarray, shape (Nframes, Ny, Mx, 3)
    #     hkl       : ndarray, shape (Nframes, Ny, Mx, 3)
    #     intensity : ndarray, shape (Nframes, Ny, Mx)
    #     z_bound   : tuple (z0, z1) inclusive frame boundaries
    #     y_bound   : tuple (y0, y1) inclusive row boundaries
    #     x_bound   : tuple (x0, x1) inclusive column boundaries

    # Returns:
    #     Q_crop    : ndarray, cropped Q_samp of shape (z1-z0+1, y1-y0+1, x1-x0+1, 3)
    #     hkl_crop  : ndarray, cropped hkl of same shape
    #     I_crop    : ndarray, cropped intensity of shape (z1-z0+1, y1-y0+1, x1-x0+1)
    #     bounds    : (z_bound, y_bound, x_bound)
    # """
    # z0, z1 = z_bound
    # y0, y1 = y_bound
    # x0, x1 = x_bound

    # # Validate boundaries for each dimension
    # Nframes, Ny, Mx = intensity.shape
    # if not (0 <= z0 <= z1 < Nframes):
    #     raise ValueError(f"z_bound out of range: z[{z0},{z1}], number of frames {Nframes}")
    # if not (0 <= y0 <= y1 < Ny):
    #     raise ValueError(f"y_bound out of range: y[{y0},{y1}], array shape {intensity.shape}")
    # if not (0 <= x0 <= x1 < Mx):
    #     raise ValueError(f"x_bound out of range: x[{x0},{x1}], array shape {intensity.shape}")

    # Q_crop   = Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
    # hkl_crop = hkl[z0:z1+1, y0:y1+1, x0:x1+1, :]
    # I_crop   = intensity[z0:z1+1, y0:y1+1, x0:x1+1]

    # return Q_crop, hkl_crop, I_crop



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


# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import griddata

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# def pixel2q(i, j, setup):
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
#     Builds per-pixel Q_samp, HKL, and intensity arrays from SPEC + TIFF data.

#     Parameters:
#         spec_file         : Path to the SPEC file.
#         tiff_dir          : Directory containing TIFF images.
#         pattern           : Optional regex pattern to match TIFF filenames.
#         use_dask          : If True, use Dask for loading data.
#         process_hklscan_only : If True, process only rows with type 'hklscan'.
#         selected_scans    : Optional iterable of scan_numbers to process.
#                             Only entries whose 'scan_number' is in this list will be kept.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
#                  process_hklscan_only=False, selected_scans=None):
#         # 1) load metadata + setup + crystal
#         exp = SpecParser(spec_file)
#         self.setup   = exp.setup
#         self.UB      = exp.crystal.UB

#         # 2) get scan metadata and force ints for merging
#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         # 3) load TIFF frames
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         # 4) merge metadata and intensity on ['scan_number','data_number']
#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
        
#         # 5) optionally filter only hklscan data
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
        
#         # 6) optionally filter to only selected scans
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)].reset_index(drop=True)

#         self.df = df

#         # 7) build static Q_lab grid from pixel indices
#         s = self.setup
#         cols = np.arange(s.xpixels)
#         rows = np.arange(s.ypixels)
#         N, M = np.meshgrid(cols, rows)  # N: x indices, M: y indices
#         self.Q_lab = pixel2q(N, M, s)    # shape (Ny, Mx, 3)

#     def compute_full(self):
#         """
#         Compute per-frame Q_samp, hkl, and intensity arrays.

#         Returns:
#           Q_samp    : ndarray (Nf, Ny, Mx, 3) sample-frame Q vectors.
#           hkl       : ndarray (Nf, Ny, Mx, 3) fractional indices.
#           intensity : ndarray (Nf, Ny, Mx) pixel intensities.
#         """
#         df = self.df
#         Nf = len(df)
#         # print(Nf, "frames to process")
#         Ny, Mx = df['intensity'].iat[0].shape

#         Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr   = np.zeros_like(Q_samp)
#         intens_arr= np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             print(df['scan_number'].iat[i], df['data_number'].iat[i], end='\r')
#             I   = row['intensity']
#             phi = row['phi']
#             chi = row['chi']
#             tth = row['tth']
#             th  = row['th']

#             phi_tot = phi + self.setup.phi
#             tth_tot = tth + self.setup.theta
#             th_tot  = th

#             rot_det  = R.from_euler('zxy', [phi_tot, chi, tth_tot], degrees=True).as_matrix()
#             rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
#             rot_full = rot_samp @ rot_det

#             Qs = self.Q_lab @ rot_full.T

#             Q_samp[i]     = Qs
#             UB = row['ub'] if 'ub' in row and row['ub'] is not None else self.UB
#             # print(UB)
#             hkl_arr[i]    = q2hkl(Qs, UB)
#             intens_arr[i] = I

#         self.Q_samp    = Q_samp
#         self.hkl       = hkl_arr
#         self.intensity = intens_arr

#         return Q_samp, hkl_arr, intens_arr

#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Define a uniform Q-space grid for regridding.

#         Parameters:
#             grid_ranges: Tuple of 3 (min, max) ranges for Qx, Qy, Qz.
#             grid_shape : Tuple of 3 ints (nx, ny, nz).
#         """
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def setup_hkl_grid(self, hkl_ranges, grid_shape):
#         """
#         Define a uniform HKL-space grid for regridding.

#         Parameters:
#             hkl_ranges: Tuple of 3 (min, max) ranges for h, k, l.
#             grid_shape : Tuple of 3 ints (nx, ny, nz).
#         """
#         self.hkl_ranges = hkl_ranges
#         self.hkl_shape  = grid_shape
#         self.hkl_edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(hkl_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         """
#         Bin intensities onto the defined uniform grid (either 'q' or 'hkl').

#         Parameters:
#             method: 'sum' (default) or 'mean'
#             space : 'q' to use Q_samp or 'hkl' to use hkl

#         Returns:
#             rsm   : 3D array of binned intensities.
#             edges : List of bin-edge arrays.
#         """
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid() first.")
#             pts   = self.Q_samp.reshape(-1, 3)
#             edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid() first.")
#             pts   = self.hkl.reshape(-1, 3)
#             edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

#         if method == 'sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_rsm = H_sum / H_cnt
#             mean_rsm[np.isnan(mean_rsm)] = 0
#         return mean_rsm, edges
#     def regrid_auto(self, space='q', grid_shape=(200, 200, 200), method='mean'):
#         """
#         Auto-compute grid ranges from Q_samp or hkl and regrid.

#         Parameters:
#             space: 'q' or 'hkl'
#             grid_shape: (nx, ny, nz)
#             method: 'sum' or 'mean'
#         """
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")

#         if space == 'q':
#             arr = self.Q_samp
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_grid(ranges, grid_shape)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("HKL not available. Call compute_full().")
#             arr = self.hkl
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_hkl_grid(ranges, grid_shape)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         return self.regrid_intensity(method=method, space=space)


# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import griddata

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# def pixel2q(i, j, setup):
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
#     Builds per-pixel Q_samp, HKL, and intensity arrays from SPEC + TIFF data.

#     Parameters:
#         spec_file         : Path to the SPEC file.
#         tiff_dir          : Directory containing TIFF images.
#         pattern           : Optional regex pattern to match TIFF filenames.
#         use_dask          : If True, use Dask for loading data.
#         process_hklscan_only : If True, process only rows with type 'hklscan'.
#         selected_scans    : Optional iterable of scan_numbers to process.
#                             Only entries whose 'scan_number' is in this list will be kept.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
#                  process_hklscan_only=False, selected_scans=None):
#         # 1) load metadata + setup + crystal
#         exp = SpecParser(spec_file)
#         self.setup   = exp.setup
#         self.UB      = exp.crystal.UB

#         # 2) get scan metadata and force ints for merging
#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         # 3) load TIFF frames
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         # 4) merge metadata and intensity on ['scan_number','data_number']
#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
        
#         # 5) optionally filter only hklscan data
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
        
#         # 6) optionally filter to only selected scans
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)].reset_index(drop=True)

#         self.df = df

#         # 7) build static Q_lab grid from pixel indices
#         s = self.setup
#         cols = np.arange(s.xpixels)
#         rows = np.arange(s.ypixels)
#         N, M = np.meshgrid(cols, rows)  # N: x indices, M: y indices
#         self.Q_lab = pixel2q(N, M, s)    # shape (Ny, Mx, 3)

#     def compute_full(self):
#         """
#         Compute per-frame Q_samp, hkl, and intensity arrays.

#         Returns:
#           Q_samp    : ndarray (Nf, Ny, Mx, 3) sample-frame Q vectors.
#           hkl       : ndarray (Nf, Ny, Mx, 3) fractional indices.
#           intensity : ndarray (Nf, Ny, Mx) pixel intensities.
#         """
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'].iat[0].shape

#         Q_samp    = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr   = np.zeros_like(Q_samp)
#         intens_arr= np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             print(df['scan_number'].iat[i], df['data_number'].iat[i], end='\r')
#             I   = row['intensity']
#             phi = row['phi']
#             chi = row['chi']
#             tth = row['tth']
#             th  = row['th']

#             phi_tot = phi + self.setup.phi
#             tth_tot = tth + self.setup.theta
#             th_tot  = th

#             rot_det  = R.from_euler('zxy', [phi_tot, chi, tth_tot], degrees=True).as_matrix()
#             rot_samp = R.from_euler('x', th_tot, degrees=True).as_matrix()
#             rot_full = rot_samp @ rot_det

#             Qs = self.Q_lab @ rot_full.T

#             Q_samp[i]     = Qs
#             UB_current = row['ub'] if 'ub' in row and row['ub'] is not None else self.UB
#             hkl_arr[i]    = q2hkl(Qs, UB_current)
#             intens_arr[i] = I

#         self.Q_samp    = Q_samp
#         self.hkl       = hkl_arr
#         self.intensity = intens_arr

#         return Q_samp, hkl_arr, intens_arr

#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Define a uniform Q-space grid for regridding.

#         Parameters:
#             grid_ranges: Tuple of 3 (min, max) ranges for Qx, Qy, Qz.
#             grid_shape : Tuple of 3 ints (nx, ny, nz).
#         """
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def setup_hkl_grid(self, hkl_ranges, grid_shape):
#         """
#         Define a uniform HKL-space grid for regridding.

#         Parameters:
#             hkl_ranges: Tuple of 3 (min, max) ranges for h, k, l.
#             grid_shape : Tuple of 3 ints (nx, ny, nz).
#         """
#         self.hkl_ranges = hkl_ranges
#         self.hkl_shape  = grid_shape
#         self.hkl_edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(hkl_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         """
#         Bin intensities onto the defined uniform grid (either 'q' or 'hkl').

#         Parameters:
#             method: 'sum' (default) or 'mean'
#             space : 'q' to use Q_samp or 'hkl' to use hkl

#         Returns:
#             rsm   : 3D array of binned intensities.
#             edges : List of bin-edge arrays.
#         """
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid() first.")
#             pts   = self.Q_samp.reshape(-1, 3)
#             edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid() first.")
#             pts   = self.hkl.reshape(-1, 3)
#             edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

#         if method == 'sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_rsm = H_sum / H_cnt
#             mean_rsm[np.isnan(mean_rsm)] = 0
#         return mean_rsm, edges

#     def regrid_auto(self, space='q', grid_shape=(200, 200, 200), method='mean'):
#         """
#         Auto-compute grid ranges from Q_samp or hkl and regrid.

#         Parameters:
#             space: 'q' or 'hkl'
#             grid_shape: Tuple (nx, ny, nz)
#             method: 'sum' or 'mean'
#         """
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")

#         if space == 'q':
#             arr = self.Q_samp
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_grid(ranges, grid_shape)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("HKL not available. Call compute_full().")
#             arr = self.hkl
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_hkl_grid(ranges, grid_shape)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200, 200, 200), method='linear'):
#         """
#         Interpolate scattered intensities onto a regular 3D grid using griddata.

#         Parameters:
#             space: 'q' to use Q_samp or 'hkl' to use hkl.
#             grid_shape: Tuple (nx, ny, nz) defining grid resolution.
#             method: Interpolation method ('linear', 'nearest', or 'cubic').

#         Returns:
#             grid_data : 3D ndarray of interpolated intensity values.
#             grid_axes : Tuple of three 1D arrays (xi, yi, zi) representing the grid coordinates.
#         """
#         if space == 'q':
#             if not hasattr(self, 'Q_samp'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.Q_samp.reshape(-1, 3)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.hkl.reshape(-1, 3)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         intensity_vals = self.intensity.ravel()

#         # Optionally filter out zero or near-zero intensities (threshold can be adjusted)
#         mask = intensity_vals > 0
#         pts = pts[mask]
#         intensity_vals = intensity_vals[mask]

#         # Compute grid boundaries based on scattered points
#         xmin, ymin, zmin = pts.min(axis=0)
#         xmax, ymax, zmax = pts.max(axis=0)

#         nx, ny, nz = grid_shape
#         xi = np.linspace(xmin, xmax, nx)
#         yi = np.linspace(ymin, ymax, ny)
#         zi = np.linspace(zmin, zmax, nz)

#         XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
#         grid_data = griddata(pts, intensity_vals, (XI, YI, ZI), method=method, fill_value=0)
#         grid_axes = (xi, yi, zi)
#         return grid_data, grid_axes


# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R  # (not used for Q_samp now)
# from scipy.interpolate import griddata
# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData
# Import the hklpy diffractometer engine.
# import gi
# gi.require_version("Hkl", "5.0")
# from hkl import E4CV
# from ophyd import Component as Cpt, PseudoSingle, SoftPositioner

# # Define a FourCircle diffractometer using hklpy.
# class FourCircle(E4CV):
#     # Pseudo axes for h, k, l
#     h = Cpt(PseudoSingle, '')
#     k = Cpt(PseudoSingle, '')
#     l = Cpt(PseudoSingle, '')
#     # Real motor axes in canonical order.
#     omega = Cpt(SoftPositioner)
#     chi   = Cpt(SoftPositioner)
#     phi   = Cpt(SoftPositioner)
#     tth   = Cpt(SoftPositioner)

# # Create a global instance and set wavelength (example for 18 keV).
# fourc = FourCircle("", name="fourc")
# fourc.calc.wavelength = 12.398419843320026 / 18.0

# def pixel2q_hklpy(setup, fourc):
#     """
#     Compute lab-frame Q vectors for each pixel using hklpy.
    
#     Parameters:
#         setup : An ExperimentSetup-like object with attributes:
#                 xcenter, ycenter, xpixels, ypixels, pitch, distance, wavelength.
#         fourc : A configured FourCircle diffractometer instance.
    
#     Returns:
#         Q_lab : ndarray of shape (Ny, Mx, 3) where each (Qx, Qy, Qz) is computed via 
#                 fourc.calc.scatter (which uses the current motor settings).
#     """
#     # Create a meshgrid for pixel indices.
#     cols = np.arange(setup.xpixels)
#     rows = np.arange(setup.ypixels)
#     X, Y = np.meshgrid(cols, rows)  # shape (Ny, Mx)
#     # Convert pixel indices to physical coordinates (in same units as pitch)
#     x = (X - setup.xcenter) * setup.pitch
#     y = (Y - setup.ycenter) * setup.pitch
#     D = setup.distance
#     wl = setup.wavelength
#     k0 = 2 * np.pi / wl

#     # Compute the scattering angles in radians.
#     two_theta = np.arctan2(np.hypot(x, y), D)
#     eta = np.arctan2(y, x)
#     # Convert to degrees for hklpy.
#     two_theta_deg = np.degrees(two_theta)
#     eta_deg = np.degrees(eta)
#     # Use hklpy's calc.scatter to compute k_f.
#     kf = fourc.calc.scatter_vecter(two_theta=two_theta_deg, eta=eta_deg)
#     # fourc.calc.scatter returns an array with shape (3, Ny, Mx); rotate axes.
#     kf = np.moveaxis(kf, 0, -1)  # now (Ny, Mx, 3)
#     # The incident beam is assumed along +z.
#     ki = np.array([0.0, 0.0, k0])
#     # Q_lab = k_f - k_i.
#     return kf - ki

# def q2hkl(Q_samp, UB):
#     """
#     Convert sample-frame Q vectors to fractional (h, k, l) indices.

#     Parameters:
#         Q_samp : ndarray (..., 3) of Q vectors.
#         UB     : 3x3 orientation matrix.

#     Returns:
#         hkl : ndarray of same shape as Q_samp.
#     """
#     UB_inv = np.linalg.inv(UB)
#     flat = Q_samp.reshape(-1, 3)
#     hkl = flat @ UB_inv.T
#     return hkl.reshape(Q_samp.shape)

# class RSMBuilder:
#     """
#     Builds per-pixel Q (sample-frame), HKL, and intensity arrays from SPEC + TIFF data.
    
#     The Q_samp calculation now uses hklpy's scattering engine.
    
#     Parameters:
#         spec_file         : Path to the SPEC file.
#         tiff_dir          : Directory containing TIFF images.
#         pattern           : Optional regex pattern to match TIFF filenames.
#         use_dask          : If True, use Dask for loading data.
#         process_hklscan_only : If True, process only rows with type 'hklscan'.
#         selected_scans    : Optional iterable of scan_numbers to process.
#                             Only entries whose 'scan_number' is in this list will be kept.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
#                  process_hklscan_only=False, selected_scans=None):
#         # 1) load metadata, setup, and crystal from SPEC.
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = exp.crystal.UB

#         # 2) Obtain scan metadata; force ints for merging.
#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         # 3) Load TIFF frames.
#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         # 4) Merge metadata and intensity on ['scan_number','data_number'].
#         df = pd.merge(df_meta, df_int, on=['scan_number', 'data_number'])
        
#         # 5) Optionally filter only hklscan data.
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
        
#         # 6) Optionally filter to only selected scans.
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)].reset_index(drop=True)
        
#         self.df = df

#         # 7) Calculate a static lab‑frame Q grid using hklpy.
#         # This grid is computed using current fourc motor settings.
#         self.Q_lab = pixel2q_hklpy(self.setup, fourc)  # shape (Ny, Mx, 3)

#     def compute_full(self):
#         """
#         For each frame, update the diffractometer motor angles in fourc,
#         recalculate the Q grid using hklpy (i.e. pixel2q_hklpy), and then compute HKL.

#         Returns:
#             Q_samp    : ndarray (Nf, Ny, Mx, 3) sample-frame Q vectors.
#             hkl       : ndarray (Nf, Ny, Mx, 3) fractional coordinates.
#             intensity : ndarray (Nf, Ny, Mx) pixel intensities.
#         """
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'].iat[0].shape

#         Q_samp = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr = np.zeros_like(Q_samp)
#         intens_arr = np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             print(f"Processing scan {df['scan_number'].iat[i]} data {df['data_number'].iat[i]}", end='\r')
#             I = row['intensity']
#             # Retrieve motor angles from the metadata.
#             # Here, we assume:
#             #   - 'th' is used for omega,
#             #   - 'chi' for chi,
#             #   - 'phi' for phi,
#             #   - 'tth' for two-theta.
#             omega_angle = row['th']
#             chi_angle = row['chi']
#             phi_angle = row['phi']
#             tth_angle = row['tth']
#             # Optionally add global offsets from setup.
#             omega_tot = omega_angle + self.setup.phi  # adjust as needed
#             tth_tot = tth_angle + self.setup.theta
#             # Set fourc motor values.
#             fourc.omega.value = omega_tot
#             fourc.chi.value = chi_angle
#             fourc.phi.value = phi_angle
#             fourc.tth.value = tth_tot

#             # Recalculate the lab-frame Q grid for this frame using hklpy.
#             Qs = pixel2q_hklpy(self.setup, fourc)
#             Q_samp[i] = Qs
#             # Use per-scan UB if available; otherwise, fall back to the global UB.
#             UB_current = row['ub'] if ('ub' in row and row['ub'] is not None) else self.UB
#             hkl_arr[i] = q2hkl(Qs, UB_current)
#             intens_arr[i] = I

#         self.Q_samp = Q_samp
#         self.hkl = hkl_arr
#         self.intensity = intens_arr

#         return Q_samp, hkl_arr, intens_arr

#     # (The regridding, cropping, and interpolation methods remain unchanged.)
#     def setup_grid(self, grid_ranges, grid_shape):
#         self.grid_ranges = grid_ranges
#         self.grid_shape = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def setup_hkl_grid(self, hkl_ranges, grid_shape):
#         self.hkl_ranges = hkl_ranges
#         self.hkl_shape = grid_shape
#         self.hkl_edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(hkl_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid() first.")
#             pts = self.Q_samp.reshape(-1,3)
#             edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid() first.")
#             pts = self.hkl.reshape(-1,3)
#             edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method == 'sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_rsm = H_sum / H_cnt
#             mean_rsm[np.isnan(mean_rsm)] = 0
#         return mean_rsm, edges

#     def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             arr = self.Q_samp
#             ranges = (
#                 (arr[...,0].min(), arr[...,0].max()),
#                 (arr[...,1].min(), arr[...,1].max()),
#                 (arr[...,2].min(), arr[...,2].max()),
#             )
#             self.setup_grid(ranges, grid_shape)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("Call compute_full() first.")
#             arr = self.hkl
#             ranges = (
#                 (arr[...,0].min(), arr[...,0].max()),
#                 (arr[...,1].min(), arr[...,1].max()),
#                 (arr[...,2].min(), arr[...,2].max()),
#             )
#             self.setup_hkl_grid(ranges, grid_shape)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
#         if space == 'q':
#             if not hasattr(self, 'Q_samp'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.Q_samp.reshape(-1,3)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.hkl.reshape(-1,3)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         intensity_vals = self.intensity.ravel()
#         mask = intensity_vals > 0
#         pts = pts[mask]
#         intensity_vals = intensity_vals[mask]
#         xmin, ymin, zmin = pts.min(axis=0)
#         xmax, ymax, zmax = pts.max(axis=0)
#         nx, ny, nz = grid_shape
#         xi = np.linspace(xmin, xmax, nx)
#         yi = np.linspace(ymin, ymax, ny)
#         zi = np.linspace(zmin, zmax, nz)
#         XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
#         grid_data = griddata(pts, intensity_vals, (XI, YI, ZI), method=method, fill_value=0)
#         grid_axes = (xi, yi, zi)
#         return grid_data, grid_axes

# ...existing imports...
# Remove the broken pixel2q_hklpy / scatter_vecter usage.
# Keep/use hklpy only for UB handling (orientation & wavelength). hklpy does not provide
# a per-pixel scatter API; we still compute pixel → Q_lab analytically.

# import gi
# gi.require_version("Hkl", "5.0")
# from hkl import E4CV
# from ophyd import Component as Cpt, PseudoSingle, SoftPositioner

# class FourCircle(E4CV):
#     h = Cpt(PseudoSingle, '')
#     k = Cpt(PseudoSingle, '')
#     l = Cpt(PseudoSingle, '')
#     omega = Cpt(SoftPositioner)
#     chi   = Cpt(SoftPositioner)
#     phi   = Cpt(SoftPositioner)
#     tth   = Cpt(SoftPositioner)

# fourc = FourCircle("", name="fourc")
# # Set wavelength (Å) from your experiment; you can override after import.
# fourc.calc.wavelength = 12.398419843320026 / 18.0

# def pixel2q(i, j, setup):
#     """
#     Corrected pixel → Q_lab calculation
#     """
#     x = (i - setup.xcenter) * setup.pitch
#     y = (j - setup.ycenter) * setup.pitch
#     D = setup.distance
#     wl = setup.wavelength
#     k0 = 2 * np.pi / wl
    
#     # Correct scattering angles
#     two_theta = np.arctan2(np.hypot(x, y), D)
#     eta = np.arctan2(y, x)
    
#     # Correct scattered wavevector components
#     kfx = k0 * np.sin(two_theta) * np.cos(eta)
#     kfy = k0 * np.sin(two_theta) * np.sin(eta) 
#     kfz = k0 * np.cos(two_theta)
    
#     # Q = kf - ki (with ki = [0, 0, k0])
#     Qx = kfx - 0
#     Qy = kfy - 0  
#     Qz = kfz - k0
    
#     return np.stack((Qx, Qy, Qz), axis=-1)

# def rot_axis(axis_xyz, ang_deg):
#     a = np.asarray(axis_xyz, dtype=float)
#     a /= np.linalg.norm(a)
#     th = np.deg2rad(ang_deg)
#     K = np.array([[    0, -a[2],  a[1]],
#                   [ a[2],     0, -a[0]],
#                   [-a[1],  a[0],     0]])
#     I = np.eye(3)
#     return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

# def q2hkl(Q_samp, UB):
#     UB_inv = np.linalg.inv(UB)
#     flat = Q_samp.reshape(-1, 3)
#     hkl  = flat @ UB_inv.T
#     return hkl.reshape(Q_samp.shape)

# class RSMBuilder:
#     """
#     Use hklpy (fourc) for wavelength & UB management.
#     Pixel → Q uses analytic geometry; sample rotation uses E4CV axis conventions:
#       omega about -y, chi about +x, phi about -y (right-hand rule).
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
#                  process_hklscan_only=False, selected_scans=None):
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         # Sync fourc wavelength to setup
#         fourc.calc.wavelength = self.setup.wavelength
#         # Global fallback UB (from SPEC crystal)
#         self.UB = exp.crystal.UB

#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)]
#         df = df.reset_index(drop=True)
#         self.df = df

#         s = self.setup
#         cols = np.arange(s.xpixels)
#         rows = np.arange(s.ypixels)
#         N, M = np.meshgrid(cols, rows)
#         self.Q_lab = pixel2q(N, M, s)  # (Ny, Mx, 3)

#     def compute_full(self):
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'].iat[0].shape
#         Q_samp = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr = np.zeros_like(Q_samp)
#         I_arr = np.zeros((Nf, Ny, Mx), dtype=float)

#         for i, row in df.iterrows():
#             I = row['intensity']
#             omega = row['th']     # sample rotation
#             chi = row['chi']      # sample tilt  
#             phi = row['phi']      # sample azimuth
#             tth = row['tth']      # detector two-theta

#             # Apply global offsets
#             omega_tot = omega
#             chi_tot = chi
#             phi_tot = phi + self.setup.phi
#             tth_tot = tth + getattr(self.setup, "theta", 0.0)

#             # Method 1: Apply detector rotation first, then sample rotation
#             # (appropriate if detector moves independently)
#             R_tth = rot_axis([0, -1, 0], tth_tot)  # detector arm rotation
#             Q_det_rotated = (R_tth @ self.Q_lab.reshape(-1, 3).T).T.reshape(Ny, Mx, 3)
            
#             # Sample rotation (crystal→lab)
#             R_omega = rot_axis([0, -1, 0], omega_tot)
#             R_chi = rot_axis([1, 0, 0], chi_tot)  
#             R_phi = rot_axis([0, -1, 0], phi_tot)
#             R_sample = R_omega @ R_chi @ R_phi
            
#             # Transform to sample frame: lab→sample = R_sample.T
#             Qc = (R_sample.T @ Q_det_rotated.reshape(-1, 3).T).T.reshape(Ny, Mx, 3)
            
#             Q_samp[i] = Qc
#             UB_use = row['ub'] if ('ub' in row and row['ub'] is not None) else self.UB
#             hkl_arr[i] = q2hkl(Qc, UB_use)
#             I_arr[i] = I
            
#             print(f"scan {row['scan_number']} frame {row['data_number']}", end='\r')

#         self.Q_samp = Q_samp
#         self.hkl = hkl_arr
#         self.intensity = I_arr
#         return Q_samp, hkl_arr, I_arr

#     def setup_grid(self, ranges, shape):
#         self.edges = [np.linspace(r[0], r[1], shape[i]+1) for i, r in enumerate(ranges)]

#     def setup_hkl_grid(self, ranges, shape):
#         self.hkl_edges = [np.linspace(r[0], r[1], shape[i]+1) for i, r in enumerate(ranges)]

#     def regrid_intensity(self, method='sum', space='q'):
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid first.")
#             pts = self.Q_samp.reshape(-1,3); edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid first.")
#             pts = self.hkl.reshape(-1,3); edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method == 'sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             H_mean = H_sum / H_cnt
#             H_mean[np.isnan(H_mean)] = 0
#         return H_mean, edges

#     def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
#         if space == 'q':
#             arr = self.Q_samp
#             rng = [(arr[...,k].min(), arr[...,k].max()) for k in range(3)]
#             self.setup_grid(rng, grid_shape)
#         elif space == 'hkl':
#             arr = self.hkl
#             rng = [(arr[...,k].min(), arr[...,k].max()) for k in range(3)]
#             self.setup_hkl_grid(rng, grid_shape)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
#         if space == 'q':
#             pts = self.Q_samp.reshape(-1,3)
#         elif space == 'hkl':
#             pts = self.hkl.reshape(-1,3)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")
#         vals = self.intensity.ravel()
#         mask = vals > 0
#         pts = pts[mask]; vals = vals[mask]
#         mins = pts.min(axis=0); maxs = pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         X, Y, Z = np.meshgrid(*axes, indexing='ij')
#         G = griddata(pts, vals, (X, Y, Z), method=method, fill_value=0)
#         return G, axes

#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("compute_full first.")
#         Nf, Ny, Mx = self.intensity.shape
#         z0,z1 = (0, Nf-1) if z_bound is None else z_bound
#         y0,y1 = (0, Ny-1) if y_bound is None else y_bound
#         x0,x1 = (0, Mx-1) if x_bound is None else x_bound
#         Qc   = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Hc   = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Ic   = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
#         return Qc, Hc, Ic


# import os
# import numpy as np
# import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import griddata

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# # ... (hklpy imports can be removed if not used, or kept for future use)

# def pixel2q(i, j, setup):
#     """
#     Corrected: Calculate Q vector in the detector's local frame (at tth=0).
#     The scattered wavevector kf is calculated using standard spherical coordinates
#     derived from pixel positions.
#     """
#     x = (i - setup.xcenter) * setup.pitch
#     y = (j - setup.ycenter) * setup.pitch
#     D = setup.distance
#     wl = setup.wavelength
#     k0 = 2 * np.pi / wl

#     # Scattering angles from pixel coordinates
#     two_theta = np.arctan2(np.hypot(x, y), D)
#     eta = np.arctan2(y, x)

#     # CORRECTED: kf components using sin(two_theta) and cos(two_theta)
#     kfx = k0 * np.sin(two_theta) * np.cos(eta)
#     kfy = k0 * np.sin(two_theta) * np.sin(eta)
#     kfz = k0 * np.cos(two_theta)

#     # Q = kf - ki, where ki = [0, 0, k0]
#     Qx = kfx
#     Qy = kfy
#     Qz = kfz - k0

#     return np.stack((Qx, Qy, Qz), axis=-1)

# def rot_axis(axis_xyz, ang_deg):
#     """Rodrigues' rotation formula for a given axis and angle."""
#     a = np.asarray(axis_xyz, dtype=float)
#     a /= np.linalg.norm(a)
#     th = np.deg2rad(ang_deg)
#     K = np.array([[    0, -a[2],  a[1]],
#                   [ a[2],     0, -a[0]],
#                   [-a[1],  a[0],     0]])
#     I = np.eye(3)
#     return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

# def q2hkl(Q_samp, UB):
#     """Converts sample-frame Q vectors to HKL using the UB matrix."""
#     UB_inv = np.linalg.inv(UB)
#     flat = Q_samp.reshape(-1, 3)
#     hkl = flat @ UB_inv.T
#     return hkl.reshape(Q_samp.shape)

# class RSMBuilder:
#     """
#     Builds 3D Reciprocal Space Maps from SPEC and TIFF data.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None, use_dask=False,
#                  process_hklscan_only=False, selected_scans=None):
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = exp.crystal.UB

#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         df = pd.merge(df_meta, df_int, on=['scan_number', 'data_number'])
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)]
#         self.df = df.reset_index(drop=True)

#         s = self.setup
#         cols, rows = np.meshgrid(np.arange(s.xpixels), np.arange(s.ypixels))
#         self.Q_lab = pixel2q(cols, rows, s)  # Q vectors in detector frame (tth=0)

#     def compute_full(self):
#         """
#         Corrected: Computes Q_samp and HKL with proper rotation sequence.
#         The logic follows a standard four-circle diffractometer geometry.
#         """
#         df = self.df
#         Nf = len(df)
#         Ny, Mx = df['intensity'].iat[0].shape
#         Q_samp = np.zeros((Nf, Ny, Mx, 3), dtype=float)
#         hkl_arr = np.zeros_like(Q_samp)
#         I_arr = np.zeros((Nf, Ny, Mx), dtype=float)

#         # Cache the flattened detector Q grid
#         Q_det_flat = self.Q_lab.reshape(-1, 3).T

#         for i, row in df.iterrows():
#             I = row['intensity']
#             omega, chi, phi, tth = row['th'], row['chi'], row['phi'], row['tth']

#             # Apply global motor offsets from setup
#             omega_tot = omega
#             chi_tot = chi
#             phi_tot = phi + self.setup.phi
#             tth_tot = tth + getattr(self.setup, "theta", 0.0)

#             # 1. Define sample rotation matrix (crystal frame -> lab frame)
#             # Using E4CV convention: omega(-y), chi(+x), phi(-y)
#             R_omega = rot_axis([0, -1, 0], omega_tot)
#             R_chi = rot_axis([1, 0, 0], chi_tot)
#             R_phi = rot_axis([0, -1, 0], phi_tot)
#             R_sample = R_omega @ R_chi @ R_phi

#             # 2. Define detector rotation matrix (about lab -y axis)
#             R_tth = rot_axis([0, -1, 0], tth_tot)

#             # 3. Calculate Q in the lab frame for this detector position
#             Q_lab_frame = R_tth @ Q_det_flat

#             # 4. Transform Q from lab frame to sample frame
#             # Q_sample = inv(R_sample) @ Q_lab = R_sample.T @ Q_lab
#             Qc = (R_sample.T @ Q_lab_frame).T.reshape(Ny, Mx, 3)

#             Q_samp[i] = Qc
#             UB_use = row.get('ub') if row.get('ub') is not None else self.UB
#             hkl_arr[i] = q2hkl(Qc, UB_use)
#             I_arr[i] = I
#             print(f"scan {row['scan_number']} frame {row['data_number']}", end='\r')

#         self.Q_samp = Q_samp
#         self.hkl = hkl_arr
#         self.intensity = I_arr
#         return Q_samp, hkl_arr, I_arr

#     # ... (All other methods like setup_grid, regrid_intensity, etc. remain the same) ...
#     def setup_grid(self, grid_ranges, grid_shape):
#         self.grid_ranges = grid_ranges
#         self.grid_shape  = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def setup_hkl_grid(self, hkl_ranges, grid_shape):
#         self.hkl_ranges = hkl_ranges
#         self.hkl_shape  = grid_shape
#         self.hkl_edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(hkl_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")
#         if space == 'q':
#             if not hasattr(self, 'edges'):
#                 raise RuntimeError("Call setup_grid() first.")
#             pts   = self.Q_samp.reshape(-1, 3)
#             edges = self.edges
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl_edges'):
#                 raise RuntimeError("Call setup_hkl_grid() first.")
#             pts   = self.hkl.reshape(-1, 3)
#             edges = self.hkl_edges
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

#         if method == 'sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_rsm = H_sum / H_cnt
#             mean_rsm[np.isnan(mean_rsm)] = 0
#         return mean_rsm, edges

#     def regrid_auto(self, space='q', grid_shape=(200, 200, 200), method='mean'):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("Call compute_full() first.")

#         if space == 'q':
#             arr = self.Q_samp
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_grid(ranges, grid_shape)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("HKL not available. Call compute_full().")
#             arr = self.hkl
#             ranges = (
#                 (arr[..., 0].min(), arr[..., 0].max()),
#                 (arr[..., 1].min(), arr[..., 1].max()),
#                 (arr[..., 2].min(), arr[..., 2].max()),
#             )
#             self.setup_hkl_grid(ranges, grid_shape)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200, 200, 200), method='linear'):
#         if space == 'q':
#             if not hasattr(self, 'Q_samp'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.Q_samp.reshape(-1, 3)
#         elif space == 'hkl':
#             if not hasattr(self, 'hkl'):
#                 raise RuntimeError("Call compute_full() first.")
#             pts = self.hkl.reshape(-1, 3)
#         else:
#             raise ValueError("space must be 'q' or 'hkl'")

#         intensity_vals = self.intensity.ravel()
#         mask = intensity_vals > 0
#         pts = pts[mask]
#         intensity_vals = intensity_vals[mask]

#         xmin, ymin, zmin = pts.min(axis=0)
#         xmax, ymax, zmax = pts.max(axis=0)

#         nx, ny, nz = grid_shape
#         xi = np.linspace(xmin, xmax, nx)
#         yi = np.linspace(ymin, ymax, ny)
#         zi = np.linspace(zmin, zmax, nz)

#         XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
#         grid_data = griddata(pts, intensity_vals, (XI, YI, ZI), method=method, fill_value=0)
#         grid_axes = (xi, yi, zi)
#         return grid_data, grid_axes

#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         if not hasattr(self, 'Q_samp'):
#             raise RuntimeError("compute_full first.")
#         Nf, Ny, Mx = self.intensity.shape
#         z0,z1 = (0, Nf-1) if z_bound is None else z_bound
#         y0,y1 = (0, Ny-1) if y_bound is None else y_bound
#         x0,x1 = (0, Mx-1) if x_bound is None else x_bound
#         Qc   = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Hc   = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Ic   = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
#         return


# import os
# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# # ─── Rotation helpers ──────────────────────────────────────────────────────────
# def Rx(a):
#     ca, sa = np.cos(a), np.sin(a)
#     return np.array([[1, 0, 0],
#                      [0, ca, -sa],
#                      [0, sa,  ca]])

# def Ry(b):
#     cb, sb = np.cos(b), np.sin(b)
#     return np.array([[ cb, 0, sb],
#                      [  0, 1,  0],
#                      [-sb, 0, cb]])

# def Rz(g):
#     cg, sg = np.cos(g), np.sin(g)
#     return np.array([[cg, -sg, 0],
#                      [sg,  cg, 0],
#                      [ 0,   0, 1]])

# def r_det_to_lab(x_det, y_det, z_det, R_det):
#     ny, nx = x_det.shape
#     v = np.stack((x_det, y_det, z_det), axis=0).reshape(3, -1)
#     v_lab = (R_det @ v).reshape(3, ny, nx)
#     return v_lab

# # ─── Q_lab map ─────────────────────────────────────────────────────────────────
# def q_lab_map(shape, px_size_m, dist_m, beam_center,
#               wavelength_A, det_tilt_deg=(0.0,0.0,0.0), y_down=True):
#     """
#     Compute per-pixel Q in lab frame (Å⁻¹).
#     shape: (ny, nx)
#     px_size_m: (px, py) in meters
#     dist_m: detector distance in meters
#     beam_center: (cx, cy) in pixel indices
#     wavelength_A: wavelength in Å
#     det_tilt_deg: (pitch_x, yaw_y, roll_z) in degrees
#     """
#     ny, nx = shape
#     px, py = px_size_m
#     cx, cy = beam_center
#     pitch_x, yaw_y, roll_z = np.deg2rad(det_tilt_deg)

#     j = np.arange(nx); i = np.arange(ny)
#     J, I = np.meshgrid(j, i)
#     x = (J - cx) * px
#     y = (I - cy) * py
#     if y_down:
#         y = -y
#     z = np.full_like(x, dist_m)

#     R_det = Rz(roll_z) @ Ry(yaw_y) @ Rx(pitch_x)
#     v_lab = r_det_to_lab(x, y, z, R_det)
#     norms = np.linalg.norm(v_lab, axis=0)
#     s_out = v_lab / norms
#     s_in = np.zeros_like(s_out); s_in[2,:,:] = 1.0

#     k = 2 * np.pi / wavelength_A
#     return k * (s_out - s_in)

# # ─── HKL map from Q ────────────────────────────────────────────────────────────
# def hkl_map_from_Q(Q_lab, UB, R_samp=np.eye(3)):
#     """
#     Convert lab-frame Q to HKL.
#     Q_lab: (3, ny, nx)
#     UB: 3×3 matrix (Å⁻¹ → HKL)
#     R_samp: sample→lab rotation matrix
#     """
#     ny, nx = Q_lab.shape[1:]
#     UB_inv = np.linalg.inv(UB)
#     Qs = (R_samp.T @ Q_lab.reshape(3, -1))
#     HKL = (UB_inv @ Qs).reshape(3, ny, nx)
#     return HKL[0], HKL[1], HKL[2]

# # ─── RSMBuilder ────────────────────────────────────────────────────────────────
# class RSMBuilder:
#     """
#     Builds 3D reciprocal‐space maps (Q and HKL volumes) from SPEC + TIFF data.
#     """
#     def __init__(self, spec_file, tiff_dir, pattern=None,
#                  use_dask=False, process_hklscan_only=False,
#                  selected_scans=None):
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = exp.crystal.UB

#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         rd = ReadData(tiff_dir, pattern=pattern, use_dask=use_dask)
#         df_int = rd.load_data()

#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
#         if process_hklscan_only:
#             df = df[df['type'].str.lower() == 'hklscan']
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)]
#         self.df = df.reset_index(drop=True)

#     def compute_full(self):
#         """
#         For each frame, compute per-pixel Q_sample and HKL arrays.
#         Returns Q_samp (Nf, ny, nx, 3), hkl (Nf, ny, nx, 3), intensity (Nf, ny, nx).
#         """
#         df = self.df
#         Nf = len(df)
#         ny, nx = df['intensity'].iat[0].shape

#         Q_samp = np.zeros((Nf, ny, nx, 3), dtype=float)
#         hkl_arr = np.zeros_like(Q_samp)
#         I_arr = np.zeros((Nf, ny, nx), dtype=float)

#         shape = (ny, nx)
#         px_size_m   = (self.setup.pitch, self.setup.pitch)
#         dist_m      = self.setup.distance
#         beam_center = (self.setup.xcenter, self.setup.ycenter)
#         wl_A        = self.setup.wavelength

#         for i, row in df.iterrows():
#             I   = row['intensity']
#             phi = row['phi']
#             chi = row['chi']
#             th  = row['th']
#             tth = row['tth']

#             # apply global offsets
#             phi_tot = phi + getattr(self.setup, 'phi', 0.0)
#             tth_tot = tth + getattr(self.setup, 'theta', 0.0)

#             # compute lab-frame Q at this detector orientation
#             det_tilt = (0.0, tth_tot, phi_tot)
#             Q_lab = q_lab_map(shape, px_size_m, dist_m,
#                               beam_center, wl_A,
#                               det_tilt, y_down=True)

#             # sample rotation (phi, chi, th) about Z, Y, X
#             R_samp = Rz(np.deg2rad(phi)) @ \
#                      Ry(np.deg2rad(chi)) @ \
#                      Rx(np.deg2rad(th))

#             # sample-frame Q and HKL
#             Qc = (R_samp.T @ Q_lab.reshape(3, -1)).T.reshape(ny, nx, 3)
#             UB_use = row.get('ub') if row.get('ub') is not None else self.UB
#             H, K, L = hkl_map_from_Q(Q_lab, UB_use, R_samp)

#             Q_samp[i]   = Qc
#             hkl_arr[i]  = np.stack((H, K, L), axis=-1)
#             I_arr[i]    = I
#             print(f"scan {row['scan_number']} frame {row['data_number']}", end='\r')

#         self.Q_samp, self.hkl, self.intensity = Q_samp, hkl_arr, I_arr
#         return Q_samp, hkl_arr, I_arr

#     def setup_grid(self, grid_ranges, grid_shape):
#         self.grid_ranges = grid_ranges
#         self.grid_shape = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         if space == 'q':
#             pts, edges = self.Q_samp.reshape(-1,3), self.edges
#         else:
#             pts, edges = self.hkl.reshape(-1,3), self.hkl_edges
#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method=='sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             Hm = H_sum / H_cnt
#             Hm[np.isnan(Hm)] = 0
#         return Hm, edges

#     def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
#         arr = self.Q_samp if space=='q' else self.hkl
#         ranges = tuple((arr[...,k].min(), arr[...,k].max()) for k in range(3))
#         if space=='q':
#             self.setup_grid(ranges, grid_shape)
#         else:
#             self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i]+1)
#                               for i,r in enumerate(ranges)]
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
#         pts = (self.Q_samp if space=='q' else self.hkl).reshape(-1,3)
#         vals = self.intensity.ravel()
#         mask = vals>0
#         pts, vals = pts[mask], vals[mask]
#         mins, maxs = pts.min(axis=0), pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         XI, YI, ZI = np.meshgrid(*axes, indexing='ij')
#         G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0)
#         return G, axes

#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         Nf, ny, nx = self.intensity.shape
#         z0,z1 = (0,Nf-1) if z_bound is None else z_bound
#         y0,y1 = (0,ny-1) if y_bound is None else y_bound
#         x0,x1 = (0,nx-1) if x_bound is None else x_bound
#         Qc = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Hc = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Ic = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
        # return


# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from pyFAI.integrator.azimuthal import AzimuthalIntegrator
# from scipy.spatial.transform import Rotation as _Rot

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData

# # ─── RSMBuilder with pyFAI Q‐mapping ────────────────────────────────────────────
# class RSMBuilder:
#     """
#     Builds 3D reciprocal‐space maps from SPEC + TIFF data using pyFAI for Q‐mapping.
    
#     Parameters:
#       spec_file       : Path to SPEC file
#       tiff_dir        : Directory with TIFF frames
#       poni_file       : Optional pyFAI .poni calibration file
#       use_dask        : If True, load with Dask
#       process_hklscan_only : If True, filter only 'hklscan' entries
#       selected_scans  : Iterable of scan_numbers to include
#     """
#     def __init__(self,
#                  spec_file,
#                  tiff_dir,
#                  poni_file=None,
#                  use_dask=False,
#                  process_hklscan_only=False,
#                  selected_scans=None):
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB    = exp.crystal.UB

#         # Merge SPEC metadata + TIFF intensities
#         df_meta = exp.to_pandas()
#         df_meta['scan_number'] = df_meta['scan_number'].astype(int)
#         df_meta['data_number'] = df_meta['data_number'].astype(int)

#         rd = ReadData(tiff_dir, use_dask=use_dask)
#         df_int = rd.load_data()

#         df = pd.merge(df_meta, df_int, on=['scan_number','data_number'])
#         if process_hklscan_only:
#             df = df[df['type'].str.lower()=='hklscan']
#         if selected_scans is not None:
#             df = df[df['scan_number'].isin(selected_scans)]
#         self.df = df.reset_index(drop=True)

#         # Instantiate pyFAI integrator
#         if poni_file:
#             self.ai = AzimuthalIntegrator(poni_file=poni_file)
#         else:
#             # Build from ExperimentSetup (distance in m, poni in m, wavelength in Å)
#             self.ai = AzimuthalIntegrator()
#            # pyFAI.setFit2D signature is (poni1, poni2, dist, rot1, rot2, rot3)
#             self.ai.setFit2D(
#                 self.setup.distance,
#                 self.setup.xcenter * self.setup.pitch,
#                 self.setup.ycenter * self.setup.pitch,
#                 0.0, 
#                 0.0, 
#                 self.setup.pitch,
#                 self.setup.pitch,
#             )
#             self.ai.wavelength = self.setup.wavelength

#         # Precompute lab‐frame Q‐map at iki=0 orientation
#         shape = (self.setup.ypixels, self.setup.xpixels)
#         two_theta = self.ai.twoThetaArray(shape)
#         chi       = self.ai.chiArray(shape)
#         k0        = 2 * np.pi / self.setup.wavelength
#         self.Q_lab = np.stack((
#             k0 * np.sin(two_theta) * np.cos(chi),
#             k0 * np.sin(two_theta) * np.sin(chi),
#             k0 * (np.cos(two_theta) - 1.0),
#         ), axis=-1)  # (ny, nx, 3)

#     def compute_full(self):
#         """
#         For each frame, apply sample rotation and compute Q_samp & hkl per pixel.
#         Returns:
#           Q_samp    : (Nf, ny, nx, 3)
#           hkl       : (Nf, ny, nx, 3)
#           intensity : (Nf, ny, nx)
#         """
#         df   = self.df
#         Nf   = len(df)
#         ny,nx= df['intensity'].iat[0].shape

#         Q_samp    = np.zeros((Nf, ny, nx, 3), dtype=float)
#         hkl_arr   = np.zeros_like(Q_samp)
#         I_arr     = np.zeros((Nf, ny, nx), dtype=float)

#         for idx, row in df.iterrows():
#             I   = row['intensity']
#             th  = row['th']    # sample θ
#             chi = row['chi']   # sample χ
#             phi = row['phi']   # sample φ

#             # Build sample rotation: Z(φ) → Y(χ) → X(θ)
#             Rz = _Rot.from_euler('z', phi, degrees=True).as_matrix()
#             Ry = _Rot.from_euler('y', chi, degrees=True).as_matrix()
#             Rx = _Rot.from_euler('x', th,  degrees=True).as_matrix()
#             R_sample = Rz @ Ry @ Rx

#             # Rotate lab‐frame Q into sample frame
#             Qc = (R_sample.T @ self.Q_lab.reshape(-1,3).T).T.reshape(ny, nx, 3)

#             UB_use = row.get('ub') if row.get('ub') is not None else self.UB
#             # Map to fractional HKL
#             hkl_flat = np.linalg.inv(UB_use) @ Qc.reshape(-1,3).T
#             hkl_map  = hkl_flat.T.reshape(ny, nx, 3)

#             Q_samp[idx]  = Qc
#             hkl_arr[idx] = hkl_map
#             I_arr[idx]   = I
#             print(f"Processed scan {row['scan_number']} frame {row['data_number']}", end="\r")

#         self.Q_samp    = Q_samp
#         self.hkl       = hkl_arr
#         self.intensity = I_arr
#         return Q_samp, hkl_arr, I_arr

#     def setup_grid(self, grid_ranges, grid_shape):
#         self.grid_ranges = grid_ranges
#         self.grid_shape = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         if space == 'q':
#             pts, edges = self.Q_samp.reshape(-1,3), self.edges
#         else:
#             pts, edges = self.hkl.reshape(-1,3), self.hkl_edges
#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method=='sum':
#             return H_sum, edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             Hm = H_sum / H_cnt
#             Hm[np.isnan(Hm)] = 0
#         return Hm, edges

#     def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
#         arr = self.Q_samp if space=='q' else self.hkl
#         ranges = tuple((arr[...,k].min(), arr[...,k].max()) for k in range(3))
#         if space=='q':
#             self.setup_grid(ranges, grid_shape)
#         else:
#             self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i]+1)
#                               for i,r in enumerate(ranges)]
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
#         pts = (self.Q_samp if space=='q' else self.hkl).reshape(-1,3)
#         vals = self.intensity.ravel()
#         mask = vals>0
#         pts, vals = pts[mask], vals[mask]
#         mins, maxs = pts.min(axis=0), pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         XI, YI, ZI = np.meshgrid(*axes, indexing='ij')
#         G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0)
#         return G, axes

#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         Nf, ny, nx = self.intensity.shape
#         z0,z1 = (0,Nf-1) if z_bound is None else z_bound
#         y0,y1 = (0,ny-1) if y_bound is None else y_bound
#         x0,x1 = (0,nx-1) if x_bound is None else x_bound
#         Qc = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Hc = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Ic = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
#         return

#     # ... (regrid, regrid_auto, regrid_interpolate, crop_by_positions unchanged) ...

# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.spatial.transform import Rotation as _Rot

# try:
#     # Preferred public import path
#     from pyFAI import AzimuthalIntegrator
# except Exception:
#     # Fallback for older layouts
#     from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData


# # ─── RSMBuilder with pyFAI Q‐mapping ────────────────────────────────────────────
# class RSMBuilder:
#     """
#     Builds 3D reciprocal‐space maps from SPEC + TIFF data using pyFAI for Q‐mapping.

#     Assumptions about `self.setup` (from SpecParser):
#       - distance       : sample-to-detector distance in meters
#       - xcenter/ycenter: beam center in pixels
#       - pitch          : pixel size in meters (square pixels)
#       - wavelength     : wavelength in Å (Angstroms)  ← converted to meters for pyFAI
#       - xpixels/ypixels: detector dimensions in pixels

#     Parameters
#     ----------
#     spec_file : str
#         Path to SPEC file.
#     tiff_dir : str
#         Directory with TIFF frames.
#     poni_file : str or None
#         Optional pyFAI .poni calibration file. If provided, overrides setup geometry.
#     use_dask : bool
#         If True, load TIFFs with Dask (delegated to ReadData).
#     process_hklscan_only : bool
#         If True, keep only rows with type == 'hklscan'.
#     selected_scans : Iterable[int] or None
#         If provided, include only these scan numbers.

#     Notes
#     -----
#     - All internal Q vectors are in Å⁻¹ by default (k0 = 2π/λ with λ in Å).
#     - pyFAI geometry is configured in SI (meters, radians).
#     """

#     def __init__(
#         self,
#         spec_file,
#         tiff_dir,
#         poni_file=None,
#         use_dask=False,
#         process_hklscan_only=False,
#         selected_scans=None,
#         dtype=np.float32,
#     ):
#         self.dtype = dtype

#         # ── Parse SPEC and merge with image data ────────────────────────────────
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = np.asarray(exp.crystal.UB, dtype=np.float64)

#         df_meta = exp.to_pandas()
#         df_meta["scan_number"] = df_meta["scan_number"].astype(int)
#         df_meta["data_number"] = df_meta["data_number"].astype(int)

#         rd = ReadData(tiff_dir, use_dask=use_dask)
#         df_int = rd.load_data()  # expects columns: scan_number, data_number, intensity

#         df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")
#         if process_hklscan_only:
#             # safer: handle missing/NaN & case-insensitive
#             df = df[df["type"].str.lower().eq("hklscan", na=False)]
#         if selected_scans is not None:
#             selected_scans = set(selected_scans)
#             df = df[df["scan_number"].isin(selected_scans)]

#         if df.empty:
#             raise ValueError("No frames to process after filtering/merge.")

#         self.df = df.reset_index(drop=True)

#         # ── Detector shape from actual data ─────────────────────────────────────
#         # (more robust than relying exclusively on setup.xpixels/ypixels)
#         self.img_shape = tuple(int(x) for x in self.df["intensity"].iat[0].shape)
#         ny, nx = self.img_shape

#         # ── Instantiate pyFAI integrator ───────────────────────────────────────
#         if poni_file:
#             # Robust load of a PONI file with wavelength awareness
#             self.ai = AzimuthalIntegrator()
#             self.ai.load(poni_file)
#             # If your SPEC provides wavelength, prefer the SPEC wavelength:
#             lam_A = float(self.setup.wavelength)
#             if lam_A < 1e-3:  # looks like meters (e.g., 1e-10)
#                 lam_A *= 1e10  # convert m -> Å
#             self.ai.wavelength = lam_A * 1e-10  # meters
#         else:
#             # Build from ExperimentSetup (explicit SI units for pyFAI)
#             pitch_m = float(self.setup.pitch)  # MUST be meters
#             dist_m = float(self.setup.distance)
#             x0_pix = float(self.setup.xcenter)
#             y0_pix = float(self.setup.ycenter)

#             lam_A = float(self.setup.wavelength)
#             if lam_A < 1e-3:  # if user accidentally passed meters
#                 lam_A *= 1e10
#             lam_m = lam_A * 1e-10

#             # Convert beam center in pixels to PONI in meters
#             poni1_m = y0_pix * pitch_m  # vertical axis in pyFAI is pixel1 (row)
#             poni2_m = x0_pix * pitch_m  # horizontal axis is pixel2 (col)

#             self.ai = AzimuthalIntegrator(
#                 dist=dist_m,
#                 poni1=poni1_m,
#                 poni2=poni2_m,
#                 rot1=0.0,
#                 rot2=0.0,
#                 rot3=0.0,
#                 pixel1=pitch_m,
#                 pixel2=pitch_m,
#                 wavelength=lam_m,
#             )

#         # ── Precompute lab‐frame Q map at detector geometry (Å⁻¹) ──────────────
#         two_theta = self.ai.twoThetaArray(self.img_shape)  # radians
#         chi_det = self.ai.chiArray(self.img_shape)         # radians, [-π, π]

#         # k0 in Å⁻¹ for downstream crystallography
#         lam_A = (self.ai.wavelength or 0.0) * 1e10  # ai stores in meters
#         if lam_A <= 0:
#             # Fall back to setup value in Å if ai has missing wavelength
#             lam_A = float(self.setup.wavelength)
#         k0 = 2.0 * np.pi / lam_A

#         # Q in lab frame: (qx, qy, qz) in Å⁻¹
#         # k_out = k0 [sin(2θ) cos χ, sin(2θ) sin χ, cos(2θ)]
#         # Q = k_out − k_in = [ ... , ... , k0 (cos(2θ) − 1)]
#         sin2t = np.sin(two_theta).astype(self.dtype, copy=False)
#         cos2t = np.cos(two_theta).astype(self.dtype, copy=False)
#         cosc = np.cos(chi_det).astype(self.dtype, copy=False)
#         sinc = np.sin(chi_det).astype(self.dtype, copy=False)

#         qx = (k0 * sin2t * cosc).astype(self.dtype, copy=False)
#         qy = (k0 * sin2t * sinc).astype(self.dtype, copy=False)
#         qz = (k0 * (cos2t - 1.0)).astype(self.dtype, copy=False)

#         self.Q_lab = np.stack((qx, qy, qz), axis=-1)  # (ny, nx, 3), dtype=self.dtype

#     # ────────────────────────────────────────────────────────────────────────────
#     # Core computation
#     # ────────────────────────────────────────────────────────────────────────────
#     def compute_full(self, verbose=True):
#         """
#         For each frame, apply sample rotation and compute Q_sample & hkl per pixel.

#         Returns
#         -------
#         Q_sample : (Nf, ny, nx, 3) array, dtype=self.dtype, in Å⁻¹
#         hkl      : (Nf, ny, nx, 3) array, dtype=self.dtype
#         intensity: (Nf, ny, nx)     array, dtype=self.dtype
#         """
#         df = self.df
#         Nf = len(df)
#         ny, nx = self.img_shape

#         Q_sample = np.empty((Nf, ny, nx, 3), dtype=self.dtype)
#         hkl_arr = np.empty_like(Q_sample)
#         I_arr = np.empty((Nf, ny, nx), dtype=self.dtype)

#         invUB_default = np.linalg.inv(self.UB).astype(np.float64, copy=False)

#         for idx, row in df.iterrows():
#             I = np.asarray(row["intensity"], dtype=self.dtype, order="C")
#             if I.shape != (ny, nx):
#                 raise ValueError(
#                     f"Frame shape {I.shape} != expected {self.img_shape}"
#                 )

#             # Sample motors (degrees) -> rotation matrices
#             th_s = float(row["th"])
#             chi_s = float(row["chi"])
#             phi_s = float(row["phi"])

#             Rz = _Rot.from_euler("z", phi_s, degrees=True).as_matrix()
#             Ry = _Rot.from_euler("y", chi_s, degrees=True).as_matrix()
#             Rx = _Rot.from_euler("x", th_s, degrees=True).as_matrix()
#             R_sample = (Rz @ Ry @ Rx).astype(np.float64, copy=False)

#             # Rotate lab‐frame Q into sample frame:
#             # Q_sample = R_sample^T * Q_lab  <=>  Q_sample(row) = Q_lab(row) @ R_sample
#             Qc = (self.Q_lab @ R_sample).astype(self.dtype, copy=False)

#             # UB may be per frame; fall back to global UB
#             UB_use = row.get("ub", None)
#             if UB_use is not None:
#                 invUB = np.linalg.inv(np.asarray(UB_use, dtype=np.float64))
#             else:
#                 invUB = invUB_default

#             # Map to fractional HKL: hkl = inv(UB) * Q
#             # Row-vector form: hkl_row = Q_row @ invUB^T
#             hkl_map = (Qc @ invUB.T).astype(self.dtype, copy=False)

#             Q_sample[idx] = Qc
#             hkl_arr[idx] = hkl_map
#             I_arr[idx] = I

#             if verbose and (idx % 10 == 0 or idx == Nf - 1):
#                 print(f"Processed {idx+1}/{Nf} frames", end="\r")

#         self.Q_samp = Q_sample
#         self.hkl = hkl_arr
#         self.intensity = I_arr
#         return Q_sample, hkl_arr, I_arr

#     # ────────────────────────────────────────────────────────────────────────────
#     # Gridding utilities
#     # ────────────────────────────────────────────────────────────────────────────
#     def setup_grid(self, grid_ranges, grid_shape):
#         """
#         Manually set bin edges for histogramming in Q or HKL space.

#         Parameters
#         ----------
#         grid_ranges : ((min,max),(min,max),(min,max))
#         grid_shape  : (nq1, nq2, nq3)
#         """
#         if len(grid_ranges) != 3 or len(grid_shape) != 3:
#             raise ValueError("grid_ranges and grid_shape must be length-3 iterables.")
#         self.grid_ranges = tuple(tuple(r) for r in grid_ranges)
#         self.grid_shape = tuple(int(n) for n in grid_shape)
#         self.edges = [
#             np.linspace(r[0], r[1], self.grid_shape[i] + 1) for i, r in enumerate(self.grid_ranges)
#         ]

#     def regrid_intensity(self, method="sum", space="q"):
#         """
#         Histogram regridding in either Q or HKL space.

#         Parameters
#         ----------
#         method : {"sum", "mean"}
#         space  : {"q", "hkl"}

#         Returns
#         -------
#         volume, edges
#           volume: 3D array of shape derived from edges
#           edges : list of 3 edge arrays
#         """
#         if space not in {"q", "hkl"}:
#             raise ValueError("space must be 'q' or 'hkl'.")

#         if space == "q":
#             if not hasattr(self, "edges"):
#                 raise RuntimeError("Call setup_grid() or regrid_auto(space='q') first.")
#             pts, edges = self.Q_samp.reshape(-1, 3), self.edges
#         else:
#             if not hasattr(self, "hkl_edges"):
#                 raise RuntimeError("Call regrid_auto(space='hkl') first.")
#             pts, edges = self.hkl.reshape(-1, 3), self.hkl_edges

#         vals = self.intensity.ravel().astype(np.float64, copy=False)  # accumulate in higher precision
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)

#         if method == "sum":
#             return H_sum.astype(self.dtype, copy=False), edges

#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             Hm = H_sum / H_cnt
#             Hm[~np.isfinite(Hm)] = 0.0
#         return Hm.astype(self.dtype, copy=False), edges

#     def regrid_auto(self, space="q", grid_shape=(200, 200, 200), method="mean"):
#         """
#         Automatically set ranges from data min/max and histogram.
#         """
#         if space == "q":
#             arr = self.Q_samp
#             ranges = tuple((arr[..., k].min(), arr[..., k].max()) for k in range(3))
#             self.setup_grid(ranges, grid_shape)
#         else:
#             arr = self.hkl
#             ranges = tuple((arr[..., k].min(), arr[..., k].max()) for k in range(3))
#             self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i] + 1) for i, r in enumerate(ranges)]
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space="q", grid_shape=(200, 200, 200), method="linear"):
#         """
#         Interpolation-based regridding (slower, smooth). Only uses positive intensities.
#         """
#         arr = self.Q_samp if space == "q" else self.hkl
#         pts = arr.reshape(-1, 3)
#         vals = self.intensity.ravel()
#         mask = vals > 0
#         pts, vals = pts[mask], vals[mask]

#         mins, maxs = pts.min(axis=0), pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         XI, YI, ZI = np.meshgrid(*axes, indexing="ij")
#         G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0.0)
#         return G.astype(self.dtype, copy=False), axes

#     # ────────────────────────────────────────────────────────────────────────────
#     # Cropping
#     # ────────────────────────────────────────────────────────────────────────────
#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         """
#         Crop the (frame, y, x) cube by index ranges.

#         Parameters
#         ----------
#         z_bound, y_bound, x_bound : (start, stop) using inclusive indices
#         in_place : bool

#         Returns
#         -------
#         (Q_crop, hkl_crop, I_crop) if in_place is False; otherwise None
#         """
#         Nf, ny, nx = self.intensity.shape
#         z0, z1 = (0, Nf - 1) if z_bound is None else z_bound
#         y0, y1 = (0, ny - 1) if y_bound is None else y_bound
#         x0, x1 = (0, nx - 1) if x_bound is None else x_bound

#         # inclusive bounds -> slice stop is +1
#         Qc = self.Q_samp[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1, :]
#         Hc = self.hkl[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1, :]
#         Ic = self.intensity[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]

#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
#             return None
#         return Qc, Hc, Ic


# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from scipy.spatial.transform import Rotation as _Rot

# # try:
# #     from pyFAI import AzimuthalIntegrator
# # except Exception:
# #     from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData


# class RSMBuilder:
#     """
#     Build 3D reciprocal-space maps (Q, HKL) from SPEC+TIFF using pyFAI.

#     Assumptions about `exp.setup` coming from SpecParser:
#       - distance  [m]          : sample-detector distance
#       - xcenter,ycenter [pix]  : beam center (pixel indices, col=x, row=y)
#       - pitch    [m]           : pixel size (square pixels)
#       - wavelength [Å]         : beam wavelength (Å)
#       - xpixels, ypixels [pix] : detector shape

#     Notes
#     -----
#     * Q is computed in **Å⁻¹**.
#     * By default we assume UB/B is in the **no-2π convention**
#       (a* = 1/a, etc.). Then HKL uses Q/(2π). Set `ub_includes_2pi=True`
#       if your UB already includes 2π (a* = 2π/a).
#     * Q-components use pyFAI's native unit mapping (`array_from_unit`)
#       to avoid trig mistakes. See pyFAI tutorials on `qx/qy/qz`. 
#     """

#     def __init__(self,
#                  spec_file,
#                  tiff_dir,
#                  poni_file=None,
#                  use_dask=False,
#                  process_hklscan_only=False,
#                  selected_scans=None,
#                  dtype=np.float32,
#                  ub_includes_2pi=False,      # <-- flag to match your UB convention
#                  rotate_q_by_sample=True      # <-- set False if per-frame UB already contains orientation
#                  ):
#         self.dtype = dtype
#         self.ub_includes_2pi = bool(ub_includes_2pi)
#         self.rotate_q_by_sample = bool(rotate_q_by_sample)

#         # ── SPEC + TIFF merge ───────────────────────────────────────────────────
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = np.asarray(exp.crystal.UB, dtype=np.float64)

#         df_meta = exp.to_pandas()
#         df_meta["scan_number"] = df_meta["scan_number"].astype(int)
#         df_meta["data_number"] = df_meta["data_number"].astype(int)

#         rd = ReadData(tiff_dir, use_dask=use_dask)
#         df_int = rd.load_data()  # columns: scan_number, data_number, intensity

#         df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")
#         if process_hklscan_only:
#             df = df[df["type"].str.lower().eq("hklscan", na=False)]
#         if selected_scans is not None:
#             df = df[df["scan_number"].isin(set(selected_scans))]
#         if df.empty:
#             raise ValueError("No frames to process after filtering/merge.")
#         self.df = df.reset_index(drop=True)

#         # ── Detector shape from actual image ────────────────────────────────────
#         self.img_shape = tuple(int(v) for v in self.df["intensity"].iat[0].shape)
#         ny, nx = self.img_shape

#         # ── Build AzimuthalIntegrator ──────────────────────────────────────────
#         if poni_file:
#             ai = AzimuthalIntegrator()
#             ai.load(poni_file)
#             # Prefer SPEC wavelength if present (Å -> m)
#             lam_A = float(self.setup.wavelength)
#             if lam_A < 1e-3:  # looks like meters by mistake
#                 lam_A *= 1e10
#             ai.wavelength = lam_A * 1e-10
#         else:
#             pitch_m = float(self.setup.pitch)     # MUST be meters
#             dist_m = float(self.setup.distance)
#             x0_pix = float(self.setup.xcenter)
#             y0_pix = float(self.setup.ycenter)
#             lam_A = float(self.setup.wavelength)
#             if lam_A < 1e-3:
#                 lam_A *= 1e10
#             lam_m = lam_A * 1e-10
#             # pyFAI's PONI1 (vertical/pixel1) and PONI2 (horizontal/pixel2)
#             ai = AzimuthalIntegrator(
#                 dist=dist_m,
#                 poni1=y0_pix * pitch_m,
#                 poni2=x0_pix * pitch_m,
#                 rot1=0.0, rot2=0.0, rot3=0.0,
#                 pixel1=pitch_m, pixel2=pitch_m,
#                 wavelength=lam_m
#             )
#         self.ai = ai
        
#         print

#         # ── Q in lab frame from pyFAI (qx,qy,qz), in Å⁻¹ ───────────────────────
#         # pyFAI provides q-components as units; tutorial shows using "qx_nm^-1", etc.
#         # We request nm^-1 then convert to Å^-1 (÷10). 
#         # Ref: integrate2d in qx/qy space & fiber/GI units examples.

# # (ny, nx) from your first TIFF
#         # print(self.img_shape)
#         qx_nm = ai.array_from_unit(self.img_shape, "qx_nm^-1")
#         qy_nm = ai.array_from_unit(self.img_shape, "qy_nm^-1")
#         qz_nm = ai.array_from_unit(self.img_shape, "qz_nm^-1")

#         # conv = 0.1  # nm^-1 → Å^-1
#         # qx = (qx_nm * conv).astype(self.dtype)
#         # qy = (qy_nm * conv).astype(self.dtype)
#         # qz = (qz_nm * conv).astype(self.dtype)


#         # Sanity on shapes:
#         if qx_nm.shape != self.img_shape:
#             raise RuntimeError(f"pyFAI q-array shape {qx_nm.shape} != image shape {self.img_shape}")
#         conv = np.float32(0.1) if self.dtype == np.float32 else 0.1  # nm^-1 -> Å^-1
#         qx = (qx_nm * conv).astype(self.dtype, copy=False)
#         qy = (qy_nm * conv).astype(self.dtype, copy=False)
#         qz = (qz_nm * conv).astype(self.dtype, copy=False)
#         self.Q_lab = np.stack((qx, qy, qz), axis=-1)  # (ny, nx, 3), Å^-1

#     # ────────────────────────────────────────────────────────────────────────────
#     # Core: Q→HKL per frame
#     # ────────────────────────────────────────────────────────────────────────────
#     def compute_full(self, verbose=True):
#         """
#         For each frame: rotate Q_lab into sample frame (optional) and map to HKL.

#         Returns
#         -------
#         Q_sample : (Nf, ny, nx, 3)  in Å⁻¹ (== Q_lab if rotate_q_by_sample=False)
#         hkl      : (Nf, ny, nx, 3)
#         intensity: (Nf, ny, nx)
#         """
#         df = self.df
#         Nf = len(df)
#         ny, nx = self.img_shape

#         Q_sample = np.empty((Nf, ny, nx, 3), dtype=self.dtype)
#         HKL = np.empty_like(Q_sample)
#         Icube = np.empty((Nf, ny, nx), dtype=self.dtype)

#         invUB_default = np.linalg.inv(self.UB).astype(np.float64, copy=False)

#         # pre-reshape for speed
#         Qlab_flat = self.Q_lab.reshape(-1, 3).astype(np.float64, copy=False)

#         # Factor for 2π convention: if UB excludes 2π (default), divide Q by 2π
#         two_pi = 2.0 * np.pi
#         conv2pi = (1.0 if self.ub_includes_2pi else 1.0 / two_pi)

#         for idx, row in df.iterrows():
#             I = np.asarray(row["intensity"], dtype=self.dtype, order="C")
#             if I.shape != (ny, nx):
#                 raise ValueError(f"Frame shape {I.shape} != expected {self.img_shape}")

#             # Rotation from sample motors (degrees)
#             th = float(row["th"])
#             chi = float(row["chi"])
#             phi = float(row["phi"])
#             Rz = _Rot.from_euler("z", phi, degrees=True).as_matrix()
#             Ry = _Rot.from_euler("y", chi, degrees=True).as_matrix()
#             Rx = _Rot.from_euler("x", th,  degrees=True).as_matrix()
#             R_sample = (Rz @ Ry @ Rx)  # sample → lab active rotation

#             if self.rotate_q_by_sample:
#                 # Convert lab→sample coordinates: q_s_row = q_lab_row @ R_sample
#                 Qs = (Qlab_flat @ R_sample).reshape(ny, nx, 3).astype(self.dtype, copy=False)
#             else:
#                 Qs = self.Q_lab  # per-frame UB already encodes orientation

#             # Choose UB: per-frame 'ub' overrides default if present
#             UB_use = row.get("ub", None)
#             invUB = np.linalg.inv(np.asarray(UB_use, dtype=np.float64)) if UB_use is not None else invUB_default

#             # HKL mapping: use the proper 2π convention
#             # Row-vector form: hkl_row = (Q / (2π if needed)) @ invUB^T
#             hkl_map = (Qs.reshape(-1, 3).astype(np.float64, copy=False) * conv2pi) @ invUB.T
#             hkl_map = hkl_map.reshape(ny, nx, 3).astype(self.dtype, copy=False)

#             Q_sample[idx] = Qs
#             HKL[idx] = hkl_map
#             Icube[idx] = I

#             if verbose and (idx % 10 == 0 or idx == Nf - 1):
#                 print(f"Processed {idx+1}/{Nf} frames", end="\r")

#         self.Q_samp = Q_sample
#         self.hkl = HKL
#         self.intensity = Icube
#         return Q_sample, HKL, Icube

#     # ────────────────────────────────────────────────────────────────────────────
#     # Gridding
#     # ────────────────────────────────────────────────────────────────────────────
#     def setup_grid(self, grid_ranges, grid_shape):
#         if len(grid_ranges) != 3 or len(grid_shape) != 3:
#             raise ValueError("grid_ranges and grid_shape must be 3-tuples.")
#         self.grid_ranges = tuple(tuple(r) for r in grid_ranges)
#         self.grid_shape = tuple(int(n) for n in grid_shape)
#         self.edges = [
#             np.linspace(r[0], r[1], self.grid_shape[i] + 1) for i, r in enumerate(self.grid_ranges)
#         ]

#     def regrid_intensity(self, method="sum", space="q"):
#         if space not in {"q", "hkl"}:
#             raise ValueError("space must be 'q' or 'hkl'")
#         if space == "q":
#             if not hasattr(self, "edges"):
#                 raise RuntimeError("Call setup_grid() or regrid_auto(space='q') first.")
#             pts, edges = self.Q_samp.reshape(-1, 3), self.edges
#         else:
#             if not hasattr(self, "hkl_edges"):
#                 raise RuntimeError("Call regrid_auto(space='hkl') first.")
#             pts, edges = self.hkl.reshape(-1, 3), self.hkl_edges

#         vals = self.intensity.ravel().astype(np.float64, copy=False)
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method == "sum":
#             return H_sum.astype(self.dtype, copy=False), edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             Hm = H_sum / H_cnt
#             Hm[~np.isfinite(Hm)] = 0.0
#         return Hm.astype(self.dtype, copy=False), edges

#     def regrid_auto(self, space="q", grid_shape=(200, 200, 200), method="mean"):
#         arr = self.Q_samp if space == "q" else self.hkl
#         ranges = tuple((arr[..., k].min(), arr[..., k].max()) for k in range(3))
#         if space == "q":
#             self.setup_grid(ranges, grid_shape)
#         else:
#             self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i] + 1) for i, r in enumerate(ranges)]
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space="q", grid_shape=(200, 200, 200), method="linear"):
#         arr = self.Q_samp if space == "q" else self.hkl
#         pts = arr.reshape(-1, 3)
#         vals = self.intensity.ravel()
#         mask = vals > 0
#         pts, vals = pts[mask], vals[mask]
#         mins, maxs = pts.min(axis=0), pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         XI, YI, ZI = np.meshgrid(*axes, indexing="ij")
#         G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0.0)
#         return G.astype(self.dtype, copy=False), axes

#     # ────────────────────────────────────────────────────────────────────────────
#     # Cropping
#     # ────────────────────────────────────────────────────────────────────────────
#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         Nf, ny, nx = self.intensity.shape
#         z0, z1 = (0, Nf - 1) if z_bound is None else z_bound
#         y0, y1 = (0, ny - 1) if y_bound is None else y_bound
#         x0, x1 = (0, nx - 1) if x_bound is None else x_bound
#         Qc = self.Q_samp[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1, :]
#         Hc = self.hkl   [z0:z1 + 1, y0:y1 + 1, x0:x1 + 1, :]
#         Ic = self.intensity[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
# #         if in_place:
# #             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
# #             return None
# #         return Qc, Hc, Ic


# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata
# from rsm3d.spec_parser import SpecParser
# from rsm3d.data_io import ReadData
# import xrayutilities as xu

# _TWO_PI = 2.0 * np.pi

# class RSMBuilder:
#     """
#     3D RSM from SPEC + TIFF using xrayutilities QConversion.

#     Key assumptions (edit to match your diffractometer):
#       - beam along +y (r_i = (0,1,0))
#       - sample axes (outermost→innermost): Z(φ), Y(χ), X(θ)
#       - detector pixel axes: dir1 = x+ (columns, fastest), dir2 = z+ (rows)
#       - wavelength in Å; distance & pixel size (pitch) in meters; beam center in pixels

#     Parameters
#     ----------
#     ub_includes_2pi : bool
#         True  -> UB maps hkl→Q with 2π included (usual XU). (default: True)
#         False -> your UB is no-2π; we multiply by 2π before passing to XU.
#     center_is_one_based : bool
#         If your xcenter/ycenter are 1-based (common in some tools), set True.
#     """

#     def __init__(self,
#                  spec_file,
#                  tiff_dir,
#                  *,
#                  use_dask=False,
#                  process_hklscan_only=False,
#                  selected_scans=None,
#                  ub_includes_2pi=True,
#                  center_is_one_based=False,
#                  dtype=np.float32):
#         self.dtype = dtype
#         self.ub_includes_2pi = bool(ub_includes_2pi)

#         # ── SPEC + TIFF merge
#         exp = SpecParser(spec_file)
#         self.setup = exp.setup
#         self.UB = np.asarray(exp.crystal.UB, dtype=np.float64)

#         df_meta = exp.to_pandas()
#         df_meta["scan_number"] = df_meta["scan_number"].astype(int)
#         df_meta["data_number"] = df_meta["data_number"].astype(int)

#         rd = ReadData(tiff_dir, use_dask=use_dask)
#         df_int = rd.load_data()

#         df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")
#         if process_hklscan_only:
#             df = df[df["type"].str.lower().eq("hklscan", na=False)]
#         if selected_scans is not None:
#             df = df[df["scan_number"].isin(set(selected_scans))]
#         if df.empty:
#             raise ValueError("No frames to process after filtering/merge.")
#         self.df = df.reset_index(drop=True)

#         # ── image shape and geometry normalization
#         ny, nx = df["intensity"].iat[0].shape
#         self.img_shape = (ny, nx)

#         # wavelength (Å) for xrayutilities; allow energy if provided by SpecParser
#         lam_A = (getattr(self.setup, "wavelength", None) or 0.0)
#         if lam_A and lam_A < 1e-3:  # user provided meters by mistake
#             lam_A *= 1e10
#         if not lam_A and hasattr(self.setup, "energy_keV") and self.setup.energy_keV:
#             lam_A = 12.398419843320026 / float(self.setup.energy_keV)

#         if lam_A <= 0:
#             raise ValueError("Need wavelength in Å or energy in keV in setup.")

#         # distance & pixel size: use meters for both (any consistent unit is fine)
#         dist_m  = float(self.setup.distance)
#         pitch_m = float(self.setup.pitch)

#         # beam center (pixels). convert to 0-based if necessary
#         x0 = float(self.setup.xcenter) - (1.0 if center_is_one_based else 0.0)  # columns
#         y0 = float(self.setup.ycenter) - (1.0 if center_is_one_based else 0.0)  # rows

#         # ── Configure xrayutilities QConversion for an area detector
#         # sample axes: outermost → innermost; our motors are φ (Z), χ (Y), θ (X)
#         sampleAxis = ['z+', 'y+', 'x+']   # order matters! (φ, χ, θ)
#         detectorAxis = []                 # no explicit detector rotation circles
#         r_i = (0, 1, 0)                   # beam along +y (XU default)   [oai_citation:6‡xrayutilities.sourceforge.io](https://xrayutilities.sourceforge.io/_modules/xrayutilities/experiment.html)

#         self.qconv = xu.experiment.QConversion(sampleAxis, detectorAxis, r_i, wl=lam_A)
#         # detectorDir1 is fastest varying → match NumPy's x/columns:
#         # dir1 = x+, dir2 = z+ ; center channels cch1 (x), cch2 (y)
#         self.qconv.init_area(
#             'x+', 'z+',
#             cch1=x0, cch2=y0, Nch1=nx, Nch2=ny,
#             distance=dist_m, pwidth1=pitch_m, pwidth2=pitch_m,
#             detrot=0.0, tiltazimuth=0.0, tilt=0.0
#         )
#         # self.qconv.init_area(
#         #     'z+', 'x+',              # detectorDir1=rows (slow), detectorDir2=cols (fast)
#         #     cch1=y0, cch2=x0,        # center channels match those axes
#         #     Nch1=ny, Nch2=nx,
#         #     distance=dist_m, pwidth1=pitch_m, pwidth2=pitch_m,
#         #     detrot=0.0, tiltazimuth=0.0, tilt=0.0
#         #     )
#         # API references: QConversion, init_area, area.  [oai_citation:7‡xrayutilities.sourceforge.io](https://xrayutilities.sourceforge.io/_modules/xrayutilities/experiment.html)

#     def compute_full(self, verbose=True):
#         """
#         For each frame, compute per-pixel Q (Å⁻¹) and HKL via xrayutilities.

#         Returns
#         -------
#         Q_samp : (Nf, ny, nx, 3) float32 (Å⁻¹)
#         hkl    : (Nf, ny, nx, 3) float32
#         I      : (Nf, ny, nx)    float32
#         """
#         df = self.df
#         Nf = len(df)
#         ny, nx = self.img_shape

#         Q_samp = np.empty((Nf, ny, nx, 3), dtype=self.dtype)
#         HKL    = np.empty_like(Q_samp)
#         Icube  = np.empty((Nf, ny, nx), dtype=self.dtype)

#         # If your UB is no-2π, upgrade it to XU's 2π convention before passing in
#         UB2pi_default = (self.UB if self.ub_includes_2pi else (_TWO_PI * self.UB))

#         for i, row in df.iterrows():
#             I = np.asarray(row["intensity"], dtype=self.dtype, order="C")
#             if I.shape != (ny, nx):
#                 raise ValueError(f"Frame shape {I.shape} != expected {(ny, nx)}")

#             # angles in degrees; pass in order of sampleAxis: (φ, χ, θ)
#             # --- inside compute_full loop, replacing the qpos/hklpos block ---

# # angles must be in the order of sampleAxis you configured (Z φ, Y χ, X θ)
#             phi = float(row["phi"])
#             chi = float(row["chi"])
#             th  = float(row["th"])

# # If your UB is "no-2π", pre-multiply to match XU’s 2π convention
#             UB_row = row.get("ub", None)
#             UB2pi  = (np.asarray(UB_row, dtype=np.float64) if UB_row is not None else self.UB)
#             if not self.ub_includes_2pi:
#                 UB2pi = 2.0 * np.pi * UB2pi

#             # 1) Q: area() returns a tuple of 2D arrays (Nch1, Nch2)
#             qx, qy, qz = self.qconv.area(phi, chi, th, wl=self.qconv.wavelength, deg=True)

#             # Your images are (ny, nx). With init_area(Nch1=nx, Nch2=ny) the tuple is (nx, ny),
#             # so transpose each to match (ny, nx) and then stack:
#             Qf = np.stack((qx.T, qy.T, qz.T), axis=-1).astype(self.dtype, copy=False)

#             # 2) HKL directly from XU by passing UB -> returns (h, k, l) arrays, same shape
#             h, k, l = self.qconv.area(phi, chi, th, wl=self.qconv.wavelength, deg=True, UB=UB2pi)
#             HKLf = np.stack((h.T, k.T, l.T), axis=-1).astype(self.dtype, copy=False)

#             Q_samp[i] = Qf
#             HKL[i]    = HKLf
#             Icube[i]  = I
#             # th  = float(row["th"])
#             # chi = float(row["chi"])
#             # phi = float(row["phi"])

#             # # per-frame UB override allowed
#             # UB_row = row.get("ub", None)
#             # UB2pi  = (np.asarray(UB_row, dtype=np.float64) if UB_row is not None else UB2pi_default)
#             # if not self.ub_includes_2pi and UB_row is not None:
#             #     UB2pi = _TWO_PI * UB2pi

#             # # Q in Å^-1 (returns flat list with detectorDir1 fastest → x fastest)
#             # qpos = self.qconv.area(phi, chi, th, wl=self.qconv.wavelength, deg=True)
#             # # reshape to (ny, nx, 3)
#             # qpos = qpos.reshape(ny, nx, 3)   # dir2 (rows) slow, dir1 (cols) fast

#             # # HKL directly from xrayutilities by giving UB (faster & consistent)
#             # hklpos = self.qconv.area(phi, chi, th, wl=self.qconv.wavelength, deg=True, UB=UB2pi)
#             # hklpos = hklpos.reshape(ny, nx, 3)

#             # Q_samp[i] = qpos.astype(self.dtype, copy=False)
#             # HKL[i]    = hklpos.astype(self.dtype, copy=False)
#             # Icube[i]  = I

#             if verbose and (i % 10 == 0 or i == Nf - 1):
#                 print(f"Processed {i+1}/{Nf} frames", end="\r")

#         self.Q_samp = Q_samp
#         self.hkl    = HKL
#         self.intensity = Icube
#         return Q_samp, HKL, Icube

#     # ── regridding helpers (unchanged logic, kept from your version)
#     def setup_grid(self, grid_ranges, grid_shape):
#         self.grid_ranges = grid_ranges
#         self.grid_shape = grid_shape
#         self.edges = [
#             np.linspace(r[0], r[1], grid_shape[i] + 1)
#             for i, r in enumerate(grid_ranges)
#         ]

#     def regrid_intensity(self, method='sum', space='q'):
#         if space == 'q':
#             if not hasattr(self, "edges"):
#                 raise RuntimeError("Call setup_grid() or regrid_auto(space='q') first.")
#             pts, edges = self.Q_samp.reshape(-1,3), self.edges
#         else:
#             if not hasattr(self, "hkl_edges"):
#                 raise RuntimeError("Call regrid_auto(space='hkl') first.")
#             pts, edges = self.hkl.reshape(-1,3), self.hkl_edges
#         vals = self.intensity.ravel()
#         H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
#         if method=='sum':
#             return H_sum.astype(self.dtype, copy=False), edges
#         H_cnt, _ = np.histogramdd(pts, bins=edges)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             Hm = H_sum / H_cnt
#             Hm[~np.isfinite(Hm)] = 0
#         return Hm.astype(self.dtype, copy=False), edges

#     def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
#         arr = self.Q_samp if space=='q' else self.hkl
#         ranges = tuple((arr[...,k].min(), arr[...,k].max()) for k in range(3))
#         if space=='q':
#             self.setup_grid(ranges, grid_shape)
#         else:
#             self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i]+1)
#                               for i,r in enumerate(ranges)]
#         return self.regrid_intensity(method=method, space=space)

#     def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
#         pts = (self.Q_samp if space=='q' else self.hkl).reshape(-1,3)
#         vals = self.intensity.ravel()
#         mask = vals>0
#         pts, vals = pts[mask], vals[mask]
#         mins, maxs = pts.min(axis=0), pts.max(axis=0)
#         axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
#         XI, YI, ZI = np.meshgrid(*axes, indexing='ij')
#         G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0)
#         return G.astype(self.dtype, copy=False), axes

#     def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
#         Nf, ny, nx = self.intensity.shape
#         z0,z1 = (0,Nf-1) if z_bound is None else z_bound
#         y0,y1 = (0,ny-1) if y_bound is None else y_bound
#         x0,x1 = (0,nx-1) if x_bound is None else x_bound
#         Qc = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Hc = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
#         Ic = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
#         if in_place:
#             self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
#             return None
#         return Qc, Hc, Ic


# rsm3d_xu_fourc.py — RSM builder for 4-circle geometry using xrayutilities

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xrayutilities as xu

from rsm3d.spec_parser import SpecParser
from rsm3d.data_io import ReadData

_TWO_PI = 2.0 * np.pi

def _energy_keV_to_lambda_A(E_keV: float) -> float:
    """λ[Å] = 12.398419843320026 / E[keV]."""
    return 12.398419843320026 / float(E_keV)

class RSMBuilder:
    """
    3D reciprocal-space maps (Q, HKL) from SPEC + TIFF using xrayutilities
    configured for a 4-circle diffractometer with area detector.

    Geometry (defaults):
      - Four-circle ZXZ: φ(Z) → χ(X) → ω(Z).
      - Beam along +Y (xrayutilities default).
      - Detector axes set so per-pixel arrays come back as (ny, nx) (match image).
      - Units: wavelength in Å; distance & pixel size in meters; beam center in pixels (0-based).

    Parameters
    ----------
    spec_file : str
    tiff_dir  : str
    use_dask  : bool
    process_hklscan_only : bool
    selected_scans : Iterable[int] | None
    ub_includes_2pi : bool
        True  -> your UB uses a* = 2π/a (XU’s convention). (default True)
        False -> your UB is “no-2π”; we multiply by 2π before passing to XU.
    center_is_one_based : bool
        Set True if beam center (xcenter/ycenter) in SPEC is 1-based; converted to 0-based.
    fourc_mode : {"ZXZ","ZYX"}
        ZXZ: sampleAxis=['z+','x+','z+'] (φ, χ, ω)
        ZYX: sampleAxis=['z+','y+','x+'] (φ, χ, ω)
    motor_map : dict
        Column names in df for the motors, defaults: {"omega":"th","chi":"chi","phi":"phi"}
    dtype : numpy dtype
    """

    def __init__(
        self,
        spec_file,
        tiff_dir,
        *,
        use_dask: bool = False,
        process_hklscan_only: bool = False,
        selected_scans=None,
        ub_includes_2pi: bool = True,
        center_is_one_based: bool = False,
        fourc_mode: str = "ZXZ",
        motor_map: dict | None = None,
        dtype=np.float32,
    ):
        self.dtype = dtype
        self.ub_includes_2pi = bool(ub_includes_2pi)

        # ── SPEC + TIFF merge
        exp = SpecParser(spec_file)
        self.setup = exp.setup
        self.UB = np.asarray(exp.crystal.UB, dtype=np.float64)

        df_meta = exp.to_pandas()
        df_meta["scan_number"] = df_meta["scan_number"].astype(int)
        df_meta["data_number"] = df_meta["data_number"].astype(int)

        rd = ReadData(tiff_dir, use_dask=use_dask)
        df_int = rd.load_data()

        df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")
        if process_hklscan_only:
            df = df[df["type"].str.lower().eq("hklscan", na=False)]
        if selected_scans is not None:
            df = df[df["scan_number"].isin(set(selected_scans))]
        if df.empty:
            raise ValueError("No frames to process after filtering/merge.")
        self.df = df.reset_index(drop=True)

        # ── image shape and geometry
        ny, nx = df["intensity"].iat[0].shape
        self.img_shape = (ny, nx)

        # wavelength (Å) from setup or energy (keV)
        lam_A = float(getattr(self.setup, "wavelength", 0.0) or 0.0)
        if lam_A and lam_A < 1e-3:  # meters by mistake → Å
            lam_A *= 1e10
        if lam_A <= 0.0 and getattr(self.setup, "energy_keV", None):
            lam_A = _energy_keV_to_lambda_A(float(self.setup.energy_keV))
        if lam_A <= 0.0:
            raise ValueError("Need positive wavelength (Å) or energy (keV) in setup.")

        # distance & pixel size in meters (consistent units for XU)
        dist_m  = float(self.setup.distance)
        pitch_m = float(self.setup.pitch)

        # beam center (pixels) → 0-based if needed
        x0 = float(self.setup.xcenter) - (1.0 if center_is_one_based else 0.0)  # cols (x)
        y0 = float(self.setup.ycenter) - (1.0 if center_is_one_based else 0.0)  # rows (z in our choice below)

        # ── 4-circle axis configuration
        fourc_mode = fourc_mode.upper()
        if fourc_mode not in {"ZXZ", "ZYX"}:
            raise ValueError("fourc_mode must be 'ZXZ' or 'ZYX'.")

        # sample axes (outer → inner) and angle order for area()
        # Angles we pass are always (phi, chi, omega) to match these lists.
        if fourc_mode == "ZXZ":
            # φ about Z, χ about X, ω about Z
            sampleAxis = ['z+', 'x+', 'z+']
        else:  # "ZYX"
            # φ about Z, χ about Y, ω about X
            sampleAxis = ['z+', 'y+', 'x+']

        # beam along +Y
        r_i = (0, 1, 0)
        self.qconv = xu.experiment.QConversion(sampleAxis, [], r_i, wl=lam_A)

        # detector axes so returned arrays are (ny, nx) (no transpose needed):
        # Dir1 (slow axis) = rows = 'z+' (Nch1=ny, cch1=y0)
        # Dir2 (fast axis) = cols = 'x+' (Nch2=nx, cch2=x0)
        self.qconv.init_area(
            'z+', 'x+',
            cch1=y0, cch2=x0,
            Nch1=ny, Nch2=nx,
            distance=dist_m,
            pwidth1=pitch_m, pwidth2=pitch_m,
            detrot=0.0, tiltazimuth=0.0, tilt=0.0
        )

        # motor names mapping
        default_motor_map = {"omega": "th", "chi": "chi", "phi": "phi"}
        self.motor_map = {**default_motor_map, **(motor_map or {})}

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

        for i, row in df.iterrows():
            I = np.asarray(row["intensity"], dtype=self.dtype, order="C")
            if I.shape != (ny, nx):
                raise ValueError(f"Frame shape {I.shape} != expected {(ny, nx)}")

            # pull motors with mapping
            omega = float(row[self.motor_map["omega"]])
            chi   = float(row[self.motor_map["chi"]])
            phi   = float(row[self.motor_map["phi"]])

            # Q in Å^-1: area() returns tuple of arrays (qx, qy, qz), each (ny, nx)
            # IMPORTANT: pass angles in the order of sampleAxis → (phi, chi, omega)
            qx, qy, qz = self.qconv.area(phi, chi, omega, wl=self.qconv.wavelength, deg=True)
            Qf = np.stack((qx, qy, qz), axis=-1).astype(self.dtype, copy=False)

            # HKL via UB (2π convention for XU). Allow per-frame UB override.
            UB_row = row.get("ub", None)
            UB2pi = np.asarray(UB_row, dtype=np.float64) if UB_row is not None else UB2pi_default
            if not self.ub_includes_2pi and UB_row is not None:
                UB2pi = _TWO_PI * UB2pi

            h, k, l = self.qconv.area(phi, chi, omega, wl=self.qconv.wavelength, deg=True, UB=UB2pi)
            HKLf = np.stack((h, k, l), axis=-1).astype(self.dtype, copy=False)

            Q_samp[i] = Qf
            HKL[i]    = HKLf
            Icube[i]  = I

            if verbose and (i % 10 == 0 or i == Nf - 1):
                print(f"Processed {i+1}/{Nf} frames", end="\r")

        self.Q_samp = Q_samp
        self.hkl    = HKL
        self.intensity = Icube
        return Q_samp, HKL, Icube
    def regrid_xu(
        self,
        *,
        space: str = "q",                 # "q" or "hkl"
        grid_shape=(200, 200, 200),       # (nx, ny, nz)
        ranges=None,                      # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or None
        fuzzy: bool = False,              # use FuzzyGridder3D
        width=None,                       # scalar or (wx,wy,wz) for fuzzy (same units as axes)
        normalize: str = "mean",          # "mean" → averaged; "sum" → accumulated
        stream: bool = False              # iterate frame-by-frame to save RAM
    ):
        import xrayutilities as xu

        assert space.lower() in ("q", "hkl")
        nx, ny, nz = map(int, grid_shape)
        arr = self.Q_samp if space.lower() == "q" else self.hkl

        G = (xu.FuzzyGridder3D if fuzzy else xu.Gridder3D)(nx, ny, nz)

        # If you’ll feed multiple chunks, keep intermediate state
        if stream:
            G.KeepData(True)

        # Optional fixed range (recommended for streaming)
        if ranges is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = ranges
            G.dataRange(xmin, xmax, ymin, ymax, zmin, zmax, fixed=True)  # fixed range gridding  [oai_citation:1‡xrayutilities.sourceforge.io](https://xrayutilities.sourceforge.io/_modules/xrayutilities/gridder3d.html)

        # Feed points
        if stream:
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
            X = arr[..., 0].ravel()
            Y = arr[..., 1].ravel()
            Z = arr[..., 2].ravel()
            W = self.intensity.ravel()
            if fuzzy and width is not None:
                G(X, Y, Z, W, width=width)
            else:
                G(X, Y, Z, W)

        # Toggle normalization then always read .data
        if normalize.lower() == "sum":
            G.Normalize(False)   # unnormalized → sums in .data
        else:
            G.Normalize(True)    # normalized → means in .data

        grid = G.data.astype(self.dtype, copy=False)   # official attribute for gridded data  [oai_citation:2‡xrayutilities.sourceforge.io](https://xrayutilities.sourceforge.io/_modules/xrayutilities/gridder.html)
        xax, yax, zax = G.xaxis, G.yaxis, G.zaxis
        return grid, (xax, yax, zax)
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
    #     stream: bool = False
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
    def setup_grid(self, grid_ranges, grid_shape):
        self.grid_ranges = grid_ranges
        self.grid_shape = grid_shape
        self.edges = [
            np.linspace(r[0], r[1], grid_shape[i] + 1)
            for i, r in enumerate(grid_ranges)
        ]

    def regrid_intensity(self, method='sum', space='q'):
        if space == 'q':
            if not hasattr(self, "edges"):
                raise RuntimeError("Call setup_grid() or regrid_auto(space='q') first.")
            pts, edges = self.Q_samp.reshape(-1,3), self.edges
        else:
            if not hasattr(self, "hkl_edges"):
                raise RuntimeError("Call regrid_auto(space='hkl') first.")
            pts, edges = self.hkl.reshape(-1,3), self.hkl_edges
        vals = self.intensity.ravel().astype(np.float64, copy=False)
        H_sum, _ = np.histogramdd(pts, bins=edges, weights=vals)
        if method=='sum':
            return H_sum.astype(self.dtype, copy=False), edges
        H_cnt, _ = np.histogramdd(pts, bins=edges)
        with np.errstate(divide='ignore', invalid='ignore'):
            Hm = H_sum / H_cnt
            Hm[~np.isfinite(Hm)] = 0
        return Hm.astype(self.dtype, copy=False), edges

    def regrid_auto(self, space='q', grid_shape=(200,200,200), method='mean'):
        arr = self.Q_samp if space=='q' else self.hkl
        ranges = tuple((arr[...,k].min(), arr[...,k].max()) for k in range(3))
        if space=='q':
            self.setup_grid(ranges, grid_shape)
        else:
            self.hkl_edges = [np.linspace(r[0], r[1], grid_shape[i]+1)
                              for i,r in enumerate(ranges)]
        return self.regrid_intensity(method=method, space=space)

    def regrid_interpolate(self, space='q', grid_shape=(200,200,200), method='linear'):
        pts = (self.Q_samp if space=='q' else self.hkl).reshape(-1,3)
        vals = self.intensity.ravel()
        mask = vals>0
        pts, vals = pts[mask], vals[mask]
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        axes = [np.linspace(mins[d], maxs[d], grid_shape[d]) for d in range(3)]
        XI, YI, ZI = np.meshgrid(*axes, indexing='ij')
        G = griddata(pts, vals, (XI, YI, ZI), method=method, fill_value=0)
        return G.astype(self.dtype, copy=False), axes

    def crop_by_positions(self, z_bound=None, y_bound=None, x_bound=None, in_place=True):
        Nf, ny, nx = self.intensity.shape
        z0,z1 = (0,Nf-1) if z_bound is None else z_bound
        y0,y1 = (0,ny-1) if y_bound is None else y_bound
        x0,x1 = (0,nx-1) if x_bound is None else x_bound
        Qc = self.Q_samp[z0:z1+1, y0:y1+1, x0:x1+1, :]
        Hc = self.hkl   [z0:z1+1, y0:y1+1, x0:x1+1, :]
        Ic = self.intensity[z0:z1+1, y0:y1+1, x0:x1+1]
        if in_place:
            self.Q_samp, self.hkl, self.intensity = Qc, Hc, Ic
            return None
        return Qc, Hc, Ic