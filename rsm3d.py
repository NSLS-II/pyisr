import os
import numpy as np
import pandas as pd
import xrayutilities as xu
import tifffile
from scipy.interpolate import griddata
import vtk
from vtk.util import numpy_support


def read_data(spec_path, data_dir, filename_pattern):
    """
    Reads a SPEC file and corresponding TIFF images, returning a DataFrame with columns ['Intensity', 'h', 'k', 'l'].
    
    Parameters
    ----------
    spec_path : str
        Path to the SPEC file.
    data_dir : str
        Directory containing TIFF images.
    filename_pattern : str
        Pattern for TIFF filenames, e.g., 'setup_6oct23_{scan:03d}_{data:03d}_data_000001.tiff'.
        Use '{scan}' and '{data}' as placeholders.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['Intensity', 'h', 'k', 'l'].
    """
    s = xu.io.SPECFile(spec_path)
    records = []
    for scan in s:
        scan.ReadData()
        scan_number = scan.nr
        for i in range(len(scan.data)):
            data_number = i
            h = scan.data[i][1]
            k = scan.data[i][2]
            l = scan.data[i][3]
            filename = filename_pattern.format(scan=int(scan_number), data=int(data_number))
            tiff_path = os.path.join(data_dir, filename)
            data = tifffile.imread(tiff_path)
            record = {'Intensity': data, 'h': h, 'k': k, 'l': l}
            records.append(record)
    df = pd.DataFrame(records)
    return df


def build_3d_rsm(df: pd.DataFrame,
                nh: int = 100, nk: int = 100, nl: int = 100,
                method: str = 'linear'):
    """
    Interpolate scattered (h,k,l,Intensity) → regular 3D HKL volume.
    """
    if method not in ('linear', 'nearest'):
        raise ValueError("method must be 'linear' or 'nearest' for 3D")

    h = df['h'].to_numpy()
    k = df['k'].to_numpy()
    l = df['l'].to_numpy()
    I = df['Intensity'].to_numpy()

    # build regular HKL grid
    hl = np.linspace(h.min(), h.max(), nh)
    kl = np.linspace(k.min(), k.max(), nk)
    ll = np.linspace(l.min(), l.max(), nl)
    Hg, Kg, Lg = np.meshgrid(hl, kl, ll, indexing='ij')

    # vectorized interpolation
    sample_pts = np.column_stack((h, k, l))
    grid_pts   = np.column_stack((Hg.ravel(), Kg.ravel(), Lg.ravel()))
    RSM_flat = griddata(sample_pts, I, grid_pts,
                        method=method, fill_value=0)
    RSM = RSM_flat.reshape((nh, nk, nl))

    return Hg, Kg, Lg, RSM

# ──────────────────────────────────────────────────────────────────────────────
# 2) Convert HKL grid → Q-space grid (orthogonal cell)
# ──────────────────────────────────────────────────────────────────────────────
def convert_hkl_to_Q(Hg, Kg, Lg, a: float, b: float, c: float):
    """
    Qx = 2π·H/a, Qy = 2π·K/b, Qz = 2π·L/c
    """
    factor = 2 * np.pi
    Qx = factor * Hg / a
    Qy = factor * Kg / b
    Qz = factor * Lg / c
    return Qx, Qy, Qz

# ──────────────────────────────────────────────────────────────────────────────
# 3) Convert 2D detector pixels → Q-space map for a single frame
# ──────────────────────────────────────────────────────────────────────────────
def pixel_to_Q_map(D: float,
                   p: float,
                   m0: float, n0: float,
                   Mx: int, Ny: int,
                   wl: float = None,
                   energy_ev: float = None,
                   phi_d: float = 0.0,
                   th0: float = 0.0,
                   dth: float = 0.0,
                   scan_index: int = 0):
    """
    Returns Qx,Qy,Qz,Qmag arrays of shape (Ny, Mx).
    """
    # wavelength
    if wl is None:
        if energy_ev is None:
            raise ValueError("Supply wl or energy_ev")
        wl = 12.3984196 / (energy_ev/1000.0)

    # pixel coords in detector plane
    xs = (np.arange(Mx) - n0) * p
    ys = (np.arange(Ny) - m0) * p
    X, Y = np.meshgrid(xs, ys)

    k0 = 2*np.pi / wl
    R = np.sqrt(X**2 + Y**2 + D**2)

    # k_out in detector frame
    kout_x = k0 * (X / R)
    kout_y = k0 * (Y / R)
    kout_z = k0 * (D / R)
    # Q in detector frame
    Qx_det = kout_x
    Qy_det = kout_y
    Qz_det = kout_z - k0

    # rotations: φ_d then θ = th0 + scan_index*dth, about detector Y-axis
    phi   = np.deg2rad(phi_d)
    theta = np.deg2rad(th0 + scan_index*dth)
    R_phi = np.array([[ np.cos(phi), 0, -np.sin(phi)],
                      [          0., 1,           0.],
                      [ np.sin(phi), 0,  np.cos(phi)]])
    R_th  = np.array([[ np.cos(theta), 0, -np.sin(theta)],
                      [            0., 1,            0.],
                      [ np.sin(theta), 0,  np.cos(theta)]])
    R_comb = R_th.dot(R_phi)

    # apply rotation
    pts = np.stack([Qx_det.ravel(), Qy_det.ravel(), Qz_det.ravel()], axis=0)
    Qs  = R_comb.dot(pts)
    Qx = Qs[0].reshape((Ny, Mx))
    Qy = Qs[1].reshape((Ny, Mx))
    Qz = Qs[2].reshape((Ny, Mx))
    Qmag = np.sqrt(Qx**2 + Qy**2 + Qz**2)

    return Qx, Qy, Qz, Qmag

# ──────────────────────────────────────────────────────────────────────────────
# 4) Save a 3D volume in Q-space to a legacy VTK file (ParaView)
# ──────────────────────────────────────────────────────────────────────────────
def save_to_vtk(Qx, Qy, Qz, RSM, filename="rsm_Q.vtk"):
    """
    Writes RSM (nh×nk×nl) sampled at Qx,Qy,Qz into a .vtk
    """
    # assume RSM.shape = (nx, ny, nz) = Qx.shape
    nx, ny, nz = RSM.shape

    origin = (Qx.min(), Qy.min(), Qz.min())
    spacing = (Qx[1,0,0] - Qx[0,0,0],
               Qy[0,1,0] - Qy[0,0,0],
               Qz[0,0,1] - Qz[0,0,0])

    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(*origin)
    img.SetSpacing(*spacing)

    flat = RSM.flatten(order='F').astype(np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("Intensity")
    img.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(img)
    writer.Write()
    print(f"Saved VTK → {filename}")

