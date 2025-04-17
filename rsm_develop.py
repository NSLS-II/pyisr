#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from os.path import isfile, join
import re
import numpy as np
import imageio
import pyvista as pv
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def rsm(path: str, fnamepng: str, cmap='jet', opacity='linear', md: int = 0, nd: int = 0, el: int = 0, az: int = 0):
    # Input info
    sclfactor = [0.6, 0.45, 0.3]
    detector_info = {'D': 781.05, 'p': 0.75, 'm0': 257, 'n0': 515, 'Mx': 1030, 'Ny': 514}
    D = detector_info['D']
    p_val = detector_info['p']
    m0 = detector_info['m0']
    n0 = detector_info['n0']
    Nx = int(detector_info['Mx'])
    Ny = int(detector_info['Ny'])
    
    scan_info = {'wl': 0.283383, 'twoth_d': 4, 'phi_d': 0, 'th0': 15.3069, 'dth': 0.04}
    wl = scan_info['wl']
    twoth_d = np.radians(scan_info['twoth_d'])
    phi_d = np.radians(scan_info['phi_d'])
    th0 = np.radians(scan_info['th0'])
    dth = np.radians(scan_info['dth'])
    
    # Precompute conversion matrices for detector coordinates
    sd = np.array([np.cos(twoth_d) * np.cos(phi_d),
                   np.cos(twoth_d) * np.sin(phi_d),
                   np.sin(twoth_d)])
    xd = np.array([np.sin(phi_d), -np.cos(phi_d), 0])
    yd = np.array([-np.sin(twoth_d) * np.cos(phi_d),
                   -np.sin(twoth_d) * np.sin(phi_d),
                   np.cos(twoth_d)])
    
    rx = np.tile(np.arange(0, Nx) - m0, (Ny, 1)) * p_val
    ry = np.tile(np.arange(Ny, 0, -1) - n0, (Nx, 1)).T * p_val
    
    Rx = D * sd[0] + rx * xd[0] + ry * yd[0]
    Ry = D * sd[1] + rx * xd[1] + ry * yd[1]
    Rz = D * sd[2] + rx * xd[2] + ry * yd[2]
    R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    
    k = 2 * np.pi / wl
    Qnm_x = k * (Rx / R - 1)
    Qnm_y = k * (Ry / R)
    Qnm_z = k * (Rz / R)
    
    # Filter TIFF files that match the pattern, e.g. "setup_6oct23_000_961_data_000001.tiff"
    pattern = re.compile(r"setup_6oct23_(\d+)_(\d+)_data")
    tifFiles = [f for f in os.listdir(path)
                if isfile(join(path, f)) and pattern.search(f)]
    tifFiles.sort(key=lambda name: (int(pattern.search(name).group(1)),
                                      int(pattern.search(name).group(2))))
    
    N = len(tifFiles)
    if N == 0:
        raise RuntimeError("No TIFF files found matching the expected pattern in the directory.")
    
    # Allocate arrays for the 3D volume and dQ calculations
    A = np.zeros((Ny, Nx, N), dtype=float)
    dQx = np.zeros((Ny, Nx, N), dtype=float)
    dQy = np.zeros((Ny, Nx, N), dtype=float)
    dQz = np.zeros((Ny, Nx, N), dtype=float)
    
    # Compute the angle for each slice.
    th = th0 + np.arange(N) * dth
    
    print("Reading TIFF files and computing dQ arrays:")
    for i, fname in enumerate(tifFiles):
        print(f"  Processing file {i+1}/{N}: {fname}")
        dQx[:, :, i] = Qnm_x * np.cos(th[i]) + Qnm_z * np.sin(th[i])
        dQy[:, :, i] = Qnm_y  # Qnm_y remains constant for all slices.
        dQz[:, :, i] = -Qnm_x * np.sin(th[i]) + Qnm_z * np.cos(th[i])
        im = imageio.imread(join(path, fname))
        imarray = np.array(im, dtype=float)
        imarray[imarray < 0] = 0  # Replace negative intensities with 0.
        A[:, :, i] = imarray.copy()
    
    # Compute the logarithmic intensity array.
    Z = np.log10(A + 1)
    whalf = np.log10(np.amax(A) + 1)
    
    print("Building 3D structured grid using PyVista...")
    # Create a structured grid with the dQ arrays as point coordinates.
    mesh = pv.StructuredGrid(dQx, dQy, dQz)
    # VTK expects data in a specific (Fortran) order.
    mesh.point_data['values'] = Z.ravel(order='F')
    # Compute isosurfaces based on the provided scale factors.
    isos = mesh.contour(isosurfaces=np.array(sclfactor) * whalf)
    
    print("Rendering 3D plot...")
    plotter = pv.Plotter()
    plotter.add_mesh(isos, cmap=cmap, opacity=opacity, show_scalar_bar=True)
    plotter.show_grid(xtitle="Qx", ytitle="Qy", ztitle="Qz", color="green", font_family="arial")
    plotter.set_background("white")
    plotter.camera.azimuth = az
    plotter.camera.elevation = el
    plotter.show(screenshot=fnamepng + "_3D.png")
    
    print("Creating 2D projection plot...")
    # Sum the volume along axis 1 for a 2D projection.
    II = np.sum(A, axis=1)
    XX = dQx[:, m0 - 1, :]
    ZZ = dQz[:, m0 - 1, :]
    
    fig, ax = plt.subplots(constrained_layout=True)
    c = ax.contourf(XX, ZZ, np.log10(II + 1), 12, cmap="viridis")
    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel(r'log$_{10}$(I)')
    ax.contour(XX, ZZ, np.log10(II + 1), 12, colors="k", linewidths=0.3)
    ax.set_xlabel(r"$\Delta Q_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$\Delta Q_z$ ($\AA^{-1}$)")
    
    plt.savefig(fnamepng + "_2D.png", dpi=200)
    plt.show()
    
    print("Processing complete.")

if __name__ == "__main__":
    # Replace this with the absolute path to your TIFF folder.
    rsm("/path/to/your/tiff/folder/", "output_rsm", md=0, nd=0, el=30, az=45)