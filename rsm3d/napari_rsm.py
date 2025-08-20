#!/usr/bin/env python3
"""
Napari plugin for RSM processing using napari’s own IO and visualization.

This plugin:
  - Loads a folder of TIFF files using napari’s reader hook.
  - Computes the reciprocal space mapping (RSM) coordinate conversion.
  - Adds the processed 3D stack (after converting detector coordinates)
    and a 2D projection (summed stack) as image layers in napari.
  - Uses napari's built‐in screenshot function rather than matplotlib.
  
To test the plugin, run this file; a napari viewer will open and a dock widget will appear.
"""

import os
from os.path import isfile, join
import re
import numpy as np
import napari
import warnings
from magicgui import magicgui

# Fallback import for the napari reader hook.
try:
    from napari.plugins.io import napari_get_reader
except ImportError:
    from napari.plugins import read_data as napari_get_reader

warnings.filterwarnings("ignore", category=UserWarning)

@magicgui(call_button="Run RSM", layout="vertical",
          directory={'label': "TIFF Directory"},
          fnamepng={'label': "Output Filename Prefix"})
def rsm_plugin(
    directory: str = "",
    fnamepng: str = "output_rsm",
    cmap: str = "jet",
    opacity: float = 1.0,
    md: int = 0,
    nd: int = 0,
    el: int = 30,
    az: int = 45,
):
    # Validate directory
    if not directory or not os.path.isdir(directory):
        print("Please provide a valid directory path containing TIFF files.")
        return

    print("Starting RSM processing...")

    # --- Input Parameters ---
    sclfactor = [0.6, 0.45, 0.3]  # scale factors from original code
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
    
    # --- Precompute conversion matrices for detector coordinates ---
    sd = np.array([
        np.cos(twoth_d) * np.cos(phi_d),
        np.cos(twoth_d) * np.sin(phi_d),
        np.sin(twoth_d)
    ])
    xd = np.array([np.sin(phi_d), -np.cos(phi_d), 0])
    yd = np.array([
        -np.sin(twoth_d) * np.cos(phi_d),
        -np.sin(twoth_d) * np.sin(phi_d),
        np.cos(twoth_d)
    ])
    
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
    
    # --- Get TIFF files ---
    tifFiles = sorted([
        f for f in os.listdir(directory)
        if isfile(join(directory, f)) and f.lower().endswith(('.tif', '.tiff'))
    ])
    N = len(tifFiles)
    if N == 0:
        print("No TIFF files found in the directory.")
        return
    print(f"Found {N} TIFF files.")
    
    # --- Preallocate arrays ---
    A = np.zeros((Ny, Nx, N), dtype=float)
    dQx = np.zeros((Ny, Nx, N), dtype=float)
    dQy = np.zeros((Ny, Nx, N), dtype=float)
    dQz = np.zeros((Ny, Nx, N), dtype=float)
    
    # Compute per-slice angle:
    th = th0 + np.arange(N) * dth

    # --- Process each TIFF file using napari_get_reader ---
    for i, fname in enumerate(tifFiles):
        print(f"Processing file {i+1}/{N}: {fname}")
        dQx[:, :, i] = Qnm_x * np.cos(th[i]) + Qnm_z * np.sin(th[i])
        dQy[:, :, i] = Qnm_y  # constant for each slice
        dQz[:, :, i] = - Qnm_x * np.sin(th[i]) + Qnm_z * np.cos(th[i])
        
        filepath = join(directory, fname)
        reader = napari_get_reader(filepath)
        if reader is None:
            print(f"Cannot read file: {filepath}")
            continue
        im_list = reader(filepath)
        imarray = np.array(im_list[0], dtype=float)
        imarray[imarray < 0] = 0
        A[:, :, i] = imarray.copy()
    
    print("Finished processing TIFF files.")
    
    # Compute logarithmic intensities.
    Z = np.log10(A + 1)
    whalf = np.log10(np.amax(A) + 1)
    
    # --- Add layers to napari ---
    viewer = napari.current_viewer()
    if viewer is None:
        viewer = napari.Viewer()
    
    # Napari expects image data in (Z, Y, X) order.
    A3d = np.transpose(A, (2, 0, 1))
    viewer.add_image(
        A3d, 
        name="3D TIFF Stack",
        contrast_limits=[A3d.min(), A3d.max()],
        blending="additive",
        colormap=cmap
    )
    
    # Create a 2D projection by summing over the stack (axis 2 of A)
    projection = np.sum(A, axis=2)
    viewer.add_image(
        projection,
        name="2D Projection",
        colormap="viridis",
        contrast_limits=[projection.min(), projection.max()]
    )
    
    print("RSM processing complete. Layers added to napari.")
    
    # Optionally, capture a screenshot using napari's built-in function.
    screenshot_path = fnamepng + "_napari_2D.png"
    viewer.screenshot(path=screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

# --- Napari Plugin Hook ---
def napari_experimental_provide_dock_widget():
    return rsm_plugin