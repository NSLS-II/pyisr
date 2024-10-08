#!/usr/bin/python3
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import imageio
import numpy as np
import pyvista as pv
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)


def rsm(path: str, fnamepng: str, cmap: str or list = 'jet',
        op: list or str = 'linear', md: int = 0, nd: int = 0, el: int = 0, az: int = 0):

    # Input info
    sclfactor = [.6, .45, .3]
    detector_info = dict(D = 781.05, p = 0.75, m0 = 257, n0 = 515, Mx = 1030, Ny = 514)
    D = detector_info.get('D')
    p = detector_info.get('p')
    m0 = detector_info.get('m0')
    n0 = detector_info.get('n0')
    Nx = int(detector_info.get('Mx'))
    Ny = int(detector_info.get('Ny'))
    scan_info = dict(wl = 0.283383, twoth_d = 4, phi_d = 0, th0 = 15.3069, dth = 0.04)
    wl = scan_info.get('wl')
    twoth_d = np.radians(scan_info.get('twoth_d'))
    # np.radians= numpy command to change degrees to radians
    phi_d = np.radians(scan_info.get('phi_d'))
    th0 = np.radians(scan_info.get('th0'))
    dth = np.radians(scan_info.get('dth'))
    # D = detector_info.get('D')
    # p = detector_info.get('p')
    # m0 = detector_info.get('m0')
    # n0 = detector_info.get('n0')
    # Nx = int(detector_info.get('Mx'))
    # Ny = int(detector_info.get('Ny'))

    # wl = scan_info.get('wl')
    # twoth_d = np.radians(scan_info.get('twoth_d'))  # deg to rad
    # phi_d = np.radians(scan_info.get('phi_d'))
    # th0 = np.radians(scan_info.get('th0'))
    # 
    # dth = np.radians(scan_info.get('dth'))

    # Common matrices to all th values
    sd = np.array([np.cos(twoth_d) * np.cos(phi_d),
                   np.cos(twoth_d) * np.sin(phi_d),
                   np.sin(twoth_d)])

    xd = np.array([np.sin(phi_d), - np.cos(phi_d), 0])

    yd = np.array([-np.sin(twoth_d) * np.cos(phi_d),
                   -np.sin(twoth_d) * np.sin(phi_d),
                   np.cos(twoth_d)])

    rx = np.tile(np.arange(0, Nx) - m0, (Ny, 1)) * p
    ry = np.tile(np.arange(Ny, 0, -1) - n0, (Nx, 1)).T * p

    Rx = D * sd[0] + rx * xd[0] + ry * yd[0]
    Ry = D * sd[1] + rx * xd[1] + ry * yd[1]
    Rz = D * sd[2] + rx * xd[2] + ry * yd[2]
    R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)

    k = 2 * np.pi / wl
    Qnm_x = k * (Rx / R - 1)
    Qnm_y = k * (Ry / R)
    Qnm_z = k * (Rz / R)

    # 3D matrices
    tifFiles = sorted([f for f in listdir(path) if isfile(join(path, f))])
    # tifFiles = sorted([f for f in listdir(path) if isfile(join(path, f))])  # sort and name tiff files
    # print("N=")
    N = len(tifFiles)
    # print(len(tifFiles))
    # print(N)
    A = np.zeros((Ny, Nx, N), dtype=float)
    # print("A")
    # print(A.shape)
    dQx = np.zeros((Ny, Nx, N), dtype=float)
    dQy = np.zeros((Ny, Nx, N), dtype=float)
    dQz = np.zeros((Ny, Nx, N), dtype=float)
    im = np.zeros((Ny, Nx, N), dtype=float)
    # I added im= np.zeros((Ny, Nx, N), dtype=float)
    # print(dQx.shape)
    # print(dQy.shape)
    # print(dQz.shape)
    th = th0 + np.arange(0, N) * dth

    for i in range(N):

        dQx[:, :, i] = Qnm_x * np.cos(th[i]) + Qnm_z * np.sin(th[i])
        dQy[:, :, i] = Qnm_y[:, :]
        dQz[:, :, i] = - Qnm_x * np.sin(th[i]) + Qnm_z * np.cos(th[i])
        # print("dqx,dqy,dqz")
        # print(dQx.shape)
        # print(dQy.shape)
        # print(dQz.shape)
        # print("the number of files is N" "  N=")
        # print(N)
        im = imageio.imread(path + tifFiles[i])
        imarray = np.array(im, dtype=float)
        # print("the file path is =")
        # print(path)
        # print("the folder /tifFiles contains 22 files:")
        # print(tifFiles)
        # print("I use the following formula to read the tif files from the folder")
        # print("im = imageio.imread(path + tifFiles[i])")
        # print("im shape is =")
        # print(im.shape)
        # print(im.size)
        # print("N")
        # print(N)
        # print(im)
        # print("imarrray- shoulde be size: 11647240")
        # print(imarray.shape)
        
        # Dead pixels
        # if nd != 0 and md != 0:
        #    imarray[nd, md] = 0
        imarray[np.where(imarray < 0)] = 0
        # print(imarray.size)
        
        A[:, :, i] = imarray[:, :]

    Z = np.log10(A + 1)

    whalf = np.log10(np.amax(A) + 1)

    # 3D-RSM using PyVista
    mesh = pv.StructuredGrid(dQx, dQy, dQz)
    mesh.point_data['values'] = Z.ravel(order='F')
    isos = mesh.contour(isosurfaces=np.array(sclfactor) * whalf)

    # You can save the isosurfaces and plot the RSM using other libraries (Paraview, vtkplotter, ParaView, Open3D, Vedo)
    # isos.save('XXXX.ply')

    p = pv.Plotter()
    p.add_mesh(isos, cmap=cmap, opacity=op, show_scalar_bar=True)

    # p.show_grid(title="Reciprocal Space Map", xtitle="r"$\Delta$Q$_x$ ($\AA^{-1}$)"", ytitle="r"$\Delta$Q$_y$ ($\AA^{-1}$)"", ztitle="r"$\Delta$Q$_z$ ($\AA^{-1}$)"", color='green', font_family="arial")
    p.show_grid(xtitle="Qx", ytitle="Qy", ztitle="Qz", color='green', font_family="arial")
    p.set_background('white')
    p.camera.azimuth = az
    p.camera.elevation = el

    p.show(screenshot=fnamepng + '_3D.png')

    # 2D projections
    II = np.sum(A, axis=1)
    XX = dQx[:, m0 - 1, :]
    ZZ = dQz[:, m0 - 1, :]

    fig, ax = plt.subplots(constrained_layout=True)

    c = ax.contourf(XX, ZZ, np.log10(II + 1), 12, cmap='viridis')

    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel(r'log$_10$ (I)')

    ax.contour(XX, ZZ, np.log10(II + 1), 12, colors=('k',), linewidths=(.3,))

    ax.set_xlabel(r"$\Delta$Q$_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$\Delta$Q$_z$ ($\AA^{-1}$)")

    plt.savefig(fnamepng + '_2D.png', dpi=200)
    plt.show()
