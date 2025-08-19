# read_data.py
import os
import re
import tifffile
import pandas as pd
import numpy as np

import vtk
from vtk.util import numpy_support

try:
    from dask import delayed
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

class ReadData:
    """
    Class to scan a directory for TIFF files matching a regex pattern,
    extract scan_number and data_number, keep 2D intensity arrays per frame,
    and return as a pandas or Dask DataFrame.

    Parameters:
        directory (str): Path to the directory containing TIFF files.
        pattern (str, optional): Regex to match filenames and capture two groups:
            scan_number and data_number. Defaults to r"^[^_]+_[^_]+_(\d{3})_(\d{3})_.*\\.tiff$".
        use_dask (bool): Whether to use Dask for lazy loading (requires dask).
    """
    def __init__(self, directory, pattern=None, use_dask=False):
        self.directory = directory
        self.use_dask = use_dask and DASK_AVAILABLE
        if use_dask and not DASK_AVAILABLE:
            raise ImportError("Dask libraries not found. Install dask to use Dask functionality.")

        default_pattern = r"^[^_]+_[^_]+_(\d{3})_(\d{3})_.*\.tiff$"
        pattern_str = pattern or default_pattern
        self._pattern = re.compile(pattern_str)

    def _process_file(self, fname):
        """Read one TIFF, extract scan/data numbers, and keep full 2D intensity."""
        match = self._pattern.match(fname)
        if not match:
            return None

        scan_number = int(match.group(1))
        data_number = int(match.group(2))
        path = os.path.join(self.directory, fname)

        # Load full image as 2D (or higher-dim if multi-page) array
        img = tifffile.imread(path)

        # Return one-row DataFrame with array in 'intensity' column
        return pd.DataFrame([{  
            'scan_number': scan_number,
            'data_number': data_number,
            'intensity': img
        }])

    def load_data(self):
        """
        Load data from all matching files.
        Returns pd.DataFrame or dd.DataFrame with each row per file,
        intensity column holding the full array.
        """
        files = [f for f in os.listdir(self.directory) if self._pattern.match(f)]
        if self.use_dask:
            delayed_dfs = [delayed(self._process_file)(f) for f in files]
            # Provide metadata for Dask
            meta = {
                'scan_number': 'i8',
                'data_number': 'i8',
                'intensity': object
            }
            return dd.from_delayed(delayed_dfs, meta=meta)
        else:
            dfs = [self._process_file(f) for f in files]
            dfs = [df for df in dfs if df is not None]
            return pd.concat(dfs, ignore_index=True)
        
        
        
        
        
        
def write_rsm_vtk(polydata, scalar_name, filename):
    """
    Write a vtk XML PolyData (.vtp) file from a vtkPolyData object.

    Parameters:
      polydata   : vtkPolyData with points and arrays set
      scalar_name: name of the scalar array to set for coloring
      filename   : output .vtp filename
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def export_rsm_vtps(Q_samp, hkl, intensity, prefix):
    """
    Export two .vtp files: one at Q-space coordinates, one at hkl indices.

    Files:
      {prefix}_q.vtp   : Q-space point cloud
      {prefix}_hkl.vtp : hkl-space point cloud
    """
    # flatten
    points_q   = Q_samp.reshape(-1,3)
    points_hkl = hkl.reshape(-1,3)
    intens     = intensity.ravel()

    # common vtkPolyData setup for both
    def make_poly(points):
        poly = vtk.vtkPolyData()
        pts  = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(points, deep=True))
        poly.SetPoints(pts)
        return poly

    # build and write Q-space
    poly_q = make_poly(points_q)
    arr_I = numpy_support.numpy_to_vtk(intens, deep=True)
    arr_I.SetName('intensity')
    poly_q.GetPointData().SetScalars(arr_I)
    write_rsm_vtk(poly_q, 'intensity', f"{prefix}_q.vtp")

    # build and write hkl-space
    poly_h = make_poly(points_hkl)
    poly_h.GetPointData().SetScalars(arr_I)
    write_rsm_vtk(poly_h, 'intensity', f"{prefix}_hkl.vtp")
    
    


def write_polydata_legacy(polydata, filename, binary=False):
    """
    Write a vtk PolyData to a legacy .vtk file.
    
    Parameters:
        polydata : vtkPolyData
        filename : str, output path ending in .vtk
        binary   : bool, if True writes binary, otherwise ASCII
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.Write()


def write_rsm_volume_to_vtk(rsm, edges, filename, binary=False):
    """
    Write a 3D RSM volume to a legacy VTK RectilinearGrid (.vtk).

    Parameters:
        rsm      : ndarray of shape (nx, ny, nz), the binned intensities
        edges    : list of three 1D arrays [x_edges, y_edges, z_edges], each
                   of length (n+1) for the bin boundaries along that axis
        filename : str, output filename ending in .vtk
        binary   : bool, whether to write in binary (True) or ASCII (False)
    """
    # Unpack edges
    x_edges, y_edges, z_edges = edges
    nx, ny, nz = rsm.shape

    # Create the rectilinear grid, dimensions = number of points = bins+1
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(nx+1, ny+1, nz+1)

    # Helper to make VTK coord arrays
    def _make_coord_array(arr):
        vtk_arr = numpy_support.numpy_to_vtk(arr.astype(np.float32), deep=True)
        vtk_arr.SetName('coord')
        return vtk_arr

    # Assign the coordinates (these are the *point* locations)
    grid.SetXCoordinates(_make_coord_array(x_edges))
    grid.SetYCoordinates(_make_coord_array(y_edges))
    grid.SetZCoordinates(_make_coord_array(z_edges))

    # Now attach the intensity as *cell* data (one cell per bin)
    cell_data = grid.GetCellData()
    vtk_int = numpy_support.numpy_to_vtk(rsm.ravel(order='C'), deep=True)
    vtk_int.SetName('intensity')
    cell_data.SetScalars(vtk_int)

    # Choose the legacy writer
    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.Write()


