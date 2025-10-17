# read_data.py
import os
import re
import tifffile
import pandas as pd
import numpy as np
import h5py
import hdf5plugin
import vtk
from vtk.util import numpy_support

try:
    from dask import delayed
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from rsm3d.spec_parser import SpecParser

class RSMDataLoader:
    """
    Load and merge SPEC metadata with TIFF intensity frames.
    Provides (setup, UB, merged DataFrame).
    """
    def __init__(
        self,
        spec_file: str,
        setup_file: str,
        tiff_dir: str,
        *,
        use_dask: bool = False,
        process_hklscan_only: bool = False,
        selected_scans=None,
    ):
        self.spec_file = spec_file
        self.setup_file = setup_file
        self.tiff_dir = tiff_dir
        self.use_dask = use_dask
        self.process_hklscan_only = process_hklscan_only
        self.selected_scans = selected_scans

    def load(self):
        exp = SpecParser(self.spec_file, self.setup_file)
        setup = exp.setup
        UB = np.asarray(exp.crystal.UB, dtype=np.float64)

        # SPEC metadata
        df_meta = exp.to_pandas()
        df_meta["scan_number"] = df_meta["scan_number"].astype(int)
        df_meta["data_number"] = df_meta["data_number"].astype(int)

        # TIFF intensities
        rd = ReadFrame(self.tiff_dir, use_dask=self.use_dask)
        df_int = rd.load_data()

        # Merge
        df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")

        # # Filters
        # if self.process_hklscan_only:
        #     df = df[df["type"].str.lower().eq("hklscan", na=False)]
        # if self.selected_scans is not None:
        #     df = df[df["scan_number"].isin(set(self.selected_scans))]
        # Filters
        if self.process_hklscan_only:
            # fill NaNs then compare to "hklscan"
            mask = df["type"].str.lower().fillna("") == "hklscan"
            df = df[mask]
        if self.selected_scans is not None:
             df = df[df["scan_number"].isin(set(self.selected_scans))]
        if df.empty:
            raise ValueError("No frames to process after filtering/merge.")
        return setup, UB, df.reset_index(drop=True)


class ReadFrame:
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


def write_rsm_volume_to_vtr(rsm, coords, filename, binary=True, compress=True):
    """
    Write a 3D RSM volume to VTK XML RectilinearGrid (.vtr).

    Parameters
    ----------
    rsm : (nx, ny, nz) ndarray
        Cell-centered intensities (one value per bin).
    coords : [x_coords, y_coords, z_coords]
        For each axis, you may pass either:
          * bin EDGES of length n+1, or
          * bin CENTERS of length n (edges will be inferred).
        Arrays may be ascending or descending; descending inputs are handled.
    filename : str
        Output path; '.vtr' will be enforced if missing.
    binary : bool
        Appended (binary) vs ASCII XML data.
    compress : bool
        Enable zlib compression when binary=True (if available in your VTK).
    """
    x_c, y_c, z_c = [np.asarray(a, dtype=np.float64) for a in coords]
    nx, ny, nz = map(int, rsm.shape)

    def _as_edges(arr, n):
        """Return edges of length n+1, from either edges (n+1) or centers (n)."""
        m = arr.size
        if m == n + 1:
            return arr.copy()
        if m == n:
            # infer edges from centers: interior = midpoints, ends extrapolated
            edges = np.empty(n + 1, dtype=np.float64)
            edges[1:-1] = 0.5 * (arr[1:] + arr[:-1])
            # use local spacing at each end
            edges[0]  = arr[0]  - 0.5 * (arr[1]  - arr[0])
            edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
            return edges
        raise ValueError(f"Coordinate array must have length {n} (centers) or {n+1} (edges); got {m}.")

    x_edges = _as_edges(x_c, nx)
    y_edges = _as_edges(y_c, ny)
    z_edges = _as_edges(z_c, nz)

    # Ensure each axis is ascending; if not, flip both coords and data
    rsm_work = np.asarray(rsm, dtype=np.float32)

    if x_edges[1] < x_edges[0]:
        x_edges = x_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=0)
    if y_edges[1] < y_edges[0]:
        y_edges = y_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=1)
    if z_edges[1] < z_edges[0]:
        z_edges = z_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=2)

    # Basic sanity: positive widths
    if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0) or np.any(np.diff(z_edges) <= 0):
        raise ValueError("Non-positive bin width detected after adjustment.")

    # Build rectilinear grid
    grid = vtk.vtkRectilinearGrid()
    # Points = bins+1 along each axis
    grid.SetDimensions(nx + 1, ny + 1, nz + 1)
    # Also set explicit extent in point-index space (optional but robust)
    grid.SetExtent(0, nx, 0, ny, 0, nz)

    # Coordinate arrays (vtkDoubleArray)
    def _vtk_coords(arr):
        v = numpy_support.numpy_to_vtk(arr, deep=True)
        # vtkRectilinearGrid ignores name here; fine to leave unset or set a label
        return v

    grid.SetXCoordinates(_vtk_coords(x_edges))
    grid.SetYCoordinates(_vtk_coords(y_edges))
    grid.SetZCoordinates(_vtk_coords(z_edges))

    # Cell data: sanitize + Fortran order so I (x) is fastest (VTK IJK)
    np.nan_to_num(rsm_work, copy=False)
    intens = rsm_work.ravel(order="F")
    vtk_int = numpy_support.numpy_to_vtk(intens, deep=True)
    vtk_int.SetName("intensity")
    grid.GetCellData().SetScalars(vtk_int)
    grid.GetCellData().SetActiveScalars("intensity")

    # Writer
    if not filename.lower().endswith(".vtr"):
        base = filename.rsplit(".", 1)[0] if "." in filename else filename
        filename = base + ".vtr"

    w = vtk.vtkXMLRectilinearGridWriter()
    try:
        w.SetInputData(grid)
    except AttributeError:
        w.SetInput(grid)
    w.SetFileName(filename)

    if binary:
        try:
            w.SetDataModeToAppended()
        except AttributeError:
            try:
                w.SetDataModeToBinary()
            except AttributeError:
                pass
        if compress:
            try:
                w.SetCompressorTypeToZLib()
            except AttributeError:
                try:
                    w.SetCompressor(vtk.vtkZLibDataCompressor())
                except Exception:
                    pass
    else:
        try:
            w.SetDataModeToAscii()
        except AttributeError:
            pass

    if w.Write() != 1:
        raise RuntimeError(f"Failed to write VTR file: {filename}")


def read_hdf5_tiff_data(directory):
    """
    Reads TIFF-like data stored at '/entry/data/data' from all HDF5 files in the specified directory,
    but only processes files that contain 'data' in the filename.
    
    Parameters:
        directory (str): The path to the directory containing HDF5 files.
    
    Returns:
        A dictionary containing the TIFF data from each file, with filenames as keys.
    """
    tiff_data_dict = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Process only files that contain 'data' in their filename
        if "data" in filename and filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            try:
                # Open the HDF5 file
                with h5py.File(file_path, 'r') as hdf_file:
                    # Access the data at the specified path
                    if '/entry/data' in hdf_file:
                        tiff_data = np.squeeze(np.array(hdf_file['/entry/data/data']))
                        tiff_data_dict[filename] = tiff_data
                        print(f"Successfully read TIFF data from: {filename}")
                    else:
                        print(f"/entry/data not found in {filename}")
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return tiff_data_dict

def save_tiff_data(tiff_data, output_dir, original_filename, normalize=True, overwrite=False):
    """
    Saves the given TIFF data as an image file in the specified output directory.
    
    The function converts the data to 32-bit unsigned integers while preserving the original data range.
    This means that no scaling is applied.
    
    Parameters:
        tiff_data (numpy array): The TIFF data array.
        output_dir (str): The directory to save the TIFF file.
        original_filename (str): The original HDF5 filename used for naming the output TIFF file.
        normalize (bool): This flag is ignored; original data range is preserved.
        overwrite (bool): If True, existing files are replaced.
    """
    # Create output filename with .tiff extension
    output_filename = os.path.splitext(original_filename)[0] + ".tiff"
    output_path = os.path.join(output_dir, output_filename)
    
    if not overwrite and os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping save.")
        return
    
    # Ensure the data is numeric
    if tiff_data.dtype.kind in {'U', 'S'}:
        print(f"Data is not numerical: {tiff_data.dtype}. Skipping conversion for {original_filename}.")
        return

    # Preserve original data range by converting directly to uint32.
    if np.issubdtype(tiff_data.dtype, np.integer):
        out_data = tiff_data.astype(np.uint32)
    elif np.issubdtype(tiff_data.dtype, np.floating):
        # For floats, round before converting to uint32
        out_data = np.rint(tiff_data).astype(np.uint32)
    else:
        out_data = tiff_data

    try:
        tifffile.imwrite(output_path, out_data)
        print(f"Saved TIFF to {output_path}")
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")

def remove_extreme(
    image: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Replace every pixel > threshold by the average of its 8-connected neighbors,
    excluding any neighbors that are themselves > threshold.
    Pure NumPy, fully vectorized.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    threshold : float
        Pixels strictly > threshold will be replaced.

    Returns
    -------
    np.ndarray
        New image of same shape and dtype, with extreme pixels replaced.
    """
    if image.ndim != 2:
        raise ValueError("Only 2D arrays supported")

    # Work in float
    arr = image.astype(float)
    mask = arr > threshold  # pixels to replace

    # Reflect‐pad for edge handling
    p   = np.pad(arr, 1, mode='reflect')
    pm  = np.pad(mask, 1, mode='reflect')

    # Extract the 8 neighbors and their masks
    p00, m00 = p[0:-2, 0:-2], pm[0:-2, 0:-2]
    p01, m01 = p[0:-2, 1:-1], pm[0:-2, 1:-1]
    p02, m02 = p[0:-2, 2:  ], pm[0:-2, 2:  ]
    p10, m10 = p[1:-1, 0:-2], pm[1:-1, 0:-2]
    p12, m12 = p[1:-1, 2:  ], pm[1:-1, 2:  ]
    p20, m20 = p[2:  , 0:-2], pm[2:  , 0:-2]
    p21, m21 = p[2:  , 1:-1], pm[2:  , 1:-1]
    p22, m22 = p[2:  , 2:  ], pm[2:  , 2:  ]

    # Sum only non-extreme neighbors
    valid00 = (~m00).astype(float); valid01 = (~m01).astype(float)
    valid02 = (~m02).astype(float); valid10 = (~m10).astype(float)
    valid12 = (~m12).astype(float); valid20 = (~m20).astype(float)
    valid21 = (~m21).astype(float); valid22 = (~m22).astype(float)

    neighbor_sum = (
        p00*valid00 + p01*valid01 + p02*valid02 +
        p10*valid10 +          p12*valid12 +
        p20*valid20 + p21*valid21 + p22*valid22
    )
    neighbor_count = (
        valid00 + valid01 + valid02 +
        valid10 +           valid12 +
        valid20 + valid21 + valid22
    )

    # Compute mean, avoid division by zero
    nbr_mean = np.zeros_like(arr)
    nonzero = neighbor_count > 0
    nbr_mean[nonzero] = neighbor_sum[nonzero] / neighbor_count[nonzero]
    # For isolated extremes with no valid neighbors, clamp to threshold
    nbr_mean[~nonzero] = threshold

    # Build result
    result = arr.copy()
    result[mask] = nbr_mean[mask]

    # Cast back to original dtype if integer
    if np.issubdtype(image.dtype, np.integer):
        result = np.rint(result).astype(image.dtype)

    return result
       
def hdf2tiff(input_directory: str, output_directory: str, overwrite: bool = False, extreme_threshold: float = None):
    """
    Main function to read HDF5 files, extract TIFF data, optionally remove extreme pixel values,
    and save them as TIFFs.

    Parameters:
        input_directory (str): Path to the input directory containing HDF5 files.
        output_directory (str): Path to the output directory for saving TIFF files.
        overwrite (bool): If True, existing TIFF files will be replaced.
        extreme_threshold (float, optional): If provided, every pixel value greater than this threshold
                                               will be replaced by the average of its valid 8-connected neighbors.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read HDF5 files and extract TIFF data
    tiff_data_dict = read_hdf5_tiff_data(input_directory)

    # Save TIFF data to the output directory
    for file_name, data in tiff_data_dict.items():
        if extreme_threshold is not None:
            data = remove_extreme(data, extreme_threshold)
        save_tiff_data(data, output_directory, file_name, overwrite=overwrite)