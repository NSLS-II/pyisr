# read_data.py
import os
import re
import numbers
from typing import Optional
import tifffile
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import yaml
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
        setup = ExperimentSetup.from_yaml(self.setup_file)
        exp = SpecParser(self.spec_file, self.setup_file)
        df_meta = exp.to_pandas()
        df_meta["scan_number"] = df_meta["scan_number"].astype(int)
        df_meta["data_number"] = df_meta["data_number"].astype(int)

        selected_list: list[int] = []
        wanted: Optional[set[int]] = None
        if self.selected_scans is not None:
            if isinstance(self.selected_scans, numbers.Integral):
                selected_list = [int(self.selected_scans)]
            else:
                try:
                    selected_list = [int(s) for s in self.selected_scans]
                except TypeError:
                    selected_list = [int(self.selected_scans)]
            wanted = set(selected_list)
            df_meta = df_meta[df_meta["scan_number"].isin(wanted)]
            if df_meta.empty:
                raise ValueError("No metadata rows match selected_scans.")

        # 2. Load TIFF frames
        rd = ReadFrame(self.tiff_dir, use_dask=self.use_dask)
        df_int = rd.load_data()

        # 3. If selected_scans provided, prune TIFF frames before merge
        if wanted is not None:
            df_int = df_int[df_int["scan_number"].isin(wanted)]
            if len(df_int) == 0:
                raise ValueError("No TIFF frames match selected_scans.")

        # 4. Merge only the needed scans
        df = pd.merge(df_meta, df_int, on=["scan_number", "data_number"], how="inner")
        if df.empty:
            raise ValueError("No frames after merging metadata and TIFF data.")

        # 5. Now apply hklscan filtering (order changed per requirement)
        if self.process_hklscan_only:
            df = df[df["type"].str.lower().fillna("") == "hklscan"]
            if df.empty:
                raise ValueError("No frames remain after applying hklscan filter.")

        df = df.reset_index(drop=True)

        fallback_ub = np.asarray(exp.crystal.UB, dtype=np.float64)

        # Resolve a default UB matrix, preferring scan-specific entries when a selection is provided.
        def _resolve_default_ub(df_frames: pd.DataFrame, scans_to_check: list[int]) -> np.ndarray:
            if "ub" not in df_frames.columns:
                if scans_to_check:
                    raise ValueError("SPEC metadata is missing UB matrices for the selected scans.")
                return fallback_ub

            def _scan_ub(scan_id: int) -> Optional[np.ndarray]:
                rows = df_frames[df_frames["scan_number"] == scan_id]
                if rows.empty:
                    return None
                candidates = []
                for val in rows["ub"]:
                    if val is None:
                        continue
                    arr = np.asarray(val, dtype=np.float64)
                    if arr.shape != (3, 3):
                        continue
                    candidates.append(arr)
                if not candidates:
                    return None
                base = candidates[0]
                for other in candidates[1:]:
                    if not np.allclose(base, other):
                        raise ValueError(
                            f"Inconsistent UB matrices encountered within scan {scan_id}."
                        )
                return base

            if scans_to_check:
                for scan_id in scans_to_check:
                    scan_ub = _scan_ub(scan_id)
                    if scan_ub is not None:
                        return scan_ub.copy()
                raise ValueError("Failed to locate a UB matrix for the requested scan(s).")

            for scan_id in df_frames["scan_number"].unique():
                scan_ub = _scan_ub(int(scan_id))
                if scan_ub is not None:
                    return scan_ub.copy()
            return fallback_ub

        UB = _resolve_default_ub(df, selected_list)

        return setup, UB, df
    

class ExperimentSetup:
    """
    Load experiment parameters from a YAML file. Wavelength is optional:
      • if energy is supplied (>0 keV), wavelength is computed from energy
      • if energy is omitted/null, a valid wavelength must be provided
      • sub-micrometer wavelengths (<1e-3) are interpreted as meters and converted to Å
    Required keys (either top-level or inside `ExperimentSetup:`):
      distance, pitch, ycenter, xcenter, xpixels, ypixels
    One of energy or wavelength must be present.
    """
    REQUIRED_KEYS = (
        "distance", "pitch", "ycenter", "xcenter",
        "xpixels", "ypixels", "energy",
    )

    def __init__(
        self,
        distance: float,
        pitch: float,
        ycenter: int,
        xcenter: int,
        xpixels: int,
        ypixels: int,
        energy: float | None = None,
        wavelength: float | None = None,
    ):
        self.distance = float(distance)
        self.pitch = float(pitch)
        self.ycenter = int(ycenter)
        self.xcenter = int(xcenter)
        self.xpixels = int(xpixels)
        self.ypixels = int(ypixels)

        if self.distance <= 0:
            raise ValueError("ExperimentSetup: 'distance' must be > 0")
        if self.pitch <= 0:
            raise ValueError("ExperimentSetup: 'pitch' must be > 0")
        if self.xpixels <= 0 or self.ypixels <= 0:
            raise ValueError("ExperimentSetup: 'xpixels' and 'ypixels' must be > 0")

        lam_input: float | None = None
        if wavelength is not None:
            if isinstance(wavelength, str):
                cleaned = wavelength.strip().lower()
                if cleaned not in {"", "none", "null"}:
                    try:
                        lam_input = float(wavelength)
                    except ValueError as exc:
                        raise ValueError("ExperimentSetup: wavelength must be numeric") from exc
            else:
                lam_input = float(wavelength)

        if lam_input is not None and 0.0 < lam_input < 1e-3:
            lam_input *= 1e10

        self.energy = None
        self.energy_keV = None

        if energy is not None:
            self.energy = float(energy)
            self.energy_keV = self.energy
            if self.energy_keV <= 0:
                raise ValueError("ExperimentSetup: 'energy' (keV) must be > 0")
            self.wavelength = self._energy_keV_to_lambda_A(self.energy_keV)
        else:
            if lam_input is None or lam_input <= 0.0:
                raise ValueError("ExperimentSetup: wavelength must be provided and positive when energy is missing")
            self.wavelength = lam_input
            self.energy_keV = self._lambda_A_to_energy_keV(self.wavelength)
            if self.energy_keV <= 0.0:
                raise ValueError("ExperimentSetup: derived energy from wavelength is non-positive")
            self.energy = self.energy_keV

    @staticmethod
    def _energy_keV_to_lambda_A(E_keV: float) -> float:
        return 12.398419843320026 / float(E_keV)

    @staticmethod
    def _lambda_A_to_energy_keV(lambda_A: float) -> float:
        return 12.398419843320026 / float(lambda_A)

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(str(value).replace("_", "").strip())
            except Exception as exc:
                raise ValueError(f"Expected float-compatible value, got {value!r}") from exc

    @staticmethod
    def _to_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(str(value).replace("_", "").strip()))
            except Exception as exc:
                raise ValueError(f"Expected int-compatible value, got {value!r}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Experiment YAML not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sec = cls._extract_section(doc)
        merged = {}
        for k in cls.REQUIRED_KEYS + ("wavelength",):
            if k in sec:
                merged[k] = sec[k]
            elif k in doc:
                merged[k] = doc[k]

        missing = [
            k for k in cls.REQUIRED_KEYS
            if k != "energy" and merged.get(k) in (None, "", "None", "null")
        ]
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")

        energy_raw = merged.get("energy")
        if isinstance(energy_raw, str) and energy_raw.strip().lower() in {"", "none", "null"}:
            energy_raw = None

        wavelength_raw = merged.get("wavelength")
        if isinstance(wavelength_raw, str) and wavelength_raw.strip().lower() in {"", "none", "null"}:
            wavelength_raw = None

        if energy_raw is None and wavelength_raw is None:
            raise ValueError("Experiment YAML must provide either energy or wavelength.")

        params = {
            "distance":  cls._to_float(merged["distance"]),
            "pitch":     cls._to_float(merged["pitch"]),
            "ycenter":   cls._to_int(merged["ycenter"]),
            "xcenter":   cls._to_int(merged["xcenter"]),
            "xpixels":   cls._to_int(merged["xpixels"]),
            "ypixels":   cls._to_int(merged["ypixels"]),
            "energy":    cls._to_float(energy_raw) if energy_raw is not None else None,
            "wavelength": cls._to_float(wavelength_raw) if wavelength_raw is not None else None,
        }
        return cls(**params)

    def __repr__(self):
        energy_display = self.energy_keV if self.energy_keV is not None else "N/A"
        return (
            f"<ExperimentSetup: distance={self.distance} m, pitch={self.pitch} m, "
            f"xcenter={self.xcenter}, ycenter={self.ycenter}, "
            f"xpixels={self.xpixels}, ypixels={self.ypixels}, "
            f"energy={energy_display} keV, wavelength={self.wavelength} Å>"
        )

    @staticmethod
    def _to_int(v):
        """Convert a value from YAML to an integer, handling common pitfalls."""
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in {"", "none", "null"}:
                return None
            try:
                return int(float(v))
            except ValueError:
                raise ValueError(f"Cannot convert to int: {v}")
        return int(v)

    @classmethod
    def _extract_section(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping of keys to values.")
        for key in ("ExperimentSetup", "experiment", "experiment_setup"):
            section = data.get(key)
            if isinstance(section, dict):
                return section
        if any(k in data for k in cls.REQUIRED_KEYS):
            return data
        for value in data.values():
            if isinstance(value, dict) and any(k in value for k in cls.REQUIRED_KEYS):
                return value
        raise ValueError(
            "Could not locate experiment setup configuration in YAML; "
            "expected an 'ExperimentSetup' section or flat keys."
        )


class ReadFrame:
    """
    Scan a directory for TIFF files, capture scan/data numbers, store 2D arrays.
    """
    def __init__(self, directory, pattern=None, use_dask=False):
        r"""
        Parameters
        ----------
        directory : str
            Root directory containing TIFF frames.
        pattern : str | None, optional
            Regex with two capturing groups for scan_number and data_number.
            Defaults to r"^(?:[^_]+_)*(\d+)_(\d+)_.*\.tiff$".
        use_dask : bool, optional
            Enable Dask-backed loading when the dependency is available.
        """
        self.directory = directory
        self.use_dask = use_dask and DASK_AVAILABLE
        if use_dask and not DASK_AVAILABLE:
            raise ImportError("Dask libraries not found. Install dask to use Dask functionality.")
        default_pattern = r"^(?:[^_]+_)*(\d+)_(\d+)_.*\.tiff$"
        pattern_str = pattern or default_pattern
        self._pattern = re.compile(pattern_str)

    def _process_file(self, fname):
        match = self._pattern.match(fname)
        if not match:
            return None

        numeric_groups = [
            g for g in (match.groups() or ())
            if g and re.fullmatch(r"-?\d+", g)
        ]
        if not numeric_groups:
            return None

        path = os.path.join(self.directory, fname)
        img = tifffile.imread(path)

        if len(numeric_groups) == 1:
            scan_number = int(numeric_groups[0])
            return pd.DataFrame([{
                "scan_number": scan_number,
                "intensity": img,
            }])

        scan_number = int(numeric_groups[0])
        data_number = int(numeric_groups[1])
        return pd.DataFrame([{
            "scan_number": scan_number,
            "data_number": data_number,
            "intensity": img,
        }])

    def load_data(self):
        files = [f for f in os.listdir(self.directory) if self._pattern.match(f)]
        if not files:
            return pd.DataFrame(columns=["scan_number", "data_number", "intensity"])

        valid_files = []
        group_counts = set()
        for fname in files:
            sample_match = self._pattern.match(fname)
            numeric_groups = [
                g for g in (sample_match.groups() or ())
                if g and re.fullmatch(r"-?\d+", g)
            ] if sample_match else []
            if not numeric_groups:
                raise ValueError(
                    f"Filename '{fname}' does not produce integer capture groups with the provided pattern."
                )
            group_counts.add(1 if len(numeric_groups) == 1 else 2)
            valid_files.append(fname)

        if not valid_files:
            return pd.DataFrame(columns=["scan_number", "data_number", "intensity"])
        if len(group_counts) != 1:
            raise ValueError("Filename pattern yields inconsistent integer captures; ensure a uniform regex.")
        single_group = group_counts.pop() == 1

        if self.use_dask:
            delayed_dfs = [delayed(self._process_file)(f) for f in valid_files]
            if single_group:
                meta = pd.DataFrame({
                    "scan_number": pd.Series(dtype="int64"),
                    "intensity": pd.Series(dtype="object"),
                })
            else:
                meta = pd.DataFrame({
                    "scan_number": pd.Series(dtype="int64"),
                    "data_number": pd.Series(dtype="int64"),
                    "intensity": pd.Series(dtype="object"),
                })
            return dd.from_delayed(delayed_dfs, meta=meta)

        dfs = [self._process_file(f) for f in valid_files]
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            columns = ["scan_number", "intensity"] if single_group else ["scan_number", "data_number", "intensity"]
            return pd.DataFrame(columns=columns)

        result = pd.concat(dfs, ignore_index=True)
        if single_group:
            return result[["scan_number", "intensity"]]
        return result


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
    points_q = Q_samp.reshape(-1, 3)
    points_hkl = hkl.reshape(-1, 3)
    intens = intensity.ravel()

    # common vtkPolyData setup for both
    def make_poly(points):
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
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
    writer.SetFileTypeToBinary() if binary else writer.SetFileTypeToASCII()
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
            edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
            edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
            return edges
        raise ValueError(f"Coordinate array must have length {n} or {n+1}; got {m}.")

    x_edges = _as_edges(x_c, nx); y_edges = _as_edges(y_c, ny); z_edges = _as_edges(z_c, nz)

    # Ensure each axis is ascending; if not, flip both coords and data
    rsm_work = np.asarray(rsm, dtype=np.float32)
    if x_edges[1] < x_edges[0]:
        x_edges = x_edges[::-1].copy(); rsm_work = np.flip(rsm_work, axis=0)
    if y_edges[1] < y_edges[0]:
        y_edges = y_edges[::-1].copy(); rsm_work = np.flip(rsm_work, axis=1)
    if z_edges[1] < z_edges[0]:
        z_edges = z_edges[::-1].copy(); rsm_work = np.flip(rsm_work, axis=2)

    # Basic sanity: positive widths
    if (np.diff(x_edges) <= 0).any() or (np.diff(y_edges) <= 0).any() or (np.diff(z_edges) <= 0).any():
        raise ValueError("Non-positive bin width detected after adjustment.")

    # Build rectilinear grid
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(nx + 1, ny + 1, nz + 1)
    grid.SetExtent(0, nx, 0, ny, 0, nz)

    # Coordinate arrays (vtkDoubleArray)
    def _vtk_coords(arr):
        return numpy_support.numpy_to_vtk(arr, deep=True)
    grid.SetXCoordinates(_vtk_coords(x_edges))
    grid.SetYCoordinates(_vtk_coords(y_edges))
    grid.SetZCoordinates(_vtk_coords(z_edges))

    # Cell data: sanitize + Fortran order so I (x) is fastest (VTK IJK)
    np.nan_to_num(rsm_work, copy=False)
    intens = rsm_work.ravel(order="F")
    vtk_int = numpy_support.numpy_to_vtk(intens, deep=True); vtk_int.SetName("intensity")
    grid.GetCellData().SetScalars(vtk_int); grid.GetCellData().SetActiveScalars("intensity")

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
            try: w.SetDataModeToBinary()
            except AttributeError: pass
        if compress:
            try: w.SetCompressorTypeToZLib()
            except AttributeError:
                try: w.SetCompressor(vtk.vtkZLibDataCompressor())
                except Exception: pass
    else:
        try: w.SetDataModeToAscii()
        except AttributeError: pass

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


def remove_extreme(image: np.ndarray, threshold: float) -> np.ndarray:
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
    p = np.pad(arr, 1, mode='reflect')
    pm = np.pad(mask, 1, mode='reflect')

    # Extract the 8 neighbors and their masks
    p00, m00 = p[0:-2, 0:-2], pm[0:-2, 0:-2]
    p01, m01 = p[0:-2, 1:-1], pm[0:-2, 1:-1]
    p02, m02 = p[0:-2, 2:],   pm[0:-2, 2:]
    p10, m10 = p[1:-1, 0:-2], pm[1:-1, 0:-2]
    p12, m12 = p[1:-1, 2:],   pm[1:-1, 2:]
    p20, m20 = p[2:, 0:-2],   pm[2:, 0:-2]
    p21, m21 = p[2:, 1:-1],   pm[2:, 1:-1]
    p22, m22 = p[2:, 2:],     pm[2:, 2:]

    # Sum only non-extreme neighbors
    valid = [~m for m in (m00, m01, m02, m10, m12, m20, m21, m22)]
    vals =  [p00, p01, p02, p10, p12, p20, p21, p22]
    neighbor_sum = sum(v.astype(float) * val.astype(float) for v, val in zip(vals, valid))
    neighbor_count = sum(val.astype(float) for val in valid)

    # Compute mean, avoid division by zero
    nbr_mean = np.zeros_like(arr)
    nz = neighbor_count > 0
    nbr_mean[nz] = neighbor_sum[nz] / neighbor_count[nz]
    # For isolated extremes with no valid neighbors, clamp to threshold
    nbr_mean[~nz] = threshold

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


class RSMDataloader_CMS:
    """
    Minimal loader for CMS workflows: attach ExperimentSetup metadata to TIFF frames
    without UB handling and provide zeroed angle columns.
    """
    def __init__(
        self,
        setup_file: str,
        tiff_dir: str,
        *,
        file_pattern: str = r".*_(\d+)_.*\.tiff$",
        use_dask: bool = False,
        selected_scans=None,
        crop_window: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ):
        self.setup_file = setup_file
        self.tiff_dir = tiff_dir
        self.file_pattern = file_pattern
        self.use_dask = use_dask
        self.selected_scans = selected_scans
        self.crop_window = crop_window

    @staticmethod
    def _crop_image(image, crop_window):
        arr = np.asarray(image)
        if arr.ndim < 2:
            raise ValueError("RSMDataloader_CMS: intensity frames must be at least 2D.")
        (r0, r1), (c0, c1) = crop_window
        if not (0 <= r0 < r1 <= arr.shape[0] and 0 <= c0 < c1 <= arr.shape[1]):
            raise ValueError("RSMDataloader_CMS: crop_window is out of bounds for the intensity frame.")
        return arr[r0:r1, c0:c1]

    def load(self):
        setup = ExperimentSetup.from_yaml(self.setup_file)
        frames = ReadFrame(self.tiff_dir,
                           pattern=self.file_pattern,
                           use_dask=self.use_dask).load_data()

        if DASK_AVAILABLE and isinstance(frames, dd.DataFrame):
            frames = frames.compute()

        if not isinstance(frames, pd.DataFrame) or frames.empty:
            raise ValueError("RSMDataloader_CMS: no TIFF frames found.")

        if self.selected_scans is not None:
            try:
                if isinstance(self.selected_scans, numbers.Integral):
                    wanted = {int(self.selected_scans)}
                else:
                    wanted = {int(s) for s in self.selected_scans}
            except TypeError:
                wanted = {int(self.selected_scans)}
            frames = frames[frames["scan_number"].isin(wanted)]
            if frames.empty:
                raise ValueError("RSMDataloader_CMS: no TIFF frames match selected_scans.")

        intensities = frames["intensity"].tolist()
        if self.crop_window is not None:
            intensities = [self._crop_image(img, self.crop_window) for img in intensities]

        df = pd.DataFrame(
            {
                "scan_number": frames["scan_number"].astype(int),
                "intensity": intensities,
                "tth": 0.0,
                "th": 0.0,
                "chi": 0.0,
                "phi": 0.0,
            }
        )
        return setup, df.reset_index(drop=True)