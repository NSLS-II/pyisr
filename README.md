# PyISR: HDF5 → TIFF and Reciprocal Space Mapping (RSM)

Tools to convert beamline HDF5 data to TIFF and build/visualize reciprocal-space maps (RSM). Uses hklpy (E4CV) for Q/HKL transforms and supports per-scan UB from SPEC (#G3).

## Quickstart (Pixi)

1) Install Pixi (Linux)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

2) Clone the repo
```bash
git clone git@github.com:NSLS-II/pyisr.git
cd pyisr
```

3) Create/use the environment
```bash
pixi install
pixi shell
```

Note: VTK/PyVista may require an OpenGL-capable system. On headless Linux, use OSMesa/EGL builds or xvfb-run.

## Features
- HDF5 → TIFF extraction for downstream visualization
- RSM building and gridding
- hklpy-based geometry (E4CV) for Q/HKL transforms
- Per-scan UB from SPEC (#G3) applied in Q→HKL
- 3D/2D visualization outputs (VTK, TIFF)

## Usage

### HDF5 → TIFF
```python
# Inside `pixi shell`
from rsm3d.data_io import hdf2tiff

input_directory = "/path/to/input/hdf5"
output_directory = "/path/to/output/tiff"
hdf2tiff(input_directory, output_directory)
```

### RSM pipeline
```python
# Inside `pixi shell`
from rsm3d.spec_parser import SpecParser
from rsm3d.rsm3d import RSMBuilder

spec_file = "/path/to/spec_file"
tiff_dir  = "/path/to/tiff_dir"

# Per-scan UB is read from #G3 in the SPEC file
builder = RSMBuilder(spec_file, tiff_dir, selected_scans=(17, 18, 19))
Q_samp, hkl, intensity = builder.compute_full()

# Option A: auto bounds and regrid in Q or HKL
rsm_q,   edges_q   = builder.regrid_auto(space='q',   grid_shape=(200, 200, 200), method='mean')
rsm_hkl, edges_hkl = builder.regrid_auto(space='hkl', grid_shape=(200, 200, 200), method='mean')

# Option B: Q auto helper (equivalent to space='q')
rsm_q2, edges_q2 = builder.regrid_q_auto(grid_shape=(200, 200, 200), method="mean")
```

### Export/visualization
```python
from rsm3d.data_io import write_rsm_volume_to_vtk
write_rsm_volume_to_vtk(rsm_q, edges_q, "/path/to/output/rsm_q.vtk")
write_rsm_volume_to_vtk(rsm_hkl, edges_hkl, "/path/to/output/rsm_hkl.vtk")
```

## Dependencies (managed by Pixi)
- Core: numpy, pandas, dask
- IO: h5py, tifffile, imageio
- Geometry: hklpy, ophyd
- Viz: matplotlib, pyvista, vtk

If you prefer pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas dask h5py tifffile imageio matplotlib pyvista vtk hklpy ophyd
```

## Tips
- Large datasets: prefer `pixi shell` + Python scripts over notebooks.
- Ensure your SPEC file includes #G3 lines so per-scan UB is applied.

## License
MIT License. See LICENSE.

## Contributing
Issues and pull requests are welcome:
https://github.com/NSLS-