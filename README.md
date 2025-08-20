# PyISR: HDF5 → TIFF and Reciprocal Space Mapping (RSM)

Tools to convert beamline HDF5 data to TIFF and build/visualize reciprocal-space maps (RSM). Now with a Pixi-based workflow for reliable, reproducible environments.

## Quickstart (Pixi)

Pixi is a fast, cross-platform package/environment manager.

1) Install Pixi (Linux)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
# Then ensure ~/.pixi/bin is on PATH (add to your shell rc)
export PATH="$HOME/.pixi/bin:$PATH"
```

2) Clone this repo
```bash
git clone https://github.com/<your-org>/pyisr.git
cd pyisr
```

3) Create the environment with Pixi
- Option A: If pixi.toml is provided in this repo:
```bash
pixi install
```
- Option B: Initialize and add packages (first time setup)
```bash
pixi init --python 3.11
pixi add -c conda-forge numpy h5py tifffile matplotlib pyvista imageio \
         pandas dask vtk hklpy ophyd napari
```

4) Use the environment
```bash
# Open an interactive shell with the project environment
pixi shell

# Or run one-off commands
pixi run python -c "import numpy; print('OK')"
```

Notes:
- VTK/PyVista may require an OpenGL-capable system. On headless nodes, set:
  - EGL: osmesa/egl build if available, or use xvfb-run for offscreen rendering.

## Features
- HDF5 → TIFF extraction for downstream visualization
- RSM building and gridding
- 3D/2D visualization (PyVista, TIFF outputs)
- hklpy-based geometry (E4CV) for Q/HKL transforms

## Usage

### HDF5 → TIFF
```python
# Inside `pixi shell`
from hdf2tiff import hdf2tiff

input_directory = "/path/to/input/hdf5"
output_directory = "/path/to/output/tiff"

hdf2tiff(input_directory, output_directory)
```

### RSM Pipeline (example)
```python
# Inside `pixi shell`
from rsm3d.spec_parser import SpecParser
from rsm3d.rsm3d import RSMBuilder

spec_file = "/path/to/spec_file"
tiff_dir  = "/path/to/tiff_dir"

# Build with optional scan filtering
builder = RSMBuilder(spec_file, tiff_dir, selected_scans=(17, 18, 19))
Q_samp, hkl, intensity = builder.compute_full()

# Define a grid in Q-space and regrid
grid_ranges = (
    (Q_samp[...,0].min(), Q_samp[...,0].max()),
    (Q_samp[...,1].min(), Q_samp[...,1].max()),
    (Q_samp[...,2].min(), Q_samp[...,2].max()),
)
grid_shape = (200, 200, 200)
builder.setup_grid(grid_ranges, grid_shape)
rsm_q, edges_q = builder.regrid_intensity(method="mean", space="q")
```

### Export/Visualization
```python
from rsm3d.data_io import write_rsm_volume_to_vtk
write_rsm_volume_to_vtk(rsm_q, edges_q, "/path/to/output/rsm_q.vtk")
```

### Napari (optional)
```bash
pixi run python -c "import napari; print(napari.__version__)"
```

## Dependencies (managed by Pixi)
- Core: numpy, pandas, dask
- IO: h5py, tifffile, imageio
- Geometry: hklpy, ophyd
- Viz: matplotlib, pyvista, vtk, napari (optional)

If you prefer pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy h5py tifffile matplotlib pyvista imageio pandas dask vtk hklpy ophyd napari
```

## Tips
- Large datasets: prefer `pixi shell` + Python scripts over notebooks for memory control.
- RSMBuilder can use per-scan UB parsed from SPEC (#G3). Ensure your SPEC file is provided.

## License
MIT License. See LICENSE.

## Contact
Open an issue or PR with