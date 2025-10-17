# PyISR: HDF5 → TIFF and Reciprocal Space Mapping (RSM)

Tools to convert beamline HDF5 data to TIFF and build/visualize reciprocal-space maps (RSM). Uses hklpy (E4CV) for Q/HKL transforms and supports per-scan UB from SPEC (#G3).

## Quickstart (pixi)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

git clone git@github.com:NSLS2/pyisr.git
cd pyisr
pixi install          # resolve environment
pixi shell            # enter environment
```

Test:

```bash
pixi run python -c "import rsm3d; print('ok')"
```

## Features (core)

- HDF5 → TIFF frame extraction
- Per-scan UB (#G3 in SPEC) for HKL transforms via hklpy
- Automatic HKL (or Q) range detection in re-gridding
- 3D reciprocal space volume generation
- Napari interactive viewer (volume + labels + point cloud)

## HDF5 → TIFF

```python
from rsm3d.data_io import hdf2tiff
hdf2tiff("/path/to/hdf5_scans", "/path/to/output_tiff")
```

## RSM Pipeline

```python
from rsm3d.rsm3d import RSMBuilder

builder = RSMBuilder(
    spec_file="/path/exp.spec",
    tiff_dir="/path/tiff_frames",
    selected_scans=(scan_number,),
    ub_includes_2pi=True,
    center_is_one_based=False,
)

Q_samp, hkl, intensity = builder.compute_full()

# optional crop (detector indices)
builder.crop_by_positions(y_bound=(y0, y1), x_bound=(x0, x1))

# re-grid (ranges=None → auto min/max)
grid, (Hax, Kax, Lax) = builder.regrid_xu(
    space="hkl",
    grid_shape=(100, 100, 100),
    ranges=None,
    fuzzy=False,
    normalize="mean",
    stream=True,
)
```

## Napari 3D Workflow

Notebook: `examples/rsm3d_napari_workflow.ipynb`

Launch:

```bash
pixi run jupyter lab examples/rsm3d_napari_workflow.ipynb
```

Programmatic:

```python
from rsm3d.data_viz import RSMNapariViewer
viewer = RSMNapariViewer(
    grid, (Hax, Kax, Lax),
    space="hkl",
    log_view=True,
    contrast_percentiles=(1, 99.8),
).launch()
```

## Tips

- Let `ranges=None` for automatic HKL limits.
- Crop early to reduce memory/time.
- Use log view for broad intensity dynamic range.

## Dependencies (managed by pixi)

Core: numpy, scipy, h5py, tifffile, hklpy, napari, xrayutilities (optional).
List them:

```bash
pixi run python -m pip list
```

## Contributing

```bash
git checkout -b feature/thing
# edit
pixi run pytest
git commit -am "feat: thing"
git push origin feature/thing
```

Issues / PRs welcome.
