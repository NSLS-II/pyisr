# HDF5 to TIFF Conversion and RSM Visualization

This repository contains scripts to process HDF5 data files, convert them into TIFF format, and visualize data using reciprocal space mapping (RSM). 

## Features
1. **HDF5 to TIFF Conversion**: 
   - Reads HDF5 files containing scientific data.
   - Extracts datasets and saves them as TIFF files for further analysis or visualization.

2. **Reciprocal Space Mapping (RSM)**:
   - Visualizes data from reciprocal space to real space using the `rsm` function.
   - Provides 3D and 2D projection plots to analyze diffraction patterns.

---

## Requirements
Ensure you have the following Python libraries installed:
- `numpy`
- `h5py`
- `tifffile`
- `matplotlib`
- `pyvista`
- `imageio`

Install dependencies using pip:
```bash
pip install numpy h5py tifffile matplotlib pyvista imageio
```

---

## Usage

### **HDF5 to TIFF Conversion**
1. Set the input directory containing `.h5` or `.hdf5` files.
2. Specify the output directory for saving the converted TIFF files.
3. Run the script:
   ```bash
   python hdf5_to_tiff.py
   ```

#### Example:
```python
input_directory = "/path/to/input/hdf5_files"
output_directory = "/path/to/output/tiff_files"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Convert HDF5 to TIFF
from hdf2tiff import hdf2tiff
input_directory = "/nsls2/data/staff/xyang4/data_cs/ca3ir4sn13/setup_6oct23/"
output_directory = "/nsls2/data/staff/xyang4/data_cs/ca3ir4sn13/setup_6oct23_tiff_0910_tmp/"
hdf2tiff(input_directory, output_directory)
```

---

### **Reciprocal Space Mapping (RSM)**
The `rsm` function visualizes TIFF data from reciprocal space. It generates:
- A 3D isosurface plot for the reciprocal space.
- A 2D projection plot for specific data planes.

#### Parameters:
- `path`: Directory containing TIFF files.
- `fnamepng`: Output prefix for saved visualization images.
- `cmap`: Colormap for rendering.
- `op`: Opacity for the 3D plot.
- `md`, `nd`: Dead pixel coordinates (optional).
- `el`, `az`: Camera elevation and azimuth for 3D visualization.

#### Example:
```python
from rsm_visualization import rsm

# Visualize the TIFF files from reciprocal space to real space
input_tiff_path = "/path/to/tiff/files"
output_visual_prefix = "visualization_output"

rsm(input_tiff_path, output_visual_prefix, cmap='viridis', el=30, az=45)
```

---

## Output
1. **TIFF Files**:
   - Saved in the specified output directory with meaningful names derived from the HDF5 dataset.
   
2. **RSM Visualizations**:
   - 3D Isosurface plot: `<output_prefix>_3D.png`
   - 2D Projection plot: `<output_prefix>_2D.png`

---

## License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

---

## Contact
For questions, suggestions, or contributions, feel free to open an issue or contact the author.