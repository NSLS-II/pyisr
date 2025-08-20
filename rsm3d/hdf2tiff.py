import os
import h5py
import hdf5plugin
import numpy as np
import tifffile

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