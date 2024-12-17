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

def save_tiff_data(tiff_data, output_dir, original_filename):
    """
    Save the TIFF data as an image file in a specified directory using tifffile.
    
    Parameters:
    tiff_data (numpy array): The TIFF data array.
    output_dir (str): The directory to save the TIFF file.
    original_filename (str): The original HDF5 filename, used for naming the output TIFF file.
    """
    # Remove the extension from the original filename and append .tiff
    output_filename = os.path.splitext(original_filename)[0] + ".tiff"
    output_path = os.path.join(output_dir, output_filename)
    
    # Check if the data type is string/unicode or numeric
    if tiff_data.dtype.kind in {'U', 'S'}:
        print(f"Data is not numerical: {tiff_data.dtype}. Skipping conversion for {original_filename}.")
    else:
        # Save the TIFF using tifffile
        tifffile.imwrite(output_path, tiff_data.astype('uint8'))  # Convert to uint8 or other valid format
        print(f"Saved TIFF to {output_path}")
        
        
def hdf2tiff(input_directory: str, output_directory: str):
    """
    Main function to read HDF5 files, extract TIFF data, and save them.

    Parameters:
        input_directory (str): Path to the input directory containing HDF5 files.
        output_directory (str): Path to the output directory for saving TIFF files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read HDF5 files and extract TIFF data
    tiff_data_dict = read_hdf5_tiff_data(input_directory)

    # Save TIFF data to the output directory
    for file_name, data in tiff_data_dict.items():
        save_tiff_data(data, output_directory, file_name)