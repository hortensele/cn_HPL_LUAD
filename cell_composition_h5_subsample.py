from glob import glob
import pandas as pd
import h5py
import os
import numpy as np
import argparse

def filter_csv_by_h5(hovernet_folder, h5_file, output_name, window_size):
    """
    Filters rows from CSV files based on key fields from an H5 file and saves the results to new CSVs.
    
    Parameters:
    - hovernet_folder: Parent folder containing all the Hover-Net results.
    - h5_file: H5 file containing the key fields.
    - output_name: Name of output csv.
    - window_size: Tile size to find the correct Hovernet subfolder.
    """
    
    # Number of rows to read at a time from each CSV file.
    chunk_size = 100000
    
    # Load key fields from the H5 file
    with h5py.File(h5_file, 'r') as h5f:
        h5_slides = np.array(h5f["slides"][()].astype(str))
        h5_tiles = np.array(h5f["tiles"][()].astype(str))
        
    # Change this later
    if window_size == 224:
        csv_list = glob(os.path.join(hovernet_folder, "*_files", "cell_summary", "*.csv"))
    else:
        csv_list = glob(os.path.join(hovernet_folder, "*_files", "cell_summary_" + str(window_size) + "px", "*.csv"))
        
    combined_df = []
    # Iterate through each CSV file in the folder
    for csv_path in csv_list:
        
        slide_name = os.path.basename(csv_path).split(".")[0]
        
        # Process CSV 
        reader = pd.read_csv(csv_path, chunksize=chunk_size)
        for chunk in reader:
            # Filter the h5_tiles based on h5_slides matching the slide_name
            valid_h5_tiles = h5_tiles[h5_slides == slide_name]
            valid_h5_tiles = valid_h5_tiles.flatten()
            # Filter rows where both columns match keys from the H5 file
            filtered_chunk = chunk[chunk["CellID"].isin(valid_h5_tiles)]
            filtered_chunk["patient"] = slide_name[:12]
            filtered_chunk["slide"] = slide_name

            # Append the filtered chunk to the list
            combined_df.append(filtered_chunk)

        print(f"Processed {slide_name}")
    # Combine all filtered dataframes into one
    if combined_df:
        final_df = pd.concat(combined_df, ignore_index=True)
        
        # Save the combined dataframe to a single CSV file
        output_csv_path = os.path.join(hovernet_folder, output_name)
        final_df.to_csv(output_csv_path, index=False)
        
        print(f"Combined data saved to {output_csv_path}")
    else:
        print("No data to combine.")

def main():
    parser = argparse.ArgumentParser(description='Filter CSV files based on H5 file for efficient analysis.')
    
    # Arguments for specifying input files, folder, and window size
    parser.add_argument('--hovernet_folder', type=str, help='Path to the folder containing the HoVer-Net CSV files.')
    parser.add_argument('--h5_file', type=str, help='Path to the H5 file containing key fields.')
    parser.add_argument('--output_name', type=str, default='output_name', help='Output filename to save filtered CSV.')
    parser.add_argument('--window_size', type=int, default=60, help='Tile size for current analysis.')
    
    args = parser.parse_args()
    
    # Call the filtering function with the provided arguments
    filter_csv_by_h5(
        hovernet_folder=args.hovernet_folder,
        h5_file=args.h5_file,
        output_name=args.output_name,
        window_size=args.window_size
    )

if __name__ == "__main__":
    main()