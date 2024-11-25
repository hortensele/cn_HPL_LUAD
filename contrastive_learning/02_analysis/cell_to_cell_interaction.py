import h5py
import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.spatial import distance_matrix
from itertools import combinations_with_replacement
from tqdm import tqdm
import argparse

def grab_slides_tiles(hdf5_path):
    '''
    Extracts slide and tile information from an HDF5 object.

    Args:
        hdf5_path (str): Path to the HDF5 file containing slide and tile information.

    Returns:
        pd.DataFrame: A DataFrame containing slide IDs and corresponding tile IDs.
    '''
    with h5py.File(hdf5_path, mode="r") as h5_data:
        slides = h5_data["slides"][:].astype(str)
        tiles = h5_data["tiles"][:].astype(str)  
    return pd.DataFrame({"slides": slides, "tiles": tiles})

def grab_hovernet_csv(slide_name, results_files):
    '''
    Retrieves the Hover-Net cell summary CSV for a specified slide.

    Args:
        slide_name (str): Name of the slide.
        results_files (list): List of paths to Hover-Net results directories.

    Returns:
        pd.DataFrame: A DataFrame containing cell data for the slide.
    '''
    slide_results_path = [element for element in results_files if slide_name in element][0]
    full_slide_results_path = os.path.join(slide_results_path, "cell_summary")
    csv_path = os.path.join(full_slide_results_path, os.path.basename(slide_results_path).replace("_files", ".csv"))
    return pd.read_csv(csv_path)

def find_tile_cells(tile_name, slide_df, tile_size):
    '''
    Identifies all cells within a specified tile based on their coordinates.

    Args:
        tile_name (str): Name of the tile.
        slide_df (pd.DataFrame): DataFrame containing cell data for the slide.
        tile_size (float): Size of the tile.

    Returns:
        pd.DataFrame: A subset of the slide DataFrame containing cells in the specified tile.
    '''
    tile_row = slide_df[slide_df.CellID == tile_name]
    tile_minx = tile_row.Centroid_x.values[0] - tile_size / 2
    tile_maxx = tile_row.Centroid_x.values[0] + tile_size / 2
    tile_miny = tile_row.Centroid_y.values[0] - tile_size / 2
    tile_maxy = tile_row.Centroid_y.values[0] + tile_size / 2
    tile_df = slide_df.loc[
        (slide_df.Centroid_x >= tile_minx) & (slide_df.Centroid_x < tile_maxx) &
        (slide_df.Centroid_y >= tile_miny) & (slide_df.Centroid_y < tile_maxy), :]
    return tile_df

def compute_cell_attributes(df, celltype_df):
    '''
    Calculates mean cell properties and counts for each cell type within a tile.

    Args:
        df (pd.DataFrame): DataFrame containing cell data for the tile.
        celltype_df (pd.DataFrame): DataFrame defining cell type labels and their corresponding names.

    Returns:
        pd.DataFrame: A DataFrame summarizing cell counts and mean properties by cell type.
    '''
    df_cell_properties = pd.Series(dtype="float64")
    for l in celltype_df.labels:
        cell_class = celltype_df.types[celltype_df.labels == l].values[0]
        df_cell_properties.loc[l] = len(df.CellType[df.CellType == cell_class])
        df_cell_properties.loc[f"mean_area_{l}"] = np.mean(df.Area[df.CellType == cell_class])
        # Add additional properties as required...
    return df_cell_properties.to_frame().transpose()

def compute_cell_interactions(df, distance_threshold, celltype_df):
    '''
    Computes cell-cell interactions within a tile based on a distance threshold.

    Args:
        df (pd.DataFrame): DataFrame containing cell data for the tile.
        distance_threshold (float): Maximum distance to consider cells as interacting.
        celltype_df (pd.DataFrame): DataFrame defining cell type labels and their corresponding names.

    Returns:
        pd.DataFrame: A DataFrame summarizing cell-cell interactions, including distances and cell types.
    '''
    if df.shape[0] < 2:
        return pd.DataFrame()  # No interactions to compute if fewer than 2 cells
    
    coords = df[['Centroid_x', 'Centroid_y']].values
    dist_matrix = distance_matrix(coords, coords)
    interactions = []

    cell_ids = df['CellID'].values
    classes = df['CellType'].values
    
    for i in range(len(cell_ids)):
        for j in range(i + 1, len(cell_ids)):
            distance = dist_matrix[i, j]
            if distance <= distance_threshold:
                interactions.append({
                    'cell1_id': cell_ids[i],
                    'cell2_id': cell_ids[j],
                    'distance': distance,
                    'class1': celltype_df.labels[celltype_df.types == classes[i]].values[0],
                    'class2': celltype_df.labels[celltype_df.types == classes[j]].values[0]
                })

    return pd.DataFrame(interactions)

def summarize_cell_interactions(interactions_df, interaction_types):
    '''
    Summarizes the number of different types of cell-cell interactions within a tile.

    Args:
        interactions_df (pd.DataFrame): DataFrame of cell-cell interactions.
        interaction_types (list): List of all possible interaction types (pairs of cell classes).

    Returns:
        pd.DataFrame: A one-row DataFrame with counts of each interaction type.
    '''
    all_interactions_df = pd.DataFrame(interaction_types, columns=['class1', 'class2'])
    if len(interactions_df) > 0:
        df = interactions_df.groupby(['class1', 'class2']).size().reset_index(name='count')
        summary_complete = all_interactions_df.merge(df, on=['class1', 'class2'], how='left').fillna(0)

        summary = summary_complete.set_index(['class1', 'class2'])['count']
        summary.index = summary.index.map(lambda x: f'{x[0]}_{x[1]}')
    else:
        summary_complete = all_interactions_df
        summary_complete["count"] = 0
        summary = summary_complete.set_index(['class1', 'class2'])['count']
        summary.index = summary.index.map(lambda x: f'{x[0]}_{x[1]}')
    return summary.astype(int).to_frame().transpose()

def main(hdf5_path, hovernet_results_path, out_dir, tile_size=224):
    '''
    Processes HDF5 and Hover-Net data to compute and save cell interactions for each slide.

    Args:
        hdf5_path (str): Path to the HDF5 file containing slide and tile information.
        hovernet_results_path (str): Path to the directory with Hover-Net results.
        out_dir (str): Directory to save the output CSV files.
        tile_size (int, optional): Size of the tiles. Default is 224.

    Returns:
        None
    '''
    results_files = glob(os.path.join(hovernet_results_path, "*_files"))

    df_h5py = grab_slides_tiles(hdf5_path)
    df_h5py_grouped = df_h5py.groupby("slides")

    celltypes = pd.DataFrame({"types": [0, 1, 2, 3, 4, 5],
                              "labels": ["others", "neoplastic", "inflammatory", "connective", "necrosis", "non_neoplastic"]})
    interaction_types = list(combinations_with_replacement(celltypes.labels, 2))
    
    all_slide_names = df_h5py_grouped.groups.keys()
    
    for slide_name in tqdm(all_slide_names, desc="Processing slides"):
        out_path = os.path.join(out_dir, slide_name + ".csv")
        if not os.path.exists(out_path):
            results_df = pd.DataFrame()
            group = df_h5py_grouped.get_group(slide_name)
            hovernet_df = grab_hovernet_csv(slide_name, results_files)
            run_cell_properties_flag = "mean_area_inflammatory" not in hovernet_df.columns
            group = group.reset_index()
            for i, row in group.iterrows():
                tile = row.tiles
                tile_df = find_tile_cells(tile, hovernet_df, tile_size)

                if run_cell_properties_flag:
                    tile_properties = compute_cell_attributes(tile_df, celltypes)
                    tile_properties.index = [i]

                interactions_df = compute_cell_interactions(tile_df, 30, celltypes)
                summary_interactions = summarize_cell_interactions(interactions_df, interaction_types)
                summary_interactions.index = [i]
                results_df = pd.concat([results_df, summary_interactions], axis=0, ignore_index=True)
            results_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 data and compute cell interactions.")
    parser.add_argument("--hdf5_path", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--hovernet_results_path", required=True, help="Path to the Hover-Net results.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output CSV files.")
    parser.add_argument("--tile_size", type=int, default=224, help="Size of the tiles (default: 224).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    main(args.hdf5_path, args.hovernet_results_path, args.out_dir, args.tile_size)

