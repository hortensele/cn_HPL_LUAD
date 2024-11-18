import h5py
import pandas as pd
import numpy as np
import os
from glob import glob
from scipy.spatial import distance_matrix
from itertools import combinations_with_replacement
from tqdm import tqdm

# Workflow

# 1) Load hdf5 dataset file.

# 2) Create dataframes with all slide IDs and tile IDs.

# 3a) Group by slide and load corresponding hovernet csv with all cells, 

# 3b) For each tile in hdf5 file: subset to all rows that are in the tile, compute distance between each pair of cells, and determine the types of cells of those that are the closest to each other

def rectify_h5_slides(hdf5_path):
    """ Load the slides and tiles information from the subset of tiles used for HPL """
    with h5py.File(hdf5_path, mode = "r+") as h5_data:
        slides = h5_data["slides"][:].astype(str)
        tiles = h5_data["tiles"][:].astype(str)  
    return pd.DataFrame({"slides": slides, 
                         "tiles": tiles})

def grab_slides_tiles(hdf5_path):
    """ Load the slides and tiles information from the subset of tiles used for HPL """
    with h5py.File(hdf5_path, mode = "r+") as h5_data:
        slides = h5_data["slides"][:].astype(str)
        tiles = h5_data["tiles"][:].astype(str)  
    return pd.DataFrame({"slides": slides, 
                         "tiles": tiles})

def grab_hovernet_csv(slide_name, results_files):
    """ Finds the corresponding csv file with Hover-Net results given the slide name """
    slide_results_path = [element for element in results_files if slide_name in element][0]
    full_slide_results_path = os.path.join(slide_results_path, "cell_summary")
    csv_path = os.path.join(full_slide_results_path, os.path.basename(slide_results_path).replace("_files", ".csv"))
    return pd.read_csv(csv_path)

def find_tile_cells(tile_name, slide_df, tile_size):
    """
    Find all the cells within a tile.
    
    Parameters:
    - tile_name: ID of the tile.
    - slide_df: DataFrame containing all cells within a slide.
    - tile_size: size of the tile
    
    Returns:
    - A DataFrame with all cells within a tile.
    """
    tile_row = slide_df[slide_df.CellID == tile_name]
    tile_minx = tile_row.Centroid_x.values[0] - tile_size/2
    tile_maxx = tile_row.Centroid_x.values[0] + tile_size/2
    tile_miny = tile_row.Centroid_y.values[0] - tile_size/2
    tile_maxy = tile_row.Centroid_y.values[0] + tile_size/2
    tile_df = slide_df.loc[(slide_df.Centroid_x >= tile_minx) &
                            (slide_df.Centroid_x < tile_maxx) &
                            (slide_df.Centroid_y >= tile_miny) &
                            (slide_df.Centroid_y < tile_maxy),:]
    return tile_df

def compute_cell_attributes(df, celltype_df):
    """
    Compute the mean of a couple of properties of all cells within a tile.
    
    Parameters:
    - df: DataFrame containing cells within a tile.
    - celltype_df: DataFrame with cell types labels name from Hover-Net.
    
    Returns:
    - A Series with cell properties information.
    """
    df_cell_properties = pd.Series(dtype="float64")
    for l in celltype_df.labels:
        cell_class = celltypes.types[celltype_df.labels == l].values[0]
        df_cell_properties.loc[l] = len(df.CellType[df.CellType == cell_class])
        df_cell_properties.loc["mean_area_"+l] = np.mean(df.Area[df.CellType == cell_class])
        df_cell_properties.loc["mean_perimeter_"+l] = np.mean(df.Perimeter[df.CellType == cell_class])
        df_cell_properties.loc["mean_circularity_"+l] = np.mean(df.Circularity[df.CellType == cell_class])
        df_cell_properties.loc["mean_elongation_"+l] = np.mean(df.Elongation[df.CellType == cell_class])
        df_cell_properties.loc["mean_MinDiaR_"+l] = np.mean(df.MinDiaR[df.CellType == cell_class])
        df_cell_properties.loc["mean_MaxDiaR_"+l] = np.mean(df.MaxDiaR[df.CellType == cell_class])
        df_cell_properties.loc["mean_Rec_Area2_"+l] = np.mean(df.Rec_Area2[df.CellType == cell_class])
        df_cell_properties.loc["mean_AspectRatio_"+l] = np.mean(df.AspectRatio[df.CellType == cell_class])
        df_cell_properties.loc["mean_MinAx_"+l] = np.mean(df.MinAx[df.CellType == cell_class])
        df_cell_properties.loc["mean_MaxAx_"+l] = np.mean(df.MaxAx[df.CellType == cell_class])
        df_cell_properties.loc["mean_HullArea_"+l] = np.mean(df.HullArea[df.CellType == cell_class])
        df_cell_properties.loc["mean_Solidity_"+l] = np.mean(df.Solidity[df.CellType == cell_class])
        df_cell_properties.loc["mean_Extent_"+l] = np.mean(df.Extent[df.CellType == cell_class])
    return df_cell_properties.to_frame().transpose()

def compute_cell_interactions(df, distance_threshold, celltype_df):
    """
    Compute cell-cell interactions within a tile.
    
    Parameters:
    - df: DataFrame containing cells within a tile.
    - distance_threshold: Distance threshold to consider cells neighbors.
    - celltype_df: DataFrame with cell types labels name from Hover-Net.
    
    Returns:
    - A DataFrame with interaction information.
    """
    if df.shape[0] < 2:
        return pd.DataFrame()  # No interactions to compute if less than 2 cells
    
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
    """
    Summarize number of different of interactions into a row.
    
    Parameters:
    - interactions_df: DataFrame with all interactions.
    - interaction_types: List of all possible pairs of cell types.
    
    Returns:
    - A DataFrame with summarized interactions.
    """
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
    
    
def main(hdf5_path, hovernet_results_path, out_dir, tile_size = 224):
    """ Main function to process all images and compute interactions. """
    results_files = glob(os.path.join(hovernet_results_path,"*_files"))

    df_h5py = grab_slides_tiles(hdf5_path)

    # Group by slides
    df_h5py_grouped = df_h5py.groupby("slides")

    celltypes = pd.DataFrame({"types": [0, 1, 2, 3, 4, 5],
                            "labels": ["others", "neoplastic", "inflammatory", "connective", "necrosis", "non_neoplastic"]})
    interaction_types = list(combinations_with_replacement(celltypes.labels, 2))
    
    all_slide_names = df_h5py_grouped.groups.keys()
    
    for slide_name in tqdm(all_slide_names, desc="Processing slides"):
        out_path = os.path.join(out_dir, slide_name + ".csv")
        if not os.path.exists(out_path):  
            results_df = pd.DataFrame()
            print(f"Slide: {slide_name}")
            group = df_h5py_grouped.get_group(slide_name)
            hovernet_df = grab_hovernet_csv(slide_name, results_files)
            run_cell_properties_flag = "mean_area_inflammatory" not in hovernet_df.columns
            group = group.reset_index()
            for i, row in group.iterrows():
                tile = row.tiles
                hovernet_row = hovernet_df[hovernet_df.CellID == tile]
                try:
                    hovernet_row.index = [i]
                    hovernet_row = pd.concat([hovernet_row, pd.DataFrame({"slides": slide_name}, index = [i])], axis = 1)
                    tile_df = find_tile_cells(tile, hovernet_df, tile_size)

                    if run_cell_properties_flag:
                        tile_properties = compute_cell_attributes(tile_df, celltypes)
                        tile_properties.index = [i]
                        hovernet_row = pd.concat([hovernet_row, tile_properties], axis = 1)

                    interactions_df = compute_cell_interactions(tile_df, 30, celltypes)
                    summary_interactions = summarize_cell_interactions(interactions_df, interaction_types)
                    summary_interactions.index = [i]
                    results_df = pd.concat([results_df, pd.concat([hovernet_row, summary_interactions], axis = 1)], axis = 0, ignore_index=True)
                except:
                    print("Failed with tile {} of slide {}".format(tile, slide_name))
            results_df.to_csv(out_path, index = False)
    return results_df

# Variables to specify
hdf5_path = "/gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/tcga_luad_cell_224px_20x/h224_w224_n3_zdim128/hdf5_tcga_luad_cell_224px_20x_he_complete.h5"
hovernet_results_path = "/gpfs/data/tsirigoslab/home/leh06/HoverNet/hover_net/01_results/tcga_luad_240323/004_Van_NYUape_pannuke_type/224px_tcga_luad_0.504umpx/"
out_dir = os.path.join(os.getcwd(),"results", "tcga_luad_cell_224px_20x")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
main(hdf5_path, hovernet_results_path, out_dir)
