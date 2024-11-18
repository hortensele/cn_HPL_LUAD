from glob import glob
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, io
import cv2
import openslide
import argparse
import random

##### Main #######
parser = argparse.ArgumentParser(description='Sort tiles according to cell types')
parser.add_argument('--output_dir', dest='output_dir', type = str, default = None, help = "Output directory.")
parser.add_argument('--results_dir', dest='results_dir', type=str, default=None, help='Parent directory where all Hover-Net results are stored.')
parser.add_argument('--svs_path', dest='svs_path', type=str, default=None, help='Folder where all svs files are stored.')
parser.add_argument('--window_size', dest='window_size', type = int, default = 224, help = "Size of tiles at 20x.")
parser.add_argument('--perc_train', dest='perc_train', type = float, default = 0.8, help = "Ratio of patients to belong to train set.")
parser.add_argument('--nb_tiles_train', dest='nb_tiles_train', type = int, default = None, help = "Number of tiles for train set.")
parser.add_argument('--nb_tiles_valid', dest='nb_tiles_valid', type = int, default = None, help = "Number of tiles for valid set.")
parser.add_argument('--nb_tiles_test', dest='nb_tiles_test', type = int, default = None, help = "Number of tiles for test set.")


args					= parser.parse_args()
output_dir				= args.output_dir
results_dir				= args.results_dir
svs_path				= args.svs_path
window_size				= args.window_size
perc_train				= args.perc_train
nb_tiles_train			= args.nb_tiles_train
nb_tiles_valid			= args.nb_tiles_valid
nb_tiles_test			= args.nb_tiles_test

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_paths = glob(os.path.join(results_dir,"*_files","cell_summary", "*.csv"))

# Add threshold of cells to to load from csvs in case of memory issue
target_sample_size = 100000

# Load all Hover-Net cell results stored in CSVs and separate into cell type dataframes
neoplastic_df = pd.DataFrame()
inflammatory_df = pd.DataFrame()
connective_df = pd.DataFrame()
necrosis_df = pd.DataFrame() 
non_neoplastic_df = pd.DataFrame()

for i in tqdm(csv_paths, desc="Processing csvs"):
    df = pd.read_csv(i)
    df["patient"] = os.path.basename(i)[:12]
    df["slide"] = os.path.basename(i)[:-4]
    
    # Sample required number of rows if the DataFrame is larger than the target sample size
    if len(df.loc[df.CellType == 1]) > target_sample_size:
        neoplastic_df = pd.concat([neoplastic_df, df.loc[df.CellType == 1].sample(n=target_sample_size, random_state=42)], ignore_index=True)
    else:
        neoplastic_df = pd.concat([neoplastic_df, df.loc[df.CellType == 1]], ignore_index=True)
        
    if len(df.loc[df.CellType == 2]) > target_sample_size:
        inflammatory_df = pd.concat([inflammatory_df, df.loc[df.CellType == 2].sample(n=target_sample_size, random_state=42)], ignore_index=True)
    else:
        inflammatory_df = pd.concat([inflammatory_df, df.loc[df.CellType == 2]], ignore_index=True)
        
    if len(df.loc[df.CellType == 3]) > target_sample_size:
        connective_df = pd.concat([connective_df, df.loc[df.CellType == 3].sample(n=target_sample_size, random_state=42)], ignore_index=True)
    else:
        connective_df = pd.concat([connective_df, df.loc[df.CellType == 3]], ignore_index=True)
        
    if len(df.loc[df.CellType == 4]) > target_sample_size:
        necrosis_df = pd.concat([necrosis_df, df.loc[df.CellType == 4].sample(n=target_sample_size, random_state=42)], ignore_index=True)
    else:
        necrosis_df = pd.concat([necrosis_df, df.loc[df.CellType == 4]], ignore_index=True)
        
    if len(df.loc[df.CellType == 5]) > target_sample_size:
        non_neoplastic_df = pd.concat([non_neoplastic_df, df.loc[df.CellType == 5].sample(n=target_sample_size, random_state=42)], ignore_index=True)
    else:
        non_neoplastic_df = pd.concat([non_neoplastic_df, df.loc[df.CellType == 5]], ignore_index=True)


# Combine them into a single list for easier iteration
all_cells_df = [neoplastic_df, inflammatory_df, connective_df, necrosis_df, non_neoplastic_df]

# Generate train samples and valid samples sets
samples_from_csv = np.unique([os.path.basename(i)[:12] for i in csv_paths])
random.Random(4).shuffle(samples_from_csv)
train_samples = samples_from_csv[:int(perc_train*len(samples_from_csv))]
valid_samples = samples_from_csv[int(perc_train*len(samples_from_csv)):]

def sample_tiles_subset(subset, samples, all_cells_df, nb_tiles, tiles_to_remove):
    ''' This function randomly samples tiles from each cell type for the given samples.'''
    # Initialize an empty list to store sampled tiles
    sampled_tiles = []

    # Calculate the number of tiles to sample from each dataframe
    tiles_per_dataframe = nb_tiles // len(all_cells_df)

    # Iterate over each dataframe
    for df in all_cells_df:
        
        # Remove either samples or tiles that have already been sampled.
        df = df[df.patient.isin(samples)]
        if not tiles_to_remove.empty:
            df = df[~((df.slide.isin(tiles_to_remove.slide)) & (df.CellID.isin(tiles_to_remove.CellID)))]    
        
        # Group dataframe by patient
        grouped = df.groupby('patient')

        # Initialize an empty list to store sampled tiles for this dataframe
        df_sampled_tiles = []

        # Sample tiles from each patient
        for _, group in grouped:
            # Sample tiles from this patient
            num_tiles = min(len(group), tiles_per_dataframe // len(grouped))  # Ensure sampling from each patient is proportional
            sampled_group = group.sample(n=num_tiles, random_state=42)

            # Add sampled tiles from this patient to the list
            df_sampled_tiles.append(sampled_group)

        # Concatenate sampled tiles from this dataframe
        df_sampled = pd.concat(df_sampled_tiles)

        # Add sampled tiles from this dataframe to the overall list
        sampled_tiles.append(df_sampled)

    # Concatenate sampled tiles from all dataframes
    final_sampled_tiles = pd.concat(sampled_tiles)

    # Shuffle the final sampled tiles
    final_sampled_tiles = final_sampled_tiles.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_sampled_tiles["subset"] = subset
    return final_sampled_tiles

# Tile samples for each set
tiles_to_remove = pd.DataFrame()
train_sampled_tiles = sample_tiles_subset("train", train_samples, all_cells_df, nb_tiles_train, tiles_to_remove)
valid_sampled_tiles = sample_tiles_subset("valid", valid_samples, all_cells_df, nb_tiles_valid, tiles_to_remove)

tiles_to_remove = pd.concat([train_sampled_tiles, valid_sampled_tiles], axis=0)
tiles_to_remove.reset_index(drop=True, inplace=True)

test_sampled_tiles = sample_tiles_subset("test", samples_from_csv, all_cells_df, nb_tiles_test, tiles_to_remove)

final_sampled_tiles = pd.concat([tiles_to_remove, test_sampled_tiles], axis=0)
final_sampled_tiles.reset_index(drop=True, inplace=True)

# Save all the sampled tiles into one CSV with all the cell information
final_sampled_tiles.to_csv(os.path.join(results_dir, "tiles_used_for_ssl_information.csv"), index = False)

print("Sampling done, start tiling")



def convert_mags(coords_20x, final_mag):
    ''' This function converts coordinates at 20x to 40x (original magnification of the whole slide images).'''
    final_coords = coords_20x*(447/224*0.2525)/final_mag
    return int(final_coords)

def normalize_tile(tile, NormVec):
    ''' This function performs stain normalization for each tile.'''
    NormVec = [float(x) for x in NormVec.split(',')]
    Lab = RGB_to_lab(tile)
    TileMean = [0,0,0]
    TileStd = [1,1,1]
    newMean = NormVec[0:3] 
    newStd = NormVec[3:6]
    for i in range(3):
        TileMean[i] = np.mean(Lab[:,:,i])
        TileStd[i] = np.std(Lab[:,:,i])
        tmp = ((Lab[:,:,i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
        if i == 0:
            tmp[tmp<0] = 0 
            tmp[tmp>100] = 100 
            Lab[:,:,i] = tmp
        else:
            tmp[tmp<-128] = 128 
            tmp[tmp>127] = 127 
            Lab[:,:,i] = tmp
    tile = Lab_to_RGB(Lab)
    return tile

def RGB_to_lab(tile):
    Lab = color.rgb2lab(tile)
    return Lab

def Lab_to_RGB(Lab):
    newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
    return newtile

Normalize = '57,22,-8,20,10,5'

# Tiling step: this will output tiles centered around cells detected by Hover-Net.
svs_lists = glob(os.path.join(svs_path,"*.svs"))
for slide_name in tqdm(final_sampled_tiles["slide"].unique(), desc="Processing slides"):
    final_sampled_tiles_sub = final_sampled_tiles.loc[final_sampled_tiles.slide == slide_name,:]
    slide_path = [i for i in svs_lists if os.path.splitext(os.path.basename(i))[0] == slide_name][0]
    slide = openslide.OpenSlide(slide_path)
    window_size_40x = convert_mags(window_size, float(slide.properties["openslide.mpp-x"]))
    for index, row in final_sampled_tiles_sub.iterrows():
        x = convert_mags(row["Centroid_x"], float(slide.properties["openslide.mpp-x"]))
        y = convert_mags(row["Centroid_y"], float(slide.properties["openslide.mpp-x"]))
        tile = slide.read_region((int(x-window_size_40x/2),int(y-window_size_40x/2)),0,(window_size_40x,window_size_40x))
        tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
        tile = cv2.resize(tile, (window_size,window_size)) 
        tile = Image.fromarray(normalize_tile(tile, Normalize).astype('uint8'),'RGB')
        tile_outpath = os.path.join(output_dir, row["subset"] + "_" + row["slide"] + "_" + row["CellID"])
        if not os.path.isfile(tile_outpath):
            tile.save(tile_outpath)
