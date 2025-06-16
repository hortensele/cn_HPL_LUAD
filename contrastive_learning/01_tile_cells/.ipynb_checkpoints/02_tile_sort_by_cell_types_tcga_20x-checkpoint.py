import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage import color
import openslide


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Sort tiles according to cell types')
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory.")
    parser.add_argument('--results_dir', type=str, required=True, help="Parent directory for Hover-Net results.")
    parser.add_argument('--wsi_path', type=str, required=True, help="Folder with whole slide images.")
    parser.add_argument('--window_size', type=int, default=224, help="Size of tiles at 20x.")
    parser.add_argument('--perc_train', type=float, default=0.8, help="Ratio of patients in the train set.")
    parser.add_argument('--nb_tiles_train', type=int, default=None, help="Number of tiles for the train set.")
    parser.add_argument('--nb_tiles_valid', type=int, default=None, help="Number of tiles for the validation set.")
    parser.add_argument('--nb_tiles_test', type=int, default=None, help="Number of tiles for the test set.")
    parser.add_argument('--tcga_flag', action='store_true', help="Flag for TCGA images.")
    return parser.parse_args()


def sample_tiles_subset(subset, samples, all_cells_df, nb_tiles, tiles_to_remove):
    """
    Randomly samples tiles from each cell type for the given samples.

    Args:
        subset (str): The subset name (e.g., 'train', 'valid', 'test').
        samples (list): List of samples to consider for the subset.
        all_cells_df (list of pandas.DataFrame): List of dataframes, one for each cell type.
        nb_tiles (int): Total number of tiles to sample.
        tiles_to_remove (pandas.DataFrame): DataFrame of previously sampled tiles to avoid duplicates.

    Returns:
        pandas.DataFrame: DataFrame containing the sampled tiles.
    """
    sampled_tiles = []
    tiles_per_dataframe = nb_tiles // len(all_cells_df)

    for df in all_cells_df:
        df = df[df.patient.isin(samples)]
        if not tiles_to_remove.empty:
            df = df[~((df.slide.isin(tiles_to_remove.slide)) & (df.CellID.isin(tiles_to_remove.CellID)))]

        grouped = df.groupby('patient')
        df_sampled_tiles = []

        for _, group in grouped:
            num_tiles = min(len(group), tiles_per_dataframe // len(grouped))
            df_sampled_tiles.append(group.sample(n=num_tiles, random_state=42))

        sampled_tiles.append(pd.concat(df_sampled_tiles))

    final_sampled_tiles = pd.concat(sampled_tiles).sample(frac=1, random_state=42).reset_index(drop=True)
    final_sampled_tiles["subset"] = subset
    return final_sampled_tiles


def convert_mags(coords_20x, final_mag):
    """
    Converts coordinates from 20x magnification to another magnification level.

    Args:
        coords_20x (int): Coordinates at 20x magnification.
        target_pixel_size (float): Desired target pixel size level.

    Returns:
        int: Converted coordinates at the target pixel size.
    """
    return int(coords_20x * (447 / 224 * 0.2525) / final_mag)


def normalize_tile(tile, norm_vec):
    """
    Normalizes the stain of an image tile to match a target stain vector.

    Args:
        tile (numpy.ndarray): Input image tile in RGB format.
        NormVec (str): Target stain normalization vector, a comma-separated string of six values.

    Returns:
        numpy.ndarray: Stain-normalized image tile in RGB format.
    """
    norm_vec = [float(x) for x in norm_vec.split(',')]
    lab = color.rgb2lab(tile)
    tile_mean = [np.mean(lab[:, :, i]) for i in range(3)]
    tile_std = [np.std(lab[:, :, i]) for i in range(3)]
    new_mean, new_std = norm_vec[:3], norm_vec[3:6]

    for i in range(3):
        lab[:, :, i] = ((lab[:, :, i] - tile_mean[i]) * (new_std[i] / tile_std[i])) + new_mean[i]
        if i == 0:
            lab[:, :, i] = np.clip(lab[:, :, i], 0, 100)
        else:
            lab[:, :, i] = np.clip(lab[:, :, i], -128, 127)

    return (color.lab2rgb(lab) * 255).astype(np.uint8)


def process_tiles(final_sampled_tiles, wsi_path, output_dir, window_size, normalize = '57,22,-8,20,10,5', tcga_flag=True):
    """
    Generates and saves image tiles based on a dataframe of sampled tiles. The function processes each slide, 
    extracts regions from the corresponding whole slide images (WSIs), resizes them to the desired window size, 
    and saves them as JPEG files. Optionally, tiles can be normalized.

    Args:
        final_sampled_tiles (pandas.DataFrame): DataFrame containing sampled tile information.
        wsi_path (str): Directory path containing whole slide images (WSIs).
        output_dir (str): Directory to save the generated tile images.
        window_size (int): Size (in pixels) to which each tile will be resized.
        normalize (str, optional): Normalization vector for stain normalization (57,22,-8,20,10,5).
        tcga_flag (bool, optional): Flag for TCGA-related processing (default is True).

    Returns:
        None: Saves processed tiles to the specified `output_dir` as JPEG images.
    """
    wsi_list = glob(os.path.join(wsi_path, "*.*"))
    
    for slide_name in tqdm(final_sampled_tiles["slide"].unique(), desc="Processing slides"):
        # Extract resolution information for the slide
        slide_path = next((i for i in wsi_list if os.path.splitext(os.path.basename(i))[0] == str(slide_name)), None)
        if not slide_path:
            continue
        
        try:
            slide = openslide.OpenSlide(slide_path)
            OrgPixelSizeX = float(slide.properties["openslide.mpp-x"])

            window_size_40x = convert_mags(window_size, OrgPixelSizeX)

            final_sampled_tiles_sub = final_sampled_tiles[final_sampled_tiles["slide"] == str(slide_name)]
            for _, row in final_sampled_tiles_sub.iterrows():
                try:
                    x, y = convert_mags(row["Centroid_x"], OrgPixelSizeX), convert_mags(row["Centroid_y"], OrgPixelSizeX)
                    tile = slide.read_region((int(x - window_size_40x / 2), int(y - window_size_40x / 2)), 0, (window_size_40x, window_size_40x))
                    tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
                    tile = cv2.resize(tile, (window_size, window_size))
                    tile = Image.fromarray(normalize_tile(tile, normalize).astype('uint8'), 'RGB')

                    tile_outpath = os.path.join(output_dir, f"{row['subset']}_{str(row['slide'])}_{row['CellID']}")
                    if not os.path.isfile(tile_outpath):
                        tile.save(tile_outpath)
                except Exception as e:
                    print(f"Error processing tile {row['CellID']} in {str(slide_name)}: {e}")
        except:
            try:
                # TIFF file from Qupath
                slide = cv2.imread(slide_path)
                OrgPixelSizeX = 0.2525
                
                window_size_40x = convert_mags(window_size, OrgPixelSizeX)

                final_sampled_tiles_sub = final_sampled_tiles[final_sampled_tiles["slide"] == str(slide_name)]
                for _, row in final_sampled_tiles_sub.iterrows():
                    try:
                        x, y = convert_mags(row["Centroid_x"], OrgPixelSizeX), convert_mags(row["Centroid_y"], OrgPixelSizeX)
                        xmin = int(x - window_size_40x / 2)
                        ymin = int(y - window_size_40x / 2)
                        tile = slide[xmin:int(xmin+window_size_40x), ymin:ymin+window_size_40x,...]
                        tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
                        tile = cv2.resize(tile, (window_size, window_size))
                        tile = Image.fromarray(normalize_tile(tile, normalize).astype('uint8'), 'RGB')

                        tile_outpath = os.path.join(output_dir, f"{row['subset']}_{str(row['slide'])}_{row['CellID']}")
                        if not os.path.isfile(tile_outpath):
                            tile.save(tile_outpath)
                    except Exception as e:
                        print(f"Error processing tile {row['CellID']} in {str(slide_name)}: {e}")               
            except Exception as e:
                print(f'Error reading slide {str(slide_name)}: {e}')

def main():
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    csv_paths = glob(os.path.join(args.results_dir, "*_files", f"cell_summary_{str(args.window_size)}px", "*.csv"))

    all_cells_df = []

    for path in tqdm(csv_paths, desc="Loading cell data"):
        df = pd.read_csv(path)
        if args.tcga_flag:
            df["patient"] = os.path.basename(path)[:12]
            df["slide"] = os.path.basename(path)[:-4]
        else:
            df["patient"] = "_".join(os.path.splitext(os.path.basename(path))[0].split("_")[:2])
            df["slide"] = os.path.splitext(os.path.basename(path))[0]

        for cell_type in range(1, 6):
            subset_df = df.loc[df.CellType == cell_type]
            if len(all_cells_df) < cell_type:
                all_cells_df.append(subset_df)
            else:
                all_cells_df[cell_type - 1] = pd.concat([all_cells_df[cell_type - 1], subset_df], ignore_index=True)

    samples_from_csv = [os.path.basename(path)[:12] if args.tcga_flag else "_".join(os.path.splitext(os.path.basename(path))[0].split("_")[:2]) for path in csv_paths]
    random.Random(4).shuffle(samples_from_csv)

    if args.perc_train < 1:
        train_samples = samples_from_csv[:int(args.perc_train * len(samples_from_csv))]
        valid_samples = samples_from_csv[int(args.perc_train * len(samples_from_csv)):]

        tiles_to_remove = pd.DataFrame()
        train_sampled_tiles = sample_tiles_subset("train", train_samples, all_cells_df, args.nb_tiles_train, tiles_to_remove)
        valid_sampled_tiles = sample_tiles_subset("valid", valid_samples, all_cells_df, args.nb_tiles_valid, tiles_to_remove)
        tiles_to_remove = pd.concat([train_sampled_tiles, valid_sampled_tiles])
        test_sampled_tiles = sample_tiles_subset("test", samples_from_csv, all_cells_df, args.nb_tiles_test, tiles_to_remove)
        final_sampled_tiles = pd.concat([tiles_to_remove, test_sampled_tiles])
    else:
        final_sampled_tiles = pd.concat(all_cells_df).sample(frac=1, random_state=42)
        final_sampled_tiles["subset"] = "train"

    final_sampled_tiles.to_csv(os.path.join(args.results_dir, "tiles_used_for_ssl_information.csv"), index=False)
    process_tiles(final_sampled_tiles, args.wsi_path, args.output_dir, args.window_size, 
                  tcga_flag = args.tcga_flag)


if __name__ == "__main__":
    main()
