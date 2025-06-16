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
    parser = argparse.ArgumentParser(description='Tile per CSV and process all into training set.')
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing Hover-Net CSV outputs.")
    parser.add_argument('--wsi_path', type=str, required=True, help="Folder with whole slide images.")
    parser.add_argument('--window_size', type=int, default=224, help="Size of tiles at 20x.")
    parser.add_argument('--max_tiles_per_slide', type=int, default=20000, help="Number of tiles to create for each slide.")
    parser.add_argument('--tcga_flag', action='store_true', help="Flag for TCGA image naming.")
    return parser.parse_args()



def convert_mags(coords_20x, final_mag):
    return int(coords_20x * (447 / 224 * 0.2525) / final_mag)


def normalize_tile(tile, norm_vec='57,22,-8,20,10,5'):
    norm_vec = [float(x) for x in norm_vec.split(',')]
    lab = color.rgb2lab(tile)
    tile_mean = [np.mean(lab[:, :, i]) for i in range(3)]
    tile_std = [np.std(lab[:, :, i]) for i in range(3)]
    new_mean, new_std = norm_vec[:3], norm_vec[3:6]

    for i in range(3):
        lab[:, :, i] = ((lab[:, :, i] - tile_mean[i]) * (new_std[i] / tile_std[i])) + new_mean[i]
        lab[:, :, i] = np.clip(lab[:, :, i], 0 if i == 0 else -128, 100 if i == 0 else 127)

    return (color.lab2rgb(lab) * 255).astype(np.uint8)


def process_tiles_for_csv(df, wsi_path, output_dir, window_size, tcga_flag=True):
    wsi_list = glob(os.path.join(wsi_path, "*.*"))
    slide_name = df["slide"].iloc[0]
    slide_path = next((i for i in wsi_list if os.path.splitext(os.path.basename(i))[0] == str(slide_name)), None)
    if not slide_path:
        return

    try:
        slide = openslide.OpenSlide(slide_path)
        OrgPixelSizeX = float(slide.properties["openslide.mpp-x"])
    except:
        slide = cv2.imread(slide_path)
        OrgPixelSizeX = 0.2525

    window_size_40x = convert_mags(window_size, OrgPixelSizeX)

    for _, row in df.iterrows():
        try:
            x, y = convert_mags(row["Centroid_x"], OrgPixelSizeX), convert_mags(row["Centroid_y"], OrgPixelSizeX)
            if isinstance(slide, openslide.OpenSlide):
                tile = slide.read_region((int(x - window_size_40x / 2), int(y - window_size_40x / 2)), 0, (window_size_40x, window_size_40x))
                tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
            else:
                tile = slide[int(x - window_size_40x / 2):int(x + window_size_40x / 2),
                             int(y - window_size_40x / 2):int(y + window_size_40x / 2), :]
            tile = cv2.resize(tile, (window_size, window_size))
            tile = Image.fromarray(normalize_tile(tile).astype('uint8'), 'RGB')

            outname = f"train_{row['slide']}_{row['CellID']}"
            tile_outpath = os.path.join(output_dir, outname)
            tile.save(tile_outpath)
        except Exception as e:
            print(f"Error processing tile {row['CellID']} in {str(slide_name)}: {e}")


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = glob(os.path.join(args.results_dir, "*_files", f"cell_summary_{args.window_size}px", "*.csv"))

    for path in tqdm(csv_paths, desc="Processing CSVs"):
        df = pd.read_csv(path)
        if args.tcga_flag:
            df["patient"] = os.path.basename(path)[:12]
            df["slide"] = os.path.basename(path)[:-4]
        else:
            df["patient"] = "_".join(os.path.splitext(os.path.basename(path))[0].split("_")[:2])
            df["slide"] = os.path.splitext(os.path.basename(path))[0]

        df = df[df["CellType"].isin([1, 2, 3, 4, 5])]
        df["subset"] = "train"
        df = df.sample(n=args.max_tiles_per_slide, random_state=42)
        
        process_tiles_for_csv(df, args.wsi_path, args.output_dir, args.window_size, tcga_flag=args.tcga_flag)



if __name__ == "__main__":
    main()
