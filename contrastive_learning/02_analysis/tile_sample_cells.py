from glob import glob
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

def convert_mags_from_20x(coords_20x, target_pixel_size):
    """
    Converts coordinates from 20x magnification to a specified final magnification.

    Args:
        coords_20x (float): Coordinate value at 20x magnification.
        target_pixel_size (float): Target pixel size to convert to.

    Returns:
        int: Converted coordinate value.
    """
    final_coords = coords_20x * (447 / 224 * 0.2525) / target_pixel_size
    return int(final_coords)

def RGB_to_lab(tile):
    """
    Converts an RGB image to the CIE-LAB color space.

    Args:
        tile (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: LAB representation of the input image.
    """
    Lab = color.rgb2lab(tile)
    return Lab

def Lab_to_RGB(Lab):
    """
    Converts an image from the CIE-LAB color space back to RGB.

    Args:
        Lab (numpy.ndarray): Input LAB image.

    Returns:
        numpy.ndarray: RGB representation of the input image.
    """
    newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
    return newtile

def main():
    """
    Main function to sample tiles from each cn-HPC cluster, save as images for visualization.
    """
    parser = argparse.ArgumentParser(description='Sample tiles from each cn-HPC cluster.')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default=None, 
                        help="Directory to save the output images.")
    parser.add_argument('--window_size', dest='window_size', type=int, default=224, 
                        help="Size of tiles (in pixels) at 20x magnification.")
    parser.add_argument('--cluster_csv', dest='cluster_csv', type=str, default=None, 
                        help="CSV file containing all cluster assignments for the cohort.")
    parser.add_argument('--nb_samples', dest='nb_samples', type=int, default=96, 
                        help="Number of tiles to sample per cluster.")
    parser.add_argument('--wsi_path', dest='wsi_path', type=str, default=None, 
                        help="Path to the directory containing SVS whole slide images.")
    parser.add_argument('--hovernet_dir', dest='hovernet_dir', type=str, default=None, 
                        help="Directory containing HoverNet results.")
    parser.add_argument('--tcga_flag', action='store_true', 
                        help="Flag to run pipeline for TCGA images.")
    parser.add_argument('--htan_flag', action='store_true', 
                        help="Flag to run pipeline for HTAN images.")
    args = parser.parse_args()

    output_dir = args.output_dir
    window_size = args.window_size
    cluster_csv = args.cluster_csv
    nb_samples = args.nb_samples
    wsi_path = args.wsi_path
    hovernet_dir = args.hovernet_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cluster_df = pd.read_csv(cluster_csv)
    leiden_col = cluster_df.columns[cluster_df.columns.str.contains("leiden")].values[0]

    grouped = cluster_df.groupby(leiden_col)

    for l, group in grouped:
        fig, axes = plt.subplots(8, 12, figsize=(40, 30))
        axes = axes.flatten()
        sampled_group = group.sample(n=nb_samples, random_state=0).reset_index(drop=True)
        for i, row in sampled_group.iterrows():
            slide_path = glob(os.path.join(wsi_path, row["slides"] + ".*.svs"))
            slide_path = slide_path[0]

            slide = openslide.OpenSlide(slide_path)
            
            cell_summary_path = glob(os.path.join(hovernet_dir, row["slides"] + ".*_files", "cell_summary", "*.csv"))[0]
            cell_summary_df = pd.read_csv(cell_summary_path)

            centroid_x = cell_summary_df.Centroid_x[cell_summary_df.CellID == row.tiles].values[0]
            centroid_y = cell_summary_df.Centroid_y[cell_summary_df.CellID == row.tiles].values[0]
            
            if tcga_flag:
                OrgPixelSizeX = float(slide.properties["openslide.mpp-x"])
            else if htan_flag:
                OrgPixelSizeX = float(slide.properties["tiff.XResolution"])
            else:
                raise ValueError("No valid pixel resolution found.")
            
            window_size_40x = convert_mags(window_size, OrgPixelSizeX)
            
            x = convert_mags_from_20x(centroid_x, OrgPixelSizeX)
            y = convert_mags_from_20x(centroid_y, OrgPixelSizeX)

            window_size_40x = convert_mags_from_20x(window_size, OrgPixelSizeX)
            tile = slide.read_region((int(x - window_size_40x / 2), int(y - window_size_40x / 2)), 0, 
                                     (window_size_40x, window_size_40x))
            tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
            tile = Image.fromarray(tile.astype('uint8'), 'RGB')

            axes[i].imshow(tile)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"samples_cluster{l}.png"))
        plt.close()

if __name__ == "__main__":
    main()