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

##### Main #######
parser = argparse.ArgumentParser(description='Sample tiles from each cn-HPC.')
parser.add_argument('--output_dir', dest='output_dir', type = str, default = None, help = "Output directory.")
parser.add_argument('--window_size', dest='window_size', type = int, default = 224, help = "Size of tiles at 20x.")
parser.add_argument('--cluster_csv', dest='cluster_csv', type = str, default = None, help = "CSV with all cluster assignments of cohort.")
parser.add_argument('--nb_samples', dest='nb_samples', type = int, default = 96, help = "Number of tiles to sample per cluster.")
parser.add_argument('--svs_path', dest='svs_path', type = str, default = None, help = "Path to the svs images.")
parser.add_argument('--hovernet_dir', dest='hovernet_dir', type = str, default = None, help = "Directory containing the hovernet results.")

args					= parser.parse_args()
output_dir				= args.output_dir
window_size				= args.window_size
cluster_csv				= args.cluster_csv
nb_samples				= args.nb_samples
svs_path				= args.svs_path
hovernet_dir			= args.hovernet_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
cluster_df = pd.read_csv(cluster_csv)

def convert_mags_from_20x(coords_20x, final_mag):
    ''' This function converts coordinates at 20x to 40x (original magnification of the whole slide images).'''
    final_coords = coords_20x*(447/224*0.2525)/final_mag
    return int(final_coords)

def RGB_to_lab(tile):
    Lab = color.rgb2lab(tile)
    return Lab

def Lab_to_RGB(Lab):
    newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
    return newtile

# Get the cluster column
leiden_col = cluster_df.columns[cluster_df.columns.str.contains("leiden")].values[0]

grouped = cluster_df.groupby(leiden_col)

# Sample tiles for each cluster
for l, group in grouped:
    fig, axes = plt.subplots(8, 12, figsize=(40, 30))
    axes = axes.flatten()
    # Randomly select tiles tiles from this cluster
    sampled_group = group.sample(n=nb_samples, random_state=0)
    sampled_group = sampled_group.sample(frac=1, random_state=0).reset_index(drop=True)
    sampled_group.reset_index(drop=True, inplace=True)
    for i, row in sampled_group.iterrows():
        slide_path = glob(os.path.join(svs_path,"*",row["slides"] + ".*.svs"))
        if len(slide_path)>0:
            slide_path = slide_path[0]
        else:
            slide_path = glob(os.path.join(svs_path,"*","*",row["slides"] + ".svs"))
            slide_path = slide_path[0]
        slide = openslide.OpenSlide(slide_path)
        cell_summary_path = glob(os.path.join(hovernet_dir, row["slides"] + ".*_files", "cell_summary","*.csv"))[0]
        cell_summary_df = pd.read_csv(cell_summary_path)
        centroid_x = cell_summary_df.Centroid_x[cell_summary_df.CellID == row.tiles].values[0]
        centroid_y = cell_summary_df.Centroid_y[cell_summary_df.CellID == row.tiles].values[0]
        x = convert_mags_from_20x(centroid_x, float(slide.properties["openslide.mpp-x"]))
        y = convert_mags_from_20x(centroid_y, float(slide.properties["openslide.mpp-x"]))

        window_size_40x = convert_mags_from_20x(window_size, float(slide.properties["openslide.mpp-x"]))
        tile = slide.read_region((int(x-window_size_40x/2),int(y-window_size_40x/2)),0,(window_size_large,window_size_large))
        tile = cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2RGB)
        tile = Image.fromarray(tile.astype('uint8'),'RGB')
        axes[i].imshow(tile)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "samples_cluster" + str(l)+"_tcga.png"))