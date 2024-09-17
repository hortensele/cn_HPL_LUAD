import pandas as pd
import numpy as np
import argparse


##### Main #######
parser = argparse.ArgumentParser(description='Calculate smallest distance between closest cells for TCGA dataset.')
parser.add_argument('--csv_file',         dest='csv_file',         type=str,            default=None,        help='CSV Path.')

args               = parser.parse_args()
csv_file           = args.csv_file

cell_df = pd.read_csv(csv_file)
if len(cell_df.columns) > 9:
    cell_df = cell_df[["CellID", "Centroid_x", "Centroid_y", "CellType", "BB_xmin", "BB_ymin", "BB_xmax", "BB_ymax", "Type_prob"]]
centroids = list(zip(cell_df["Centroid_x"],cell_df["Centroid_y"]))
cell_df["min_dist"] = 0
for i in np.arange(len(centroids)):
    centroid = centroids[i]
    centroids_sub = centroids.copy()
    centroids_sub.pop(i)
    centroids_sub = np.asarray(centroids_sub)
    deltas = centroids_sub - centroid
    min_distance = np.min(np.einsum('ij,ij->i', deltas, deltas))
    cell_df.loc[i, "min_dist"] = min_distance
cell_df.to_csv(csv_file, index = False)