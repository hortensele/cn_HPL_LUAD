import os
import numpy as np
import pandas as pd
import argparse
from glob import glob
import json
import cv2
from joblib import Parallel, delayed

def process_json(jason_path):
    """This function takes as input the path to a Hover-Net json file. For each detected cell, this summarizes cell properties, cell class, and centroid coordinates with respect to the original 20x whole slide image."""
    cell_summary_data = []
    with open(jason_path) as f:
        data = json.load(f)
        for kk in data["nuc"].keys():
            tile_x = os.path.basename(jason_path).replace(".json", "").split("_")[0]
            tile_y = os.path.basename(jason_path).replace(".json", "").split("_")[1]
            cnt = np.asarray(data["nuc"][kk]["contour"])
            type_prob = data["nuc"][kk]["type_prob"]
            celltype = data["nuc"][kk]["type"]
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x = M['mu20'] + M['mu02']
            y = 4 * M['mu11'] ** 2 + (M['mu20'] - M['mu02']) ** 2
            try:
                elongation = (x + y ** 0.5) / (x - y ** 0.5)
            except:
                elongation = np.nan
            rect = cv2.minAreaRect(cnt)
            MinDiaR = min(rect[1])
            MaxDiaR = max(rect[1])
            rect_area2 = MinDiaR * MaxDiaR
            extent2 = float(area) / rect_area2
            Aspect_Ratio = MinDiaR / MaxDiaR
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                MinAx = min(ellipse[1])
                MaxAx = max(ellipse[1])
            else:
                MinAx = np.sqrt(area / np.pi)
                MaxAx = np.sqrt(area / np.pi)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area

            # Append the results to the list
            cell_summary_data.append({
                'CellID': os.path.basename(jason_path).replace(".json", "_" + kk + ".jpeg"),
                'Centroid_x': data["nuc"][kk]["centroid"][0] + 224 * int(tile_x),
                'Centroid_y': data["nuc"][kk]["centroid"][1] + 224 * int(tile_y),
                'CellType': celltype,
                "Area": area,
                "Perimeter": perimeter,
                "Circularity": circularity,
                "Elongation": elongation,
                "MinDiaR": MinDiaR,
                "MaxDiaR": MaxDiaR,
                "Rec_Area2": rect_area2,
                "Extent": extent2,
                "AspectRatio": Aspect_Ratio,
                "MinAx": MinAx,
                "MaxAx": MaxAx,
                "HullArea": hull_area,
                "Solidity": solidity,
                "Type_prob": type_prob,
            })
    return cell_summary_data

# Function to process all JSON files in parallel
def process_all_jsons_parallel(json_files, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(process_json)(jason_path) for jason_path in json_files)
    # Flatten the list of lists into a single list of dictionaries
    flat_results = [item for sublist in results for item in sublist]
    return pd.DataFrame(flat_results)

##### Main #######
parser = argparse.ArgumentParser(description='Create cell compositions of each tile (224px) at 20x. All coordinates are at the 20x scale.')
parser.add_argument('--result_path', dest='result_path', type=str, default=None, help='Path where Hover-Net results are stored.')
parser.add_argument('--window_size', dest='window_size', type=int, default=None, help="Size of tiles to be covered.")

args = parser.parse_args()
result_path = args.result_path
window_size = args.window_size

# Output directory and file
output_dir = os.path.join(result_path, f"cell_summary_{str(window_size)}px")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

slide_name = os.path.basename(result_path).split("_")[0]
outfile = os.path.join(output_dir, slide_name + ".csv")

# Check if file exists to avoid reprocessing
if not os.path.exists(outfile):
    run_flag = True
else:
    df = pd.read_csv(outfile)
    if df.empty:
        run_flag = True
    else:
        run_flag = False

if run_flag:
    # Get the list of Hover-Net results files (.json)
    json_files = glob(os.path.join(result_path, "json", "*.json"))
    print(f"Found {len(json_files)} JSON files to process.")

    # Process all files in parallel
    cell_summary_df = process_all_jsons_parallel(json_files)
    
    # Cell types from PanNuke dataset
    celltypes = pd.DataFrame({
        "types": [0, 1, 2, 3, 4, 5],
        "labels": ["others", "neoplastic", "inflammatory", "connective", "necrosis", "non_neoplastic"]
    })

    # Get information about the cells detected in each of the final tiles
    centroid_df_dict = {}
    for w in window_size:
        centroid_df_dict[str(w) + "px"] = cell_summary_df.copy()

        for l in celltypes.labels:
            centroid_df_dict[str(w) + "px"][l] = 0
            for attr in ["area", "perimeter", "circularity", "elongation", "MinDiaR", "MaxDiaR", "Rec_Area2", "AspectRatio", "MinAx", "MaxAx", "HullArea", "Solidity", "Extent"]:
                centroid_df_dict[str(w) + "px"]["mean_" + attr + "_" + l] = 0

    # Iterate over each cell and calculate tile-wise summary statistics
    for index, row in cell_summary_df.iterrows():
        print(f"Processing cell {index + 1}/{len(cell_summary_df)}")

        for w in window_size:
            tile_minx = row["Centroid_x"] - w / 2
            tile_maxx = row["Centroid_x"] + w / 2
            tile_miny = row["Centroid_y"] - w / 2
            tile_maxy = row["Centroid_y"] + w / 2

            centroid_df_tile = cell_summary_df.loc[
                (cell_summary_df.Centroid_x >= tile_minx) &
                (cell_summary_df.Centroid_x < tile_maxx) &
                (cell_summary_df.Centroid_y >= tile_miny) &
                (cell_summary_df.Centroid_y < tile_maxy), 
            ]

            for l in celltypes.labels:
                cell_class = celltypes.types[celltypes.labels == l].values[0]
                filtered_cells = centroid_df_tile[centroid_df_tile.CellType == cell_class]
                centroid_df_dict[str(w) + "px"].loc[index, l] = len(filtered_cells)

                for attr in ["Area", "Perimeter", "Circularity", "Elongation", "MinDiaR", "MaxDiaR", "Rec_Area2", "AspectRatio", "MinAx", "MaxAx", "HullArea", "Solidity", "Extent"]:
                    centroid_df_dict[str(w) + "px"].loc[index, "mean_" + attr.lower() + "_" + l] = filtered_cells[attr].mean()

        # Save the dataframe as a CSV
        print("Saving the CSV with summary of tile composition")
        df_to_csv = centroid_df_dict[str(w) + "px"]
        df_to_csv.to_csv(outfile, index=False)
        print("Done!")
