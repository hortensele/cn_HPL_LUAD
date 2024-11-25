import os
import numpy as np
import pandas as pd
import argparse
from glob import glob
import json
import cv2
from joblib import Parallel, delayed

def process_json(json_path):
    """Processes a Hover-Net JSON file and summarizes cell properties and centroid coordinates."""
    cell_summary_data = []
    with open(json_path) as f:
        data = json.load(f)
        tile_x, tile_y = map(int, os.path.basename(json_path).replace(".json", "").split("_")[:2])

        for cell_id, cell_data in data["nuc"].items():
            cnt = np.asarray(cell_data["contour"])
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter else np.nan

            x = M['mu20'] + M['mu02']
            y = 4 * (M['mu11'] ** 2) + (M['mu20'] - M['mu02']) ** 2
            elongation = ((x + y ** 0.5) / (x - y ** 0.5)) if x > y ** 0.5 else np.nan

            rect = cv2.minAreaRect(cnt)
            min_dia, max_dia = sorted(rect[1])
            rect_area = min_dia * max_dia
            extent = area / rect_area if rect_area else np.nan
            aspect_ratio = min_dia / max_dia if max_dia else np.nan

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                min_ax, max_ax = sorted(ellipse[1])
            else:
                min_ax = max_ax = np.sqrt(area / np.pi)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area else np.nan

            cell_summary_data.append({
                'CellID': f"{os.path.basename(json_path).replace('.json', '')}_{cell_id}.jpeg",
                'Centroid_x': cell_data["centroid"][0] + 224 * tile_x,
                'Centroid_y': cell_data["centroid"][1] + 224 * tile_y,
                'CellType': cell_data["type"],
                "Area": area,
                "Perimeter": perimeter,
                "Circularity": circularity,
                "Elongation": elongation,
                "MinDiaR": min_dia,
                "MaxDiaR": max_dia,
                "RectArea": rect_area,
                "Extent": extent,
                "AspectRatio": aspect_ratio,
                "MinAx": min_ax,
                "MaxAx": max_ax,
                "HullArea": hull_area,
                "Solidity": solidity,
                "TypeProb": cell_data["type_prob"],
            })
    return cell_summary_data

def process_all_jsons_parallel(json_files, n_jobs=-1):
    """Processes all JSON files in parallel."""
    results = Parallel(n_jobs=n_jobs)(delayed(process_json)(path) for path in json_files)
    return pd.DataFrame([item for sublist in results for item in sublist])

def main():
    parser = argparse.ArgumentParser(description='Generate cell compositions for tiles at 20x magnification.')
    parser.add_argument('--result_path', type=str, required=True, help='Path to Hover-Net results with jsons.')
    parser.add_argument('--window_size', type=int, required=True, help='Tile size in pixels at 20x.')

    args = parser.parse_args()
    result_path, window_size = args.result_path, args.window_size

    output_dir = os.path.join(result_path, f"cell_summary_{window_size}px")
    os.makedirs(output_dir, exist_ok=True)

    slide_name = os.path.basename(result_path).replace("_files", "")
    outfile = os.path.join(output_dir, f"{slide_name}.csv")

    if os.path.exists(outfile) and not pd.read_csv(outfile).empty:
        print(f"Output file already exists: {outfile}")
        return

    json_files = glob(os.path.join(result_path, "json", "*.json"))
    print(f"Found {len(json_files)} JSON files to process.")

    cell_summary_df = process_all_jsons_parallel(json_files)

    celltypes = pd.DataFrame({
        "types": [0, 1, 2, 3, 4, 5],
        "labels": ["others", "neoplastic", "inflammatory", "connective", "necrosis", "non_neoplastic"]
    })

    print(f"Processing tiles with window size: {window_size}px")
    centroid_df = cell_summary_df.copy()
    for label in celltypes.labels:
        centroid_df[label] = 0
        for attr in ["Area", "Perimeter", "Circularity", "Elongation", "MinDiaR", "MaxDiaR", "RectArea", "AspectRatio", "MinAx", "MaxAx", "HullArea", "Solidity", "Extent"]:
            centroid_df[f"mean_{attr}_{label}"] = 0

    for index, row in cell_summary_df.iterrows():
        tile_minx, tile_maxx = row["Centroid_x"] - window_size / 2, row["Centroid_x"] + window_size / 2
        tile_miny, tile_maxy = row["Centroid_y"] - window_size / 2, row["Centroid_y"] + window_size / 2

        tile_cells = cell_summary_df[
            (cell_summary_df.Centroid_x.between(tile_minx, tile_maxx)) &
            (cell_summary_df.Centroid_y.between(tile_miny, tile_maxy))
        ]

        for label in celltypes.labels:
            cell_class = celltypes.loc[celltypes.labels == label, "types"].iloc[0]
            filtered_cells = tile_cells[tile_cells.CellType == cell_class]
            centroid_df.at[index, label] = len(filtered_cells)

            for attr in ["Area", "Perimeter", "Circularity", "Elongation", "MinDiaR", "MaxDiaR", "RectArea", "AspectRatio", "MinAx", "MaxAx", "HullArea", "Solidity", "Extent"]:
                centroid_df.at[index, f"mean_{attr.lower()}_{label}"] = filtered_cells[attr].mean()

    print(f"Saving results to {outfile}")
    centroid_df.to_csv(outfile, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    main()

