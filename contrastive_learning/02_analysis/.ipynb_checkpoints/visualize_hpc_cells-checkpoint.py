from glob import glob
import os
import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw
import argparse

parser = argparse.ArgumentParser(description='Visualize top HPC slides with Hover-Net overlay.')
parser.add_argument('--output_dir', dest='output_dir', type = str, default = None, help = "Output directory.", required=True)
parser.add_argument('--hpc', dest='hpc', type = int, default = None, help = "HPC to sample from.", required=True)
parser.add_argument('--cluster_csv', dest='cluster_csv', type = str, default = None, help = "CSV with all cluster assignments of cohort.", required=True)
parser.add_argument('--nb_samples', dest='nb_samples', type = int, default = 1, help = "Number of slides to visualize.")
parser.add_argument('--wsi_paths', dest='wsi_paths', nargs='+', help="Path(s) to the wsi images.", required=True)
parser.add_argument('--hover_net_path', dest='hover_net_path', type = str, default = None, help = "Path to Hover-Net results.", required=True)
parser.add_argument('--window_size', dest='window_size', type=int, default=224, 
                        help="Size of tiles (in pixels) at 20x magnification.")
parser.add_argument('--immune_flag', action='store_true', help = "Flag to add immune cells overlay.")


def convert_mags_from_20x(coords_20x, target_pixel_size):
    """
    Converts coordinates from 20x magnification to a desired magnification.

    Args:
        coords_20x (float): Coordinate at 20x magnification.
        target_pixel_size (float): Desired target pixel size level.

    Returns:
        int: Converted coordinate for the specified magnification.
    """
    final_coords = coords_20x * (447 / 224 * 0.2525) / target_pixel_size
    return int(final_coords)


def load_cluster_data(cluster_csv):
    """
    Loads cluster assignments from the main CSV file and its test/validation splits.

    Args:
        cluster_csv (str): Path to the main cluster assignment CSV.

    Returns:
        pd.DataFrame: Combined DataFrame of cluster assignments.
    """
    cluster_df = pd.read_csv(cluster_csv)
    cluster_df['slides'] = cluster_df['slides'].astype(str)

    for suffix in ["_test.csv", "_valid.csv"]:
        additional_csv = cluster_csv.replace(".csv", suffix)
        if os.path.exists(additional_csv):
            additional_df = pd.read_csv(additional_csv)
            cluster_df = pd.concat([cluster_df, additional_df], axis=0, ignore_index=True)

    return cluster_df


def compute_tile_composition(cluster_df, leiden_col, hpc):
    """
    Computes the composition of tiles for each slide and ranks slides by HPC of interest.

    Args:
        cluster_df (pd.DataFrame): Cluster assignments DataFrame.
        leiden_col (str): Column name containing cluster assignments.
        hpc (int): HPC of interest.

    Returns:
        pd.DataFrame: Ranked slides based on HPC composition.
        pd.DataFrame: Filtered DataFrame of tiles for the HPC of interest.
    """
    tile_counts = cluster_df.groupby(['slides', leiden_col]).size().reset_index(name='tile_count')
    total_tiles = cluster_df.groupby('slides').size().reset_index(name='total_count')
    tile_composition_vectors = pd.merge(tile_counts, total_tiles, on='slides')
    tile_composition_vectors['hpc_perc'] = (tile_composition_vectors['tile_count'] / tile_composition_vectors['total_count']) * 100

    tile_composition_hpc = tile_composition_vectors[tile_composition_vectors[leiden_col] == hpc]
    tile_composition_hpc = tile_composition_hpc.sort_values(by='hpc_perc', ascending=False)

    hpc_df = cluster_df[cluster_df[leiden_col] == hpc]
    return tile_composition_hpc, hpc_df


def add_tile_overlay(draw, hpc_slide_df, cell_summary_df, slide, downsample_factor, fill_color_hpc, window_size_40x):
    """
    Adds rectangular overlays for tiles belonging to the specified HPC.

    Args:
        draw (PIL.ImageDraw.ImageDraw): Drawing object for overlay.
        hpc_slide_df (pd.DataFrame): DataFrame of HPC tiles for the slide.
        cell_summary_df (pd.DataFrame): Cell summary DataFrame.
        slide (openslide.OpenSlide): OpenSlide object for the slide.
        downsample_factor (float): Downsampling factor from base magnification.
        fill_color_hpc (tuple): RGBA color for HPC tiles.
        window_size_40x (int): Window size at 40x magnification.
    """
    for _, row in hpc_slide_df.iterrows():
        centroid_x = cell_summary_df.loc[cell_summary_df.CellID == row.tiles, 'Centroid_x'].values[0]
        centroid_y = cell_summary_df.loc[cell_summary_df.CellID == row.tiles, 'Centroid_y'].values[0]
        
        x = convert_mags_from_20x(centroid_x, float(slide.properties["openslide.mpp-x"]))
        y = convert_mags_from_20x(centroid_y, float(slide.properties["openslide.mpp-x"]))
        
        scaled_x = int(round((x - window_size_40x / 2) / downsample_factor))
        scaled_y = int(round((y - window_size_40x / 2) / downsample_factor))
        scaled_window_size = int(round(window_size_40x / downsample_factor))

        draw.rectangle([scaled_x, scaled_y, scaled_x + scaled_window_size, scaled_y + scaled_window_size], fill=fill_color_hpc,
                      outline=(186, 142, 35), width = 2)


def add_centroid_overlay(draw, cell_summary_df, cell_type, slide, downsample_factor, dot_radius, fill_color):
    """
    Adds circular overlays for cell centroids of a specific type.

    Args:
        draw (PIL.ImageDraw.ImageDraw): Drawing object for overlay.
        cell_summary_df (pd.DataFrame): Cell summary DataFrame.
        cell_type (int): Cell type identifier.
        slide (openslide.OpenSlide): OpenSlide object for the slide.
        downsample_factor (float): Downsampling factor from base magnification.
        dot_radius (int): Radius of the circle for centroids.
        fill_color (tuple): RGBA color for the centroids.
    """
    cell_summary_filtered = cell_summary_df[cell_summary_df["CellType"] == cell_type]
    for _, row in cell_summary_filtered.iterrows():
        cx = convert_mags_from_20x(row.Centroid_x, float(slide.properties["openslide.mpp-x"]))
        cy = convert_mags_from_20x(row.Centroid_y, float(slide.properties["openslide.mpp-x"]))
        
        scaled_cx = int(round(cx / downsample_factor))
        scaled_cy = int(round(cy / downsample_factor))
        
        draw.ellipse(
            [scaled_cx - dot_radius, scaled_cy - dot_radius, scaled_cx + dot_radius, scaled_cy + dot_radius],
            fill=fill_color
        )


def main():
    """
    Main function to visualize top HPC slides with Hover-Net overlay.
    """
    # Parse arguments
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Load cluster data
    cluster_df = load_cluster_data(args.cluster_csv)
    leiden_col = cluster_df.columns[cluster_df.columns.str.contains("leiden")].values[0]

    # Compute tile composition
    tile_composition_hpc, hpc_df = compute_tile_composition(cluster_df, leiden_col, args.hpc)
    # Select the top slides by HPC percentage
    top_slides = tile_composition_hpc['slides'][:args.nb_samples]

    # Create output directory for top slides
    output_dir_sub = os.path.join(args.output_dir, "top_slides")
    os.makedirs(output_dir_sub, exist_ok=True)
    
    target_magnification = 2.5
    dot_radius = 2
    fill_color_hpc =  (255, 238, 140) # Color for HPC tiles
    fill_color_tumor = (30,144,255)  # Blue for tumor cells
    fill_color_immune = (255,0,0)  # Red for immune cells
    
    # Gather .svs files from the provided paths
    if len(args.wsi_paths) == 1:
        svs_paths = glob(os.path.join(args.wsi_paths[0], "*", "*.svs"))
    else:
        svs_paths = []
        for path in args.wsi_paths:
            svs_paths.extend(glob(os.path.join(path, "*", "*.svs")))

    # Process each top slide
    for slide_id in top_slides:
        slide_hpc_df = hpc_df[hpc_df['slides'] == slide_id]

        # Find matching .svs file
        slide_path = [i for i in svs_paths if os.path.basename(i).split(".")[0] == str(slide_id)][0]
        if slide_path is None:
            print(f"No .svs file found for slide {slide_id}")
            continue

        # Open slide and retrieve properties
        slide = openslide.OpenSlide(slide_path)
        base_magnification = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        downsample_factor =  base_magnification / target_magnification
        
        # Get WSI dimensions at the base level
        wsi_width, wsi_height = slide.dimensions

        # Calculate dimensions for downsampled WSI
        new_width = int(wsi_width / downsample_factor)
        new_height = int(wsi_height / downsample_factor)

        # Read region from the WSI at a lower magnification
        downsampled_wsi = slide.get_thumbnail((new_width, new_height))

        # Prepare a blank image for overlay
        overlay = Image.new("RGBA", downsampled_wsi.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        
        # Load HoverNet cell summary if applicable
        if "TCGA" in str(slide_id):
            cell_summary_path = glob(os.path.join(args.hover_net_path, str(slide_id) + ".*_files", f"cell_summary_{str(args.window_size)}px", "*.csv"))[0]
        else:
            cell_summary_path = glob(os.path.join(args.hover_net_path, str(slide_id) + "*_files", f"cell_summary_{str(args.window_size)}px", "*.csv"))[0]

        cell_summary_df = pd.read_csv(cell_summary_path)
        
        # Convert 20x window size of 224 to 40x
        window_size_40x = convert_mags_from_20x(args.window_size, float(slide.properties["openslide.mpp-x"]))
            
        # Add HPC tile overlays
        add_tile_overlay(draw, slide_hpc_df, cell_summary_df, slide, downsample_factor, fill_color_hpc, window_size_40x)

        # Add tumor centroid overlays 
        add_centroid_overlay(draw, cell_summary_df, cell_type=1, slide=slide, downsample_factor=downsample_factor, dot_radius=dot_radius, fill_color=fill_color_tumor)

        # Add immune centroid overlays if immune_flag is set
        if args.immune_flag:
            add_centroid_overlay(draw, cell_summary_df, cell_type=2, slide=slide, downsample_factor=downsample_factor, dot_radius=dot_radius, fill_color=fill_color_immune)
            
        # Blend the overlay with the downsampled more transparent WSI
        downsampled_wsi = downsampled_wsi.convert("RGBA")
        white_background = Image.new("RGBA", downsampled_wsi.size, (255, 255, 255, 255))
        softened_downsampled_wsi = Image.blend(white_background, downsampled_wsi, alpha=0.3)
        downsampled_wsi_overlay= Image.alpha_composite(softened_downsampled_wsi, overlay)

        # Save the overlay image
        downsampled_wsi_overlay.save(os.path.join(output_dir_sub, f"{slide_id}_hpc{str(args.hpc)}.png"))

        print(f"Processed slide {slide_id} with cn-HPC {str(args.hpc)} overlay saved.")

if __name__ == "__main__":
    main()
