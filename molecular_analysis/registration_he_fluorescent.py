import cv2
import numpy as np
import pandas as pd
import tifffile
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Align and map cn-HPCs.')
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory.")
    parser.add_argument('--fluo_img_path', type=str, required=True, help="Path to fluorescent tif image.")
    parser.add_argument('--annotations', type=str, required=True, help="Path to excel where all patients IDs are found.")
    parser.add_argument('--rotation_file', type=str, required=True, help="Path to file with rotation information.")
    parser.add_argument('--he_cores_path', type=str, required=True, help="Path to folder with all H&E tif cores.")
    parser.add_argument('--hpc_tiles_csv', type=str, required=True, help="Path to csv with all cn-HPC tile coordinates.")
    parser.add_argument('--window_size', type=int, default=224, help="Window size of the tiles from HPL.")
    return parser.parse_args()

def process_tiles(args):
    roi, mask_polygon, tile_chunk = args
    local_tiles_to_keep = []
    
    for _, row in tile_chunk.iterrows():
        tile_xmin, tile_xmax = row['tile_xmin'], row['tile_xmax']
        tile_ymin, tile_ymax = row['tile_ymin'], row['tile_ymax']
        
        # Check tile boundaries
        if (tile_xmin < 0 or tile_ymin < 0 or 
            tile_xmax > mask_polygon.shape[0] or 
            tile_ymax > mask_polygon.shape[1]):
            continue
        
        # Extract tile region
        tile_region = mask_polygon[tile_xmin:tile_xmax, tile_ymin:tile_ymax]
        if tile_region.size == 0:
            continue
        
        # Calculate intersection ratio
        intersection_area = np.sum(tile_region)
        tile_area = (tile_xmax - tile_xmin) * (tile_ymax - tile_ymin)
        
        if (intersection_area / tile_area) > 0.5:
            local_tiles_to_keep.append(row['CellID'])
    
    return roi, local_tiles_to_keep


def find_bounding_box(mask):
    # Find contours of the non-zero (green) regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the bounding box of the first contour (the circular region)
    x, y, w, h = cv2.boundingRect(contours[0])  # Get bounding box
    return x, y, x + w, y + h

def create_roi_mask(img, roi_color):
    roi_mask = cv2.inRange(img, np.array(roi_color), np.array(roi_color))
    result_roi = cv2.bitwise_and(img, img, mask=roi_mask)
    result_roi_gray = cv2.cvtColor(result_roi, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        result_roi_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,  # Inverse ratio of resolution
        minDist=200,  # Minimum distance between centers
        param1=50,  # Edge detection threshold
        param2=30,  # Threshold for center detection
        minRadius=100,  # Minimum radius
        maxRadius=300  # Maximum radius
    )
    circle = circles[0,:][0]
    cv2.circle(result_roi, (int(circle[0]), int(circle[1])), int(circle[2]), roi_color, -1)
    roi_mask = cv2.inRange(result_roi, np.array(roi_color), np.array(roi_color))
    result_roi = cv2.bitwise_and(result_roi, result_roi, mask=roi_mask)
    return result_roi, roi_mask

def get_final_mask(roi_mask, pank_mask, xmin_crop, ymin_crop, padding, scaling_factor = 4):
    intersection_mask = roi_mask & pank_mask
    # Add padding to get back to the original non-cropped image
    original_x1min = xmin_crop-padding
    original_y1min = ymin_crop-padding
    
    if original_x1min < 0:
        intersection_mask = intersection_mask[-original_x1min:,...]
    else:
        intersection_mask = cv2.copyMakeBorder(intersection_mask, 0, 0, original_x1min, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    if original_y1min < 0:
        intersection_mask = intersection_mask[...,-original_y1min:]
    else:
        intersection_mask = cv2.copyMakeBorder(intersection_mask, original_y1min, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    intersection_mask = cv2.resize(intersection_mask, 
                                   (intersection_mask.shape[0]*scaling_factor, intersection_mask.shape[1]*scaling_factor),
                                   interpolation=cv2.INTER_LINEAR)
    intersection_mask = (intersection_mask > 0).astype(np.uint8)
    return intersection_mask

def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'qc_img'), exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'roi_coords_20x'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'cn_hpc_mapping'), exist_ok=True)

    rotation_df = pd.read_csv(args.rotation_file)
    annots_df = pd.read_excel(args.annotations)
    annots_df['patient'] = annots_df['patient'].astype(str)
    annots_df['patient'] += annots_df['Type'].str[0]

    padding = 500
    patient_id = os.path.splitext(os.path.basename(args.fluo_img_path))[0]
    fluorescent_image = cv2.imread(args.fluo_img_path)
    border_color = (0, 0, 0)
    padded_image = cv2.copyMakeBorder(fluorescent_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=border_color)
    padded_gray_image  = cv2.cvtColor(padded_image,cv2.COLOR_RGB2GRAY)

    fluo_slide = annots_df.loc[annots_df.patient == patient_id, 'Scan_ID'].values[0]

    contours, hierarchy = cv2.findContours(padded_gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 1000000  # Define your area threshold
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    large_contours_areas = [cv2.contourArea(cnt) for cnt in large_contours]
    large_contour = large_contours[np.argmax(large_contours_areas)]
    (center_x, center_y), radius = cv2.minEnclosingCircle(large_contour)
    main_center = (int(center_x), int(center_y))
    main_radius = int(radius)
    
    fluorescent_image_mask = cv2.cvtColor(padded_gray_image, cv2.COLOR_GRAY2BGR)  
    cv2.circle(fluorescent_image_mask, main_center, main_radius, (0, 255, 0), 3) 

    _, binary_image = cv2.threshold(fluorescent_image_mask, 230, 255, cv2.THRESH_BINARY)
    
    gray_binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    _, thresholded_binary_image = cv2.threshold(gray_binary_image, 230, 255, cv2.THRESH_BINARY)
    circles_rois = cv2.HoughCircles(
        thresholded_binary_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=50, param2=30, minRadius=200, maxRadius=600
    )

    hsv_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for the target colors
    lower_color1 = np.array([1, 0, 0])  # Example: Green lower bound
    upper_color1 = np.array([85, 255, 255])  # Example: Green upper bound

    lower_color2 = np.array([140, 50, 50])  # Example: Red lower bound
    upper_color2 = np.array([180, 255, 255])  # Example: Red upper bound
    
    combined_mask = np.zeros_like(padded_image[:, :, 0])  
    
    for circle in circles_rois[0, :]:
        circle_mask = np.zeros_like(padded_image[:, :, 0])
        cv2.circle(circle_mask, (int(circle[0]),int(circle[1])), int(circle[2]), 255, -1)
        
        mask_color1 = cv2.inRange(hsv_image, lower_color1, upper_color1)
        mask_color2 = cv2.inRange(hsv_image, lower_color2, upper_color2)
        
        color_mask = cv2.bitwise_or(mask_color1, mask_color2)
        masked_circle = cv2.bitwise_and(color_mask, circle_mask)
        
        combined_mask = cv2.bitwise_or(combined_mask, masked_circle)
    
    roi_mask = cv2.bitwise_and(padded_image, padded_image, mask=combined_mask)
    cv2.circle(roi_mask, main_center, main_radius, (255, 255, 255), 5) 
    cv2.circle(roi_mask, main_center, 1, (255, 0, 0), -1)
    
    roi_mask_qc = cv2.bitwise_and(padded_image, padded_image, mask=combined_mask)
    
    roi_colors = [[255, 165, 0], [0, 255, 255], [255,255,0]]
    for i in np.arange(len(circles_rois[0, :])):
        x,y,radius = circles_rois[0,i]
        cv2.circle(roi_mask, (int(x),int(y)), int(radius), roi_colors[i], 5)
        cv2.circle(roi_mask_qc, (int(x),int(y)), int(radius), roi_colors[i], 5)
        text_x = int(x) + int(radius) + 10  # Text position to the right of the circle
        text_y = int(y)
        roi_name = f'ROI_{str(i)}'
        cv2.putText(roi_mask_qc, roi_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(roi_colors[i]), 2)
    
    roi_mask_qc = roi_mask_qc[padding:-padding, padding:-padding,...]
    angle = rotation_df.loc[rotation_df.fluo_slide == fluo_slide,'estimated_angle_rotation_of_fluo'].values[0]

    rotation_matrix = cv2.getRotationMatrix2D(main_center, angle, 1.0)
    rotated_image = cv2.warpAffine(roi_mask, rotation_matrix, padded_image.shape[1::-1])
    original_rotated = cv2.warpAffine(padded_image, rotation_matrix, padded_image.shape[1::-1])

    he_image = cv2.imread(os.path.join(args.he_cores_path,patient_id+'.tif'))
    he_image = cv2.resize(he_image, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR)

    padded_he_image = cv2.copyMakeBorder(he_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255,255,255))

    gray_image = cv2.cvtColor(padded_he_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get a binary mask
    _, binary_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
    
    blurred_image = cv2.GaussianBlur(binary_image, (9, 9), 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # Adjust size as needed
    closed_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
    
    # Optionally, apply dilation to reinforce the closed area
    # dilated_image = cv2.dilate(closed_image, kernel, iterations=1)
    # plt.imshow(dilated_image)
    inverted_image = cv2.bitwise_not(closed_image)
    
    
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 5000  # Define your area threshold
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    large_contours_areas = [cv2.contourArea(cnt) for cnt in large_contours]
    large_contour = large_contours[np.argmax(large_contours_areas)]
    (center_x, center_y), radius = cv2.minEnclosingCircle(large_contour)
    main_center_he = (int(center_x), int(center_y))
    main_radius_he = int(radius)
    
    he_image_mask = padded_he_image.copy()
    cv2.circle(he_image_mask, main_center_he, main_radius_he, (0, 255, 0), 3)  # Draw the circle
    cv2.circle(he_image_mask, main_center_he, 1, (0, 0, 255), -1)  # Mark the center

    mask_white_fluo = cv2.inRange(rotated_image, (255, 255, 255), (255, 255, 255))
    mask_green_he = cv2.inRange(he_image_mask, (0, 255, 0), (0, 255, 0))

    x1_min, y1_min, x1_max, y1_max = find_bounding_box(mask_green_he)
    x2_min, y2_min, x2_max, y2_max = find_bounding_box(mask_white_fluo)
    
    cropped_image1 = he_image_mask[y1_min:y1_max, x1_min:x1_max]
    cropped_image2 = rotated_image[y2_min:y2_max, x2_min:x2_max]

    cropped_image2_resized = cv2.resize(cropped_image2, (cropped_image1.shape[1], cropped_image1.shape[0]))

    cropped_image_fluo_original = original_rotated[y2_min:y2_max, x2_min:x2_max]
    cropped_image_fluo_original_resized = cv2.resize(cropped_image_fluo_original, (cropped_image1.shape[1], cropped_image1.shape[0]))
    overlay_image_qc = cv2.addWeighted(cropped_image1, 0.5, cropped_image_fluo_original_resized, 0.5, 0)
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    axes[0].imshow(cropped_image1)
    axes[0].set_title('Cropped H&E core')
    axes[0].axis('off')
    
    axes[1].imshow(cropped_image_fluo_original_resized)
    axes[1].set_title('Cropped fluorescent core')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_image_qc)
    axes[2].set_title('Overlaid cores')
    axes[2].axis('off')
    
    axes[3].imshow(roi_mask_qc)
    axes[3].set_title('Automated ROI labels on \n original fluorescent core')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/qc_img/{patient_id}.png') 

    image_hsv = cv2.cvtColor(cropped_image2_resized, cv2.COLOR_BGR2HSV)
    # Define HSV thresholds for blue and green areas
    lower_blue = np.array([140, 50, 50])    
    upper_blue = np.array([180, 255, 255])

    lower_green = np.array([35, 0, 0])     
    upper_green = np.array([85, 255, 255])
    
    # Create masks for blue and green regions
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    
    result_green = cv2.bitwise_and(cropped_image2_resized, cropped_image2_resized, mask=green_mask)
    result_blue = cv2.bitwise_and(cropped_image2_resized, cropped_image2_resized, mask=blue_mask)
    
    result_roi0, roi0_mask = create_roi_mask(cropped_image2_resized, roi_colors[0])
    result_roi1, roi1_mask = create_roi_mask(cropped_image2_resized, roi_colors[1])
    result_roi2, roi2_mask = create_roi_mask(cropped_image2_resized, roi_colors[2])

    roi_coords = {}

    roi0_pankplus = get_final_mask(roi0_mask, blue_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankplus_roi0'
    roi_coords[fullid] = roi0_pankplus
    
    roi0_pankmin = get_final_mask(roi0_mask, green_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankmin_roi0'
    roi_coords[fullid] = roi0_pankmin
    
    roi1_pankplus = get_final_mask(roi1_mask, blue_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankplus_roi1'
    roi_coords[fullid] = roi1_pankplus
    
    roi1_pankmin = get_final_mask(roi1_mask, green_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankmin_roi1'
    roi_coords[fullid] = roi1_pankmin
    
    roi2_pankplus = get_final_mask(roi2_mask, blue_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankplus_roi2'
    roi_coords[fullid] = roi2_pankplus
    
    roi2_pankmin = get_final_mask(roi2_mask, green_mask, x1_min, y1_min, padding)
    fullid = f'{patient_id}_pankmin_roi2'
    roi_coords[fullid] = roi2_pankmin
    
    # # Save dictionary
    # with open(f'{args.output_dir}/roi_coords_20x/{patient_id}.json', 'w') as f:
    #     json.dump(roi_coords, f)

    tiles_info_df = pd.read_csv(args.hpc_tiles_csv)
    tiles_info_df = tiles_info_df.loc[tiles_info_df.patient == patient_id]
    
    tiles_info_df['tile_xmin'] = (tiles_info_df['Centroid_x'].astype(int) - args.window_size / 2).astype(int)
    tiles_info_df['tile_xmax'] = (tiles_info_df['Centroid_x'].astype(int) + args.window_size / 2).astype(int)
    tiles_info_df['tile_ymin'] = (tiles_info_df['Centroid_y'].astype(int) - args.window_size / 2).astype(int)
    tiles_info_df['tile_ymax'] = (tiles_info_df['Centroid_y'].astype(int) + args.window_size / 2).astype(int)

    
    tiles_to_keep = {}

    for roi in tqdm(roi_coords, total=len(roi_coords), desc="ROIs and PanK", leave=False):
        mask_polygon = roi_coords[roi]  # Binary mask (NumPy array)

        # Split tiles into chunks for parallel processing
        num_chunks = cpu_count()
        tile_chunks = np.array_split(tiles_info_df, num_chunks)

        with Pool(num_chunks) as pool:
            results = pool.map(process_tiles, [(roi, mask_polygon, chunk) for chunk in tile_chunks])

        # Collect results
        tiles_to_keep[roi] = [cell_id for _, cell_ids in results for cell_id in cell_ids]
        
    with open(f'{args.output_dir}/cn_hpc_mapping/{patient_id}.json', 'w') as f:
        json.dump(tiles_to_keep, f)
    

if __name__ == "__main__":
    main()