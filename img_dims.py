"""
Get descriptive statistics about the images of a dataset.

Provide a path to the directory of the dataset, then run this file.
"""
import os
import numpy as np
from PIL import Image

def get_image_statistics_in_directory(directory):
    widths = []
    heights = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Open the image using Pillow
                with Image.open(file_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if widths and heights:
        width_mean = np.mean(widths)
        height_mean = np.mean(heights)
        width_median = np.median(widths)
        height_median = np.median(heights)
        width_std = np.std(widths)
        height_std = np.std(heights)
        width_min = min(widths)
        height_min = min(heights)
        width_max = max(widths)
        height_max = max(heights)

        print("Width Statistics:")
        print(f"  Mean: {width_mean}")
        print(f"  Median: {width_median}")
        print(f"  Standard Deviation: {width_std}")
        print(f"  Min: {width_min}")
        print(f"  Max: {width_max}")

        print("Height Statistics:")
        print(f"  Mean: {height_mean}")
        print(f"  Median: {height_median}")
        print(f"  Standard Deviation: {height_std}")
        print(f"  Min: {height_min}")
        print(f"  Max: {height_max}")
    else:
        print("No valid images found in the directory.")

# Example usage:
#directory_path = 'data/data'
#get_image_statistics_in_directory(directory_path)
