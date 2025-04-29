import cv2
import numpy as np
import os
import argparse
import piexif
import re
import datetime
import subprocess
from collections import defaultdict

def find_photos(image_path):
    """Finds bounding boxes of photos in a scanned image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return img, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    # We assume photos are lighter than the background we want to remove
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Optional: Add morphological operations here if needed (e.g., to remove noise)
    # kernel = np.ones((5,5),np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    photo_contours = []
    # Lower the minimum area threshold significantly
    min_area = (img.shape[0] * img.shape[1]) * 0.005 # Heuristic: photo area > 0.5% of total image
    max_area = (img.shape[0] * img.shape[1]) * 0.95 # Heuristic: photo area < 95% of total image

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Optional: Add aspect ratio filtering if needed
            # x, y, w, h = cv2.boundingRect(contour)
            # aspect_ratio = float(w)/h
            # if 0.5 < aspect_ratio < 2.0: # Example aspect ratio filter
            photo_contours.append(contour)

    # Sort contours top-to-bottom, left-to-right (optional but can be helpful)
    bounding_boxes = [cv2.boundingRect(c) for c in photo_contours]
    # Simple top-to-bottom sort based on y-coordinate
    bounding_boxes.sort(key=lambda b: b[1]) 

    return img, bounding_boxes

def main(input_dir, output_dir, preserve_structure=False):
    """Processes all images in input_dir and saves cropped photos to output_dir.
    
    Args:
        input_dir: Directory containing input scanned images
        output_dir: Directory to save cropped photos
        preserve_structure: If True, preserve the folder structure of input_dir in output_dir
                          If False (default), use a flat structure in output_dir
    """
    print(f"Starting processing from '{input_dir}' to '{output_dir}'...")
    processed_files = 0
    found_photos = 0
    
    # Track number of photos per year for sequential dating
    year_photo_count = defaultdict(int)
    
    for root, _, files in os.walk(input_dir):
        # Check parent directory name for year
        parent_dir_name = os.path.basename(root)
        exif_date_str = None
        year = None
        # Check if parent_dir_name is a 4-digit year
        if re.match(r'^\d{4}$', parent_dir_name):
            try:
                year_num = int(parent_dir_name)
                current_year = datetime.datetime.now().year
                # Basic sanity check for plausible years (adjust range as needed)
                if 1800 < year_num <= current_year: 
                    year = year_num
                    # Use mid-year date instead of January 1st
                    # This looks more like a real capture date vs a placeholder
                    # Base date for this year - will be adjusted by scan order
                    print(f"  Detected year {year} from folder '{parent_dir_name}' for EXIF.")
            except ValueError:
                pass # Not a valid integer

        for filename in files:
            # Basic check for common image extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')):
                input_path = os.path.join(root, filename)
                print(f"Processing: {input_path}")

                try:
                    original_image, bounding_boxes = find_photos(input_path)

                    if original_image is None:
                        continue # Skip if image couldn't be read
                    
                    if not bounding_boxes:
                        print(f"  No photos found in {filename}. Skipping.")
                        continue

                    # Determine output directory based on preserve_structure flag
                    if preserve_structure:
                        # Create corresponding output subdirectory matching input structure
                        relative_path = os.path.relpath(root, input_dir)
                        current_output_dir = os.path.join(output_dir, relative_path)
                    else:
                        # Completely flat structure - all photos go directly in output_dir
                        current_output_dir = output_dir
                    
                    os.makedirs(current_output_dir, exist_ok=True)

                    base_filename, ext = os.path.splitext(filename)

                    for i, (x, y, w, h) in enumerate(bounding_boxes):
                        # Add a small padding (optional)
                        padding = 5 
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(original_image.shape[1], x + w + padding)
                        y2 = min(original_image.shape[0], y + h + padding)
                        
                        cropped_photo = original_image[y1:y2, x1:x2]

                        # Construct output filename with the year if available
                        if year:
                            # Update the count for this year and use it to create sequential numbering
                            year_photo_count[year] += 1
                            photo_index = year_photo_count[year]
                            
                            # Create a sequential date within the year (starting from January)
                            # This preserves the scan order while keeping dates in the same year
                            # We'll space them evenly throughout the year (roughly 1 day apart)
                            day_of_year = min(photo_index, 365)  # Cap at 365 days
                            
                            # Convert day of year to month/day
                            date_in_year = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year-1)
                            month = date_in_year.month
                            day = date_in_year.day
                            
                            # Format with leading zeros
                            exif_date_str = f"{year}:{month:02d}:{day:02d} 12:00:00"
                            
                            output_filename = f"{base_filename}_{year}_{photo_index:03d}.jpg"
                        else:
                            output_filename = f"{base_filename}_{i+1:03d}.jpg"
                            
                        output_path = os.path.join(current_output_dir, output_filename)

                        # Save the cropped photo as JPG
                        # Quality parameter (0-100), higher is better quality/larger file
                        save_success = cv2.imwrite(output_path, cropped_photo, [cv2.IMWRITE_JPEG_QUALITY, 95])

                        if not save_success:
                            print(f"  Error: Failed to save {output_path}")
                            continue

                        # Set EXIF date with piexif and exiftool if year was detected
                        if exif_date_str:
                            try:
                                # Basic EXIF setting with piexif (might not be strictly needed if exiftool works)
                                exif_dict = {"Exif": {
                                    piexif.ExifIFD.DateTimeOriginal: exif_date_str.encode('utf-8'),
                                    piexif.ExifIFD.DateTimeDigitized: exif_date_str.encode('utf-8')
                                }}
                                try:
                                    exif_bytes = piexif.dump(exif_dict)
                                    piexif.insert(exif_bytes, output_path)
                                except Exception as piexif_e:
                                    print(f"    Warning: piexif failed to insert basic EXIF: {piexif_e}")

                                # Parse YYYY:MM:DD from exif_date_str for IPTC format
                                date_parts = exif_date_str.split()[0].split(':')
                                iptc_date_str = f"{date_parts[0]}{date_parts[1]}{date_parts[2]}"
                                
                                # Use exiftool to set ALL possible date tags and filesystem dates
                                cmd = [
                                    'exiftool',
                                    '-overwrite_original',
                                    # Set standard EXIF date tags
                                    f'-EXIF:DateTimeOriginal={exif_date_str}',
                                    f'-EXIF:CreateDate={exif_date_str}',
                                    f'-EXIF:ModifyDate={exif_date_str}',
                                    # Set XMP date tags
                                    f'-XMP:DateCreated={exif_date_str}',
                                    f'-XMP:CreateDate={exif_date_str}',
                                    f'-XMP:ModifyDate={exif_date_str}',
                                    # Set IPTC date tags (YYYYMMDD format)
                                    f'-IPTC:DateCreated={iptc_date_str}',
                                    # Set filesystem dates (this is important for Google Photos)
                                    f'-FileCreateDate={exif_date_str}',
                                    f'-FileModifyDate={exif_date_str}',
                                    # Set more date-related tags for good measure
                                    f'-AllDates={exif_date_str}',
                                    # Write tags as specific types (might help compatibility)
                                    '-E', # preserve existing tags
                                    '-F', # fix tags that don't match expected format
                                    output_path
                                ]
                                result = subprocess.run(cmd, capture_output=True, text=True, check=False) # Don't check=True initially

                                if result.returncode == 0:
                                    print(f"    Successfully set EXIF date via exiftool for {output_path}")
                                else:
                                    print(f"    Error running exiftool for {output_path}: {result.stderr}")

                            except Exception as e:
                                print(f"    Error processing EXIF/running exiftool for {output_path}: {e}")

                        print(f"  Saved cropped photo: {output_path}") # Reports save potentially including EXIF
                        found_photos += 1
                    
                    processed_files += 1

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print(f"\nProcessing complete.")
    print(f"Processed {processed_files} image files.")
    print(f"Found and saved {found_photos} individual photos.")
    
    # Print count of photos per year
    for year, count in sorted(year_photo_count.items()):
        print(f"Year {year}: {count} photos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop photos from scanned images.")
    parser.add_argument("-i", "--input", default="input", help="Input directory containing scanned images.")
    parser.add_argument("-o", "--output", default="output", help="Output directory to save cropped photos.")
    parser.add_argument("--preserve-structure", action="store_true", help="Preserve folder structure from input to output. Default is flat structure.")
    args = parser.parse_args()

    # Ensure input and output dirs are absolute paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_abs = os.path.join(script_dir, args.input)
    output_dir_abs = os.path.join(script_dir, args.output)
    
    if not os.path.isdir(input_dir_abs):
        print(f"Error: Input directory '{input_dir_abs}' not found.")
    else:
        os.makedirs(output_dir_abs, exist_ok=True)
        main(input_dir_abs, output_dir_abs, args.preserve_structure)
