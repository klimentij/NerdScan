import cv2
import numpy as np
import os
import argparse

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

def main(input_dir, output_dir):
    """Processes all images in input_dir and saves cropped photos to output_dir."""
    print(f"Starting processing from '{input_dir}' to '{output_dir}'...")
    processed_files = 0
    found_photos = 0

    for root, _, files in os.walk(input_dir):
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

                    # Create corresponding output subdirectory
                    relative_path = os.path.relpath(root, input_dir)
                    current_output_dir = os.path.join(output_dir, relative_path)
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

                        # Construct output filename
                        # If only one photo found, keep original name (or close to it)
                        # If multiple, append index
                        output_filename = f"{base_filename}.jpg" if len(bounding_boxes) == 1 else f"{base_filename}_{i+1}.jpg"
                        output_path = os.path.join(current_output_dir, output_filename)

                        # Save the cropped photo as JPG
                        # Quality parameter (0-100), higher is better quality/larger file
                        cv2.imwrite(output_path, cropped_photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        print(f"  Saved cropped photo: {output_path}")
                        found_photos += 1
                    
                    processed_files += 1

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print(f"\nProcessing complete.")
    print(f"Processed {processed_files} image files.")
    print(f"Found and saved {found_photos} individual photos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop photos from scanned images.")
    parser.add_argument("-i", "--input", default="input", help="Input directory containing scanned images.")
    parser.add_argument("-o", "--output", default="output", help="Output directory to save cropped photos.")
    args = parser.parse_args()

    # Ensure input and output dirs are absolute paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_abs = os.path.join(script_dir, args.input)
    output_dir_abs = os.path.join(script_dir, args.output)
    
    if not os.path.isdir(input_dir_abs):
        print(f"Error: Input directory '{input_dir_abs}' not found.")
    else:
        os.makedirs(output_dir_abs, exist_ok=True)
        main(input_dir_abs, output_dir_abs)
