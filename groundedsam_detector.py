#!/usr/bin/env python3
"""
Grounding DINO-based photo detector for NerdScan

This script uses the Grounding DINO model to detect and segment photos in scanned images
based on text prompts. It's designed to work with the NerdScan project to improve photo
detection accuracy.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import piexif
import re
import datetime
from pathlib import Path


class GroundedDinoDetector:
    """
    A class to detect and segment photos in scanned images using Grounding DINO model.
    """
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None):
        """
        Initialize the detector with the specified model.
        
        Args:
            model_id (str): The Hugging Face model ID for Grounding DINO.
            device (str): Device to run the model on ('cuda' or 'cpu'). 
                          If None, will use CUDA if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
    
    def detect_photos(self, image_path, text_prompt="a photo. a picture.", 
                      box_threshold=0.35, text_threshold=0.25):
        """
        Detect photos in a scanned image using the Grounding DINO model.
        
        Args:
            image_path (str): Path to the input image.
            text_prompt (str): Text prompt for detection. Must be lowercase and end with a period.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text.
            
        Returns:
            tuple: (original_image, bounding_boxes, scores, labels)
                - original_image: The PIL Image object
                - bounding_boxes: List of bounding boxes in [x1, y1, x2, y2] format
                - scores: Confidence scores for each detection
                - labels: Labels for each detection
        """
        # Load image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error: Could not read image {image_path}: {e}")
            return None, [], [], []
        
        # Ensure text prompt is properly formatted
        if not text_prompt.islower() or not text_prompt.endswith('.'):
            print("Warning: Text prompt should be lowercase and end with a period.")
            # Fix the prompt format
            text_prompt = text_prompt.lower()
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'
        
        # Process image with the model
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # [H, W] format
        )
        
        if not results or len(results) == 0 or "boxes" not in results[0]:
            print(f"No photos detected in {image_path}")
            return image, [], [], []
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"]
        
        return image, boxes, scores, labels
    
    def save_cropped_photos(self, image, boxes, output_dir, base_filename, 
                            year=None, min_size=100):
        """
        Save the cropped photos to the output directory.
        
        Args:
            image (PIL.Image): The original image.
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
            output_dir (str): Directory to save the cropped photos.
            base_filename (str): Base filename for the output images.
            year (int, optional): Year to set in EXIF data.
            min_size (int): Minimum size (width or height) for a valid photo.
            
        Returns:
            int: Number of photos saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PIL image to numpy array for OpenCV processing
        img_np = np.array(image)
        
        saved_count = 0
        
        for i, box in enumerate(boxes):
            # Convert to integers and ensure box is within image boundaries
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            # Skip if box is too small
            if (x2 - x1 < min_size) or (y2 - y1 < min_size):
                print(f"  Skipping box {i+1} - too small: {x2-x1}x{y2-y1}")
                continue
            
            # Crop the image
            cropped = img_np[y1:y2, x1:x2]
            
            # Save the cropped image
            output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.jpg")
            cv2.imwrite(output_path, cropped)
            
            # Set EXIF date if year is provided
            if year is not None:
                try:
                    # Format date as YYYY:MM:DD HH:MM:SS
                    date_str = f"{year}:01:01 00:00:00"
                    
                    # Create EXIF data
                    zeroth_ifd = {
                        piexif.ImageIFD.Make: "NerdScan",
                        piexif.ImageIFD.Software: "GroundedDinoDetector"
                    }
                    exif_ifd = {
                        piexif.ExifIFD.DateTimeOriginal: date_str,
                        piexif.ExifIFD.DateTimeDigitized: date_str
                    }
                    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd}
                    exif_bytes = piexif.dump(exif_dict)
                    
                    # Read the saved image and add EXIF data
                    img = Image.open(output_path)
                    img.save(output_path, exif=exif_bytes)
                    
                    print(f"  Set EXIF date {date_str} for {output_path}")
                except Exception as e:
                    print(f"  Error setting EXIF data: {e}")
            
            print(f"  Saved cropped photo: {output_path}")
            saved_count += 1
        
        return saved_count


def process_directory(input_dir, output_dir, text_prompt="a photo. a picture.", 
                      preserve_structure=False, box_threshold=0.35, text_threshold=0.25):
    """
    Process all images in the input directory and save cropped photos to the output directory.
    
    Args:
        input_dir (str): Directory containing input scanned images.
        output_dir (str): Directory to save cropped photos.
        text_prompt (str): Text prompt for detection.
        preserve_structure (bool): If True, preserve folder structure from input to output.
        box_threshold (float): Confidence threshold for bounding boxes.
        text_threshold (float): Confidence threshold for text.
    """
    print(f"Starting processing from '{input_dir}' to '{output_dir}'...")
    
    # Initialize detector
    detector = GroundedDinoDetector()
    
    processed_files = 0
    found_photos = 0
    
    for root, _, files in os.walk(input_dir):
        # Check parent directory name for year
        parent_dir_name = os.path.basename(root)
        year = None
        
        # Check if parent_dir_name is a 4-digit year
        if re.match(r'^\d{4}$', parent_dir_name):
            try:
                year_num = int(parent_dir_name)
                current_year = datetime.datetime.now().year
                # Basic sanity check for plausible years
                if 1800 < year_num <= current_year:
                    year = year_num
                    print(f"  Detected year {year} from folder '{parent_dir_name}' for EXIF.")
            except ValueError:
                pass  # Not a valid integer
        
        for filename in files:
            # Basic check for common image extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')):
                input_path = os.path.join(root, filename)
                print(f"Processing: {input_path}")
                
                try:
                    # Determine output directory
                    if preserve_structure:
                        rel_path = os.path.relpath(root, input_dir)
                        curr_output_dir = os.path.join(output_dir, rel_path)
                    else:
                        curr_output_dir = output_dir
                    
                    os.makedirs(curr_output_dir, exist_ok=True)
                    
                    # Detect photos
                    image, boxes, scores, labels = detector.detect_photos(
                        input_path, text_prompt, box_threshold, text_threshold
                    )
                    
                    if image is None:
                        continue
                    
                    # Base filename without extension
                    base_filename = os.path.splitext(filename)[0]
                    
                    # Save cropped photos
                    saved_count = detector.save_cropped_photos(
                        image, boxes, curr_output_dir, base_filename, year
                    )
                    
                    found_photos += saved_count
                    processed_files += 1
                    
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    
    print(f"\nProcessing complete.")
    print(f"Processed {processed_files} image files.")
    print(f"Found and saved {found_photos} individual photos.")


def main():
    """Main function to parse arguments and start processing."""
    parser = argparse.ArgumentParser(
        description="Detect and crop photos from scanned images using Grounding DINO."
    )
    parser.add_argument(
        "-i", "--input", default="input", 
        help="Input directory containing scanned images."
    )
    parser.add_argument(
        "-o", "--output", default="output", 
        help="Output directory to save cropped photos."
    )
    parser.add_argument(
        "--preserve-structure", action="store_true",
        help="Preserve folder structure from input to output. Default is flat structure."
    )
    parser.add_argument(
        "--text-prompt", default="a photo. a picture. a photograph.",
        help="Text prompt for detection. Should be lowercase and end with a period."
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.35,
        help="Confidence threshold for bounding boxes (0.0 to 1.0)."
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.25,
        help="Confidence threshold for text (0.0 to 1.0)."
    )
    parser.add_argument(
        "--single-image", 
        help="Process a single image instead of a directory."
    )
    parser.add_argument(
        "--output-image", 
        help="Output path for a single processed image."
    )
    
    args = parser.parse_args()
    
    # Ensure input and output dirs are absolute paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.single_image:
        # Process a single image
        if not os.path.isfile(args.single_image):
            print(f"Error: Input file '{args.single_image}' not found.")
            return
        
        output_path = args.output_image or os.path.join(
            script_dir, "output", os.path.basename(args.single_image)
        )
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        detector = GroundedDinoDetector()
        image, boxes, scores, labels = detector.detect_photos(
            args.single_image, args.text_prompt, 
            args.box_threshold, args.text_threshold
        )
        
        if image is not None and len(boxes) > 0:
            base_filename = os.path.splitext(os.path.basename(args.single_image))[0]
            saved_count = detector.save_cropped_photos(
                image, boxes, output_dir, base_filename
            )
            print(f"Processed 1 image and saved {saved_count} photos.")
        else:
            print("No photos detected in the input image.")
    else:
        # Process a directory
        input_dir_abs = os.path.join(script_dir, args.input)
        output_dir_abs = os.path.join(script_dir, args.output)
        
        if not os.path.isdir(input_dir_abs):
            print(f"Error: Input directory '{input_dir_abs}' not found.")
        else:
            os.makedirs(output_dir_abs, exist_ok=True)
            process_directory(
                input_dir_abs, output_dir_abs, args.text_prompt,
                args.preserve_structure, args.box_threshold, args.text_threshold
            )


if __name__ == "__main__":
    main()
