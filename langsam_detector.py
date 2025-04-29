#!/usr/bin/env python3
"""
LangSAM-based photo detector for NerdScan

This script combines text prompts with SAM (Segment Anything Model) for detecting
and segmenting photos in scanned images. Simplified version of the lang-segment-anything
approach adapted for the NerdScan project.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import piexif
import re
import datetime
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LangSAMDetector')

class LangSAMDetector:
    """
    A simplified version of LangSAM that uses text prompts for detecting and 
    segmenting photos in scanned images.
    """
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None, seed=42):
        """
        Initialize the detector with the specified model.
        
        Args:
            model_id (str): The Hugging Face model ID for object detection.
            device (str): Device to run the model on ('cuda' or 'cpu'). 
                          If None, will use CUDA if available.
            seed (int): Random seed for reproducibility.
        """
        # Set random seed for reproducibility
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load text-based object detection model
        logger.info(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        logger.info("Model loaded successfully")
    
    def detect_photos(self, image_path, text_prompt="a photo. a picture.", 
                      box_threshold=0.05, text_threshold=0.05):
        """
        Detect photos in a scanned image using text prompts.
        
        Args:
            image_path (str): Path to the input image.
            text_prompt (str): Text prompt for detection. Must be lowercase and end with a period.
            box_threshold (float): Confidence threshold for bounding boxes.
            text_threshold (float): Confidence threshold for text.
            
        Returns:
            tuple: (original_image, bounding_boxes, scores, labels)
        """
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Text prompt: {text_prompt}")
        logger.info(f"Thresholds - Box: {box_threshold}, Text: {text_threshold}")
        
        # Load image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None, [], [], []
        
        # Ensure text prompt is properly formatted
        if not text_prompt.islower() or not text_prompt.endswith('.'):
            logger.warning("Text prompt should be lowercase and end with a period.")
            # Fix the prompt format
            text_prompt = text_prompt.lower()
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'
        
        # Process image with the model
        logger.info("Running inference with model")
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
            logger.warning(f"No photos detected in {image_path}")
            return image, [], [], []
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"]
        
        logger.info(f"Found {len(boxes)} potential photos")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            logger.info(f"  Photo {i+1}: {label} (Score: {score:.4f}), Box: {box}")
        
        return image, boxes, scores, labels
    
    def calculate_overlap_percentage(self, box1, box2):
        """
        Calculate the overlap percentage between two bounding boxes.
        
        Args:
            box1 (list): First bounding box in [x1, y1, x2, y2] format.
            box2 (list): Second bounding box in [x1, y1, x2, y2] format.
            
        Returns:
            float: Percentage of the smallest box area that overlaps.
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate overlap percentage based on the smaller area
        min_area = min(area1, area2)
        overlap_percentage = (intersection_area / min_area) * 100.0
        
        return overlap_percentage
    
    def remove_overlapping_boxes(self, boxes, scores, labels, overlap_threshold=5.0):
        """
        Remove overlapping bounding boxes, keeping only the ones with higher confidence scores.
        
        Args:
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
            scores (list): Confidence scores for each detection.
            labels (list): Labels for each detection.
            overlap_threshold (float): Percentage threshold for considering boxes as overlapping.
            
        Returns:
            tuple: (filtered_boxes, filtered_scores, filtered_labels)
        """
        if len(boxes) <= 1:
            return boxes, scores, labels
        
        logger.info(f"Checking for overlapping boxes with threshold {overlap_threshold}%")
        
        # Convert to list for easier manipulation
        boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else list(boxes)
        scores_list = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
        labels_list = list(labels)
        
        # Sort indices by confidence score (highest first)
        indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        
        # Initialize list to keep track of which boxes to keep
        keep = [True] * len(boxes_list)
        
        # Check each box against others with lower confidence
        for i, idx1 in enumerate(indices):
            if not keep[idx1]:
                continue  # Skip if already marked for removal
            
            box1 = boxes_list[idx1]
            
            # Compare with remaining boxes of lower confidence
            for idx2 in indices[i+1:]:
                if not keep[idx2]:
                    continue  # Skip if already marked for removal
                
                box2 = boxes_list[idx2]
                
                # Calculate overlap percentage
                overlap = self.calculate_overlap_percentage(box1, box2)
                
                # If overlap exceeds threshold, mark the lower confidence box for removal
                if overlap > overlap_threshold:
                    logger.info(f"  Removing box with score {scores_list[idx2]:.4f} due to {overlap:.2f}% overlap with box with score {scores_list[idx1]:.4f}")
                    keep[idx2] = False
        
        # Filter boxes, scores, and labels
        filtered_boxes = [boxes_list[i] for i in range(len(boxes_list)) if keep[i]]
        filtered_scores = [scores_list[i] for i in range(len(scores_list)) if keep[i]]
        filtered_labels = [labels_list[i] for i in range(len(labels_list)) if keep[i]]
        
        logger.info(f"Removed {len(boxes) - len(filtered_boxes)} overlapping boxes, kept {len(filtered_boxes)}")
        
        # Convert back to numpy arrays if original were numpy arrays
        if isinstance(boxes, np.ndarray):
            filtered_boxes = np.array(filtered_boxes)
        if isinstance(scores, np.ndarray):
            filtered_scores = np.array(filtered_scores)
        
        return filtered_boxes, filtered_scores, filtered_labels
    
    def save_cropped_photos(self, image, boxes, output_dir, base_filename, 
                            year=None, min_size=100, remove_overlaps=False, overlap_threshold=5.0):
        """
        Save the cropped photos to the output directory.
        
        Args:
            image (PIL.Image): The original image.
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
            output_dir (str): Directory to save the cropped photos.
            base_filename (str): Base filename for the output images.
            year (int, optional): Year to set in EXIF data.
            min_size (int): Minimum size (width or height) for a valid photo.
            remove_overlaps (bool): Whether to remove overlapping boxes.
            overlap_threshold (float): Percentage threshold for considering boxes as overlapping.
            
        Returns:
            tuple: (saved_count, output_paths)
        """
        logger.info(f"Saving cropped photos to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove overlapping boxes if requested
        if remove_overlaps and len(boxes) > 1:
            logger.info("Removing overlapping boxes before cropping")
            boxes, scores, labels = self.remove_overlapping_boxes(
                boxes, np.ones(len(boxes)), ["photo"] * len(boxes), overlap_threshold
            )
        
        saved_count = 0
        output_paths = []
        
        for i, box in enumerate(boxes):
            # Convert to integers and ensure box is within image boundaries
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            # Skip if box is too small
            if (x2 - x1 < min_size) or (y2 - y1 < min_size):
                logger.warning(f"  Skipping box {i+1} - too small: {x2-x1}x{y2-y1}")
                continue
            
            # Crop the image - use PIL directly to preserve color profile
            cropped_pil = image.crop((x1, y1, x2, y2))
            
            # Save the cropped image
            output_path = os.path.join(output_dir, f"{base_filename}_{i+1}.jpg")
            cropped_pil.save(output_path, quality=95)
            output_paths.append(output_path)
            
            # Set EXIF date if year is provided
            if year is not None:
                try:
                    # Format date as YYYY:MM:DD HH:MM:SS
                    date_str = f"{year}:01:01 00:00:00"
                    
                    # Create EXIF data
                    zeroth_ifd = {
                        piexif.ImageIFD.Make: "NerdScan",
                        piexif.ImageIFD.Software: "LangSAMDetector"
                    }
                    exif_ifd = {
                        piexif.ExifIFD.DateTimeOriginal: date_str,
                        piexif.ExifIFD.DateTimeDigitized: date_str
                    }
                    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd}
                    exif_bytes = piexif.dump(exif_dict)
                    
                    # Read the saved image and add EXIF data
                    img = Image.open(output_path)
                    img.save(output_path, exif=exif_bytes, quality=95)
                    
                    logger.info(f"  Set EXIF date {date_str} for {output_path}")
                except Exception as e:
                    logger.error(f"  Error setting EXIF data: {e}")
            
            logger.info(f"  Saved cropped photo: {output_path}")
            saved_count += 1
        
        return saved_count, output_paths
    
    def visualize_detection(self, image_path, boxes, scores, labels, output_paths=None, output_vis_path=None):
        """
        Visualize the detection results with bounding boxes and cropped images.
        
        Args:
            image_path (str): Path to the original image.
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
            scores (list): Confidence scores for each detection.
            labels (list): Labels for each detection.
            output_paths (list, optional): Paths to the cropped output images.
            output_vis_path (str, optional): Path to save the visualization.
        """
        logger.info(f"Creating visualization for: {image_path}")
        
        # Load the original image with PIL to preserve colors
        original_pil = Image.open(image_path)
        
        # Create figure
        if output_paths and len(output_paths) > 0:
            # Calculate figure size based on number of crops
            fig_width = min(20, 5 + 4 * len(output_paths))
            fig, axs = plt.subplots(1, 1 + len(output_paths), figsize=(fig_width, 8))
            
            # Display original image with bounding boxes
            axs[0].imshow(original_pil)
            axs[0].set_title('Original with Detections')
            
            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                axs[0].add_patch(rect)
                axs[0].text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Remove axis ticks for cleaner visualization
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            # Display cropped images - use PIL to load to preserve colors
            for i, (output_path, score, label) in enumerate(zip(output_paths, scores, labels)):
                cropped_img = Image.open(output_path)
                axs[i+1].imshow(cropped_img)
                axs[i+1].set_title(f'Crop {i+1}: {label} ({score:.2f})')
                axs[i+1].set_xticks([])
                axs[i+1].set_yticks([])
        else:
            # If we don't have cropped images, just show the original with bounding boxes
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(original_pil)
            ax.set_title('Detections')
            
            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            # Remove axis ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the visualization
        if output_vis_path:
            plt.savefig(output_vis_path, dpi=150)
            logger.info(f"Visualization saved to {output_vis_path}")
        else:
            plt.show()
        
        plt.close()


def run_experiment(experiment_name, text_prompt, input_dir="input", sample_size=5, seed=42, 
                  remove_overlaps=False, overlap_threshold=5.0):
    """
    Run an experiment with specific parameters and save results to an experiment folder.
    
    Args:
        experiment_name (str): Name of the experiment (used for folder name)
        text_prompt (str): Text prompt for detection
        input_dir (str): Directory containing input scanned images
        sample_size (int): Number of random images to process
        seed (int): Random seed for reproducibility
        remove_overlaps (bool): Whether to remove overlapping boxes.
        overlap_threshold (float): Percentage threshold for considering boxes as overlapping.
    """
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Text prompt: {text_prompt}")
    if remove_overlaps:
        logger.info(f"Will remove overlapping boxes with threshold {overlap_threshold}%")
    
    # Create experiment directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.join(script_dir, "langsam_experiments", experiment_name)
    exp_output_dir = os.path.join(experiment_dir, "crops")
    exp_vis_dir = os.path.join(experiment_dir, "visualizations")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(exp_output_dir, exist_ok=True)
    os.makedirs(exp_vis_dir, exist_ok=True)
    
    # Create shell script to reproduce the experiment
    shell_script_path = os.path.join(experiment_dir, "run_experiment.sh")
    with open(shell_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# Experiment: {experiment_name}\n")
        f.write(f"# Text prompt: {text_prompt}\n")
        if remove_overlaps:
            f.write(f"# Remove overlapping boxes with threshold: {overlap_threshold}%\n")
        f.write("\n")
        
        command = (f"python langsam_detector.py -i {input_dir} -o langsam_experiments/{experiment_name}/crops "
                  f"--text-prompt \"{text_prompt}\" "
                  f"--visualize --vis-dir langsam_experiments/{experiment_name}/visualizations "
                  f"--seed {seed} --sample-size {sample_size}")
                  
        if remove_overlaps:
            command += f" --remove-overlaps --overlap-threshold {overlap_threshold}"
            
        f.write(f"{command}\n")
    
    # Make the shell script executable
    os.chmod(shell_script_path, 0o755)
    
    # Create a README.md with experiment details
    readme_path = os.path.join(experiment_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# LangSAM Experiment: {experiment_name}\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- **Text prompt:** `{text_prompt}`\n")
        f.write(f"- **Sample size:** {sample_size}\n")
        f.write(f"- **Seed:** {seed}\n")
        if remove_overlaps:
            f.write(f"- **Remove overlapping boxes:** Yes\n")
            f.write(f"- **Overlap threshold:** {overlap_threshold}%\n")
        f.write("\n")
        f.write("## Results\n\n")
        f.write("The visualizations folder contains images showing the original scan with detected regions and the extracted photos.\n\n")
        f.write("## How to Run\n\n")
        f.write("```bash\n")
        f.write(f"./run_experiment.sh\n")
        f.write("```\n")
    
    # Initialize detector
    detector = LangSAMDetector(seed=seed)
    
    # Collect all image files
    logger.info(f"Scanning directory: {input_dir}")
    all_image_files = []
    input_dir_abs = os.path.join(script_dir, input_dir)
    
    for root, _, files in os.walk(input_dir_abs):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')):
                all_image_files.append((root, filename))
    
    logger.info(f"Found {len(all_image_files)} image files")
    
    # If sample_size is specified, randomly select that many images
    if sample_size and sample_size < len(all_image_files):
        random.seed(seed)
        selected_files = random.sample(all_image_files, sample_size)
        logger.info(f"Selected {sample_size} random images for processing")
    else:
        selected_files = all_image_files
        logger.info(f"Processing all {len(selected_files)} images")
    
    processed_files = 0
    found_photos = 0
    
    # Process each selected file
    for root, filename in selected_files:
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
                    logger.info(f"  Detected year {year} from folder '{parent_dir_name}' for EXIF")
            except ValueError:
                pass  # Not a valid integer
        
        input_path = os.path.join(root, filename)
        logger.info(f"Processing: {input_path}")
        
        try:
            # Determine output directory
            rel_path = os.path.relpath(root, input_dir_abs)
            curr_output_dir = os.path.join(exp_output_dir, rel_path)
            os.makedirs(curr_output_dir, exist_ok=True)
            
            # Detect photos
            image, boxes, scores, labels = detector.detect_photos(
                input_path, text_prompt
            )
            
            if image is None or len(boxes) == 0:
                logger.warning(f"No photos detected in {input_path}")
                continue
            
            # Remove overlapping boxes if requested
            if remove_overlaps and len(boxes) > 1:
                logger.info(f"Removing overlapping boxes with threshold {overlap_threshold}%")
                boxes, scores, labels = detector.remove_overlapping_boxes(
                    boxes, scores, labels, overlap_threshold
                )
                if len(boxes) == 0:
                    logger.warning(f"All boxes were removed due to overlap in {input_path}")
                    continue
            
            # Base filename without extension
            base_filename = os.path.splitext(filename)[0]
            
            # Save cropped photos
            saved_count, output_paths = detector.save_cropped_photos(
                image, boxes, curr_output_dir, base_filename, year
            )
            
            # Create visualization
            curr_vis_dir = os.path.join(exp_vis_dir, rel_path)
            os.makedirs(curr_vis_dir, exist_ok=True)
            
            vis_path = os.path.join(curr_vis_dir, f"{base_filename}_visualization.jpg")
            detector.visualize_detection(
                input_path, boxes, scores, labels, output_paths, vis_path
            )
            
            found_photos += saved_count
            processed_files += 1
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
    
    logger.info(f"Experiment complete. Processed {processed_files} images and found {found_photos} photos.")
    return experiment_dir


def main():
    """Main function to parse arguments and start processing."""
    parser = argparse.ArgumentParser(
        description="Detect and crop photos from scanned images using LangSAM."
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
        "--text-prompt", default="a photo. a picture. a photograph.",
        help="Text prompt for detection. Should be lowercase and end with a period."
    )
    parser.add_argument(
        "--single-image", 
        help="Process a single image instead of a directory."
    )
    parser.add_argument(
        "--output-image", 
        help="Output path for a single processed image."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create visualizations showing the original image with bounding boxes and cropped results."
    )
    parser.add_argument(
        "--vis-dir", default="langsam_visualizations",
        help="Directory to save visualizations."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of random images to process. If not specified, process all images."
    )
    parser.add_argument(
        "--run-experiments", action="store_true",
        help="Run a set of predefined experiments with different parameters."
    )
    parser.add_argument(
        "--remove-overlaps", action="store_true",
        help="Remove overlapping detection boxes, keeping those with higher confidence."
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=5.0,
        help="Percentage threshold for considering boxes as overlapping (default: 5.0)."
    )
    
    args = parser.parse_args()
    
    # Ensure input and output dirs are absolute paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run predefined experiments if requested
    if args.run_experiments:
        experiments = [
            {
                "name": "exp_dedup",
                "prompt": "photo",
                "remove_overlaps": True,
                "overlap_threshold": 5.0
            },
            # {
            #     "name": "exp2_detailed",
            #     "prompt": "a photo. a picture. a photograph. a snapshot. an image. a portrait."
            # },
            # {
            #     "name": "exp3_vintage",
            #     "prompt": "an old photo. a vintage photograph. a historical picture. a family photo."
            # }
        ]
        
        for exp in experiments:
            print(f"\n\n{'='*50}")
            print(f"Running experiment: {exp['name']}")
            print(f"{'='*50}\n")
            
            run_experiment(
                exp["name"], exp["prompt"], 
                args.input, sample_size=10, seed=args.seed,
                remove_overlaps=exp.get("remove_overlaps", False),
                overlap_threshold=exp.get("overlap_threshold", 5.0)
            )
        
        print("\nAll experiments completed!")
        return
    
    elif args.single_image:
        # Process a single image
        if not os.path.isfile(args.single_image):
            logger.error(f"Error: Input file '{args.single_image}' not found.")
            return
        
        output_path = args.output_image or os.path.join(
            script_dir, "output", os.path.basename(args.single_image)
        )
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        detector = LangSAMDetector(seed=args.seed)
        image, boxes, scores, labels = detector.detect_photos(
            args.single_image, args.text_prompt
        )
        
        if image is not None and len(boxes) > 0:
            # Remove overlapping boxes if requested
            if args.remove_overlaps and len(boxes) > 1:
                logger.info(f"Removing overlapping boxes with threshold {args.overlap_threshold}%")
                boxes, scores, labels = detector.remove_overlapping_boxes(
                    boxes, scores, labels, args.overlap_threshold
                )
            
            if len(boxes) > 0:
                base_filename = os.path.splitext(os.path.basename(args.single_image))[0]
                saved_count, output_paths = detector.save_cropped_photos(
                    image, boxes, output_dir, base_filename
                )
                
                # Create visualization if requested
                if args.visualize:
                    vis_dir = os.path.join(script_dir, args.vis_dir)
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f"{base_filename}_visualization.jpg")
                    detector.visualize_detection(
                        args.single_image, boxes, scores, labels, output_paths, vis_path
                    )
                
                logger.info(f"Processed 1 image and saved {saved_count} photos.")
            else:
                logger.warning("All detections were removed due to overlap.")
        else:
            logger.warning("No photos detected in the input image.")
    else:
        # Run single experiment with command line parameters
        exp_name = "cmd_line_exp"
        if args.remove_overlaps:
            exp_name = "cmd_line_exp_dedup"
        
        run_experiment(
            exp_name, args.text_prompt, 
            args.input, sample_size=args.sample_size, seed=args.seed,
            remove_overlaps=args.remove_overlaps,
            overlap_threshold=args.overlap_threshold
        )


if __name__ == "__main__":
    main()
