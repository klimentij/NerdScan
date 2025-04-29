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
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GroundedDinoDetector:
    """
    A class to detect and segment photos in scanned images using Grounding DINO model.
    """
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None, seed=42):
        """
        Initialize the detector with the specified model.
        
        Args:
            model_id (str): The Hugging Face model ID for Grounding DINO.
            device (str): Device to run the model on ('cuda' or 'cpu'). 
                          If None, will use CUDA if available.
            seed (int): Random seed for reproducibility.
        """
        # Set random seed for reproducibility
        if seed is not None:
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
            tuple: (saved_count, output_paths)
                - saved_count: Number of photos saved
                - output_paths: List of paths to saved photos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PIL image to numpy array for OpenCV processing
        img_np = np.array(image)
        
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
                print(f"  Skipping box {i+1} - too small: {x2-x1}x{y2-y1}")
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
                    img.save(output_path, exif=exif_bytes, quality=95)
                    
                    print(f"  Set EXIF date {date_str} for {output_path}")
                except Exception as e:
                    print(f"  Error setting EXIF data: {e}")
            
            print(f"  Saved cropped photo: {output_path}")
            saved_count += 1
        
        return saved_count, output_paths
    
    def visualize_detection(self, image_path, boxes, scores, labels, output_paths=None, output_vis_path=None):
        """
        Visualize the detection results with bounding boxes and optionally the cropped images.
        
        Args:
            image_path (str): Path to the original image.
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
            scores (list): Confidence scores for each detection.
            labels (list): Labels for each detection.
            output_paths (list, optional): Paths to the cropped output images.
            output_vis_path (str, optional): Path to save the visualization.
            
        Returns:
            None
        """
        # Load the original image with PIL to preserve colors
        original_pil = Image.open(image_path)
        
        # Create figure
        if output_paths and len(output_paths) > 0:
            # Calculate figure size based on number of crops
            fig_width = min(20, 5 + 4 * len(output_paths))
            fig, axs = plt.subplots(1, 1 + len(output_paths), figsize=(fig_width, 8))
            
            # Display original image with bounding boxes
            axs[0].imshow(original_pil)
            axs[0].set_title('Original Image with Detections')
            
            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                axs[0].add_patch(rect)
                axs[0].text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=12, 
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Display cropped images - use PIL to load to preserve colors
            for i, output_path in enumerate(output_paths):
                cropped_img = Image.open(output_path)
                axs[i+1].imshow(cropped_img)
                axs[i+1].set_title(f'Cropped Image {i+1}')
                
                # Remove axis ticks for cleaner visualization
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
                ax.text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            # Remove axis ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Remove axis ticks from original image for cleaner visualization
        axs[0].set_xticks([]) if output_paths and len(output_paths) > 0 else None
        axs[0].set_yticks([]) if output_paths and len(output_paths) > 0 else None
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the visualization
        if output_vis_path:
            plt.savefig(output_vis_path, dpi=150)
            print(f"Visualization saved to {output_vis_path}")
        else:
            plt.show()
        
        plt.close()


def process_directory(input_dir, output_dir, text_prompt="a photo. a picture.", 
                      preserve_structure=False, box_threshold=0.35, text_threshold=0.25,
                      visualize=False, vis_dir=None, seed=42, sample_size=None):
    """
    Process all images in the input directory and save cropped photos to the output directory.
    
    Args:
        input_dir (str): Directory containing input scanned images.
        output_dir (str): Directory to save cropped photos.
        text_prompt (str): Text prompt for detection.
        preserve_structure (bool): If True, preserve folder structure from input to output.
        box_threshold (float): Confidence threshold for bounding boxes.
        text_threshold (float): Confidence threshold for text.
        visualize (bool): Whether to create visualizations of detections.
        vis_dir (str): Directory to save visualizations.
        seed (int): Random seed for reproducibility.
        sample_size (int): Number of random images to process. If None, process all.
    """
    print(f"Starting processing from '{input_dir}' to '{output_dir}'...")
    print(f"Using text prompt: '{text_prompt}'")
    print(f"Box threshold: {box_threshold}, Text threshold: {text_threshold}")
    
    # Initialize detector with seed
    detector = GroundedDinoDetector(seed=seed)
    
    processed_files = 0
    found_photos = 0
    
    if visualize and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Collect all image files first
    all_image_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')):
                all_image_files.append((root, filename))
    
    # If sample_size is specified, randomly select that many images
    if sample_size and sample_size < len(all_image_files):
        # Set random seed for reproducibility
        random.seed(seed)
        selected_files = random.sample(all_image_files, sample_size)
    else:
        selected_files = all_image_files
    
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
                    print(f"  Detected year {year} from folder '{parent_dir_name}' for EXIF.")
            except ValueError:
                pass  # Not a valid integer
        
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
            saved_count, output_paths = detector.save_cropped_photos(
                image, boxes, curr_output_dir, base_filename, year
            )
            
            # Create visualization if requested
            if visualize and vis_dir and len(boxes) > 0:
                if preserve_structure:
                    curr_vis_dir = os.path.join(vis_dir, rel_path)
                    os.makedirs(curr_vis_dir, exist_ok=True)
                else:
                    curr_vis_dir = vis_dir
                
                vis_path = os.path.join(curr_vis_dir, f"{base_filename}_visualization.jpg")
                detector.visualize_detection(
                    input_path, boxes, scores, labels, output_paths, vis_path
                )
            
            found_photos += saved_count
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    print(f"\nProcessing complete.")
    print(f"Processed {processed_files} image files.")
    print(f"Found and saved {found_photos} individual photos.")


def run_experiment(experiment_name, text_prompt, box_threshold=0.35, text_threshold=0.25, 
                   input_dir="input", output_dir="output", sample_size=10, seed=42):
    """
    Run an experiment with specific parameters and save results to an experiment folder.
    
    Args:
        experiment_name (str): Name of the experiment (used for folder name)
        text_prompt (str): Text prompt for detection
        box_threshold (float): Confidence threshold for bounding boxes
        text_threshold (float): Confidence threshold for text
        input_dir (str): Directory containing input scanned images
        output_dir (str): Base directory to save output
        sample_size (int): Number of random images to process
        seed (int): Random seed for reproducibility
    """
    # Create experiment directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.join(script_dir, "experiments", experiment_name)
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
        f.write(f"# Box threshold: {box_threshold}\n")
        f.write(f"# Text threshold: {text_threshold}\n\n")
        
        command = (f"python groundedsam_detector.py -i {input_dir} -o experiments/{experiment_name}/crops "
                  f"--text-prompt \"{text_prompt}\" --box-threshold {box_threshold} "
                  f"--text-threshold {text_threshold} --visualize --vis-dir experiments/{experiment_name}/visualizations "
                  f"--seed {seed} --sample-size {sample_size}")
        
        f.write(f"{command}\n")
    
    # Make the shell script executable
    os.chmod(shell_script_path, 0o755)
    
    # Create a README.md with experiment details
    readme_path = os.path.join(experiment_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Experiment: {experiment_name}\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- **Text prompt:** `{text_prompt}`\n")
        f.write(f"- **Box threshold:** {box_threshold}\n")
        f.write(f"- **Text threshold:** {text_threshold}\n")
        f.write(f"- **Sample size:** {sample_size}\n")
        f.write(f"- **Seed:** {seed}\n\n")
        f.write("## Results\n\n")
        f.write("The visualizations folder contains images showing the original scan with detected regions and the extracted photos.\n\n")
        f.write("## How to Run\n\n")
        f.write("```bash\n")
        f.write(f"./run_experiment.sh\n")
        f.write("```\n")
    
    # Run the experiment
    input_dir_abs = os.path.join(script_dir, input_dir)
    
    process_directory(
        input_dir_abs, exp_output_dir, text_prompt,
        preserve_structure=True, box_threshold=box_threshold, text_threshold=text_threshold,
        visualize=True, vis_dir=exp_vis_dir, seed=seed, sample_size=sample_size
    )


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
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create visualizations showing the original image with bounding boxes and cropped results."
    )
    parser.add_argument(
        "--vis-dir", default="visualizations",
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
    
    args = parser.parse_args()
    
    # Ensure input and output dirs are absolute paths relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run predefined experiments if requested
    if args.run_experiments:
        experiments = [
            {
                "name": "exp1_basic_photo",
                "prompt": "a photo. a picture. a photograph.",
                "box_threshold": 0.35,
                "text_threshold": 0.25
            },
            {
                "name": "exp2_detailed_photo",
                "prompt": "a photo. a picture. a photograph. a snapshot. an image. a portrait.",
                "box_threshold": 0.3,
                "text_threshold": 0.2
            },
            {
                "name": "exp3_vintage_photo",
                "prompt": "an old photo. a vintage photograph. a historical picture. a family photo.",
                "box_threshold": 0.3,
                "text_threshold": 0.2
            },
            {
                "name": "exp4_high_precision",
                "prompt": "a photo. a picture. a photograph.",
                "box_threshold": 0.4,
                "text_threshold": 0.3
            }
        ]
        
        for exp in experiments:
            print(f"\n\n{'='*50}")
            print(f"Running experiment: {exp['name']}")
            print(f"{'='*50}\n")
            
            run_experiment(
                exp["name"], exp["prompt"], 
                exp["box_threshold"], exp["text_threshold"],
                args.input, args.output, 
                sample_size=10, seed=args.seed
            )
        
        print("\nAll experiments completed!")
        return
    
    elif args.single_image:
        # Process a single image
        if not os.path.isfile(args.single_image):
            print(f"Error: Input file '{args.single_image}' not found.")
            return
        
        output_path = args.output_image or os.path.join(
            script_dir, "output", os.path.basename(args.single_image)
        )
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        detector = GroundedDinoDetector(seed=args.seed)
        image, boxes, scores, labels = detector.detect_photos(
            args.single_image, args.text_prompt, 
            args.box_threshold, args.text_threshold
        )
        
        if image is not None and len(boxes) > 0:
            base_filename = os.path.splitext(os.path.basename(args.single_image))[0]
            saved_count, output_paths = detector.save_cropped_photos(
                image, boxes, output_dir, base_filename
            )
            
            # Create visualization if requested
            if args.visualize and len(boxes) > 0:
                vis_dir = os.path.join(script_dir, args.vis_dir)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{base_filename}_visualization.jpg")
                detector.visualize_detection(
                    args.single_image, boxes, scores, labels, output_paths, vis_path
                )
            
            print(f"Processed 1 image and saved {saved_count} photos.")
        else:
            print("No photos detected in the input image.")
    else:
        # Process a directory
        input_dir_abs = os.path.join(script_dir, args.input)
        output_dir_abs = os.path.join(script_dir, args.output)
        vis_dir_abs = os.path.join(script_dir, args.vis_dir) if args.visualize else None
        
        if not os.path.isdir(input_dir_abs):
            print(f"Error: Input directory '{input_dir_abs}' not found.")
        else:
            os.makedirs(output_dir_abs, exist_ok=True)
            process_directory(
                input_dir_abs, output_dir_abs, args.text_prompt,
                args.preserve_structure, args.box_threshold, args.text_threshold,
                args.visualize, vis_dir_abs, args.seed, args.sample_size
            )


if __name__ == "__main__":
    main()
