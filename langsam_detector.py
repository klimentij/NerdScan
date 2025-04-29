#!/usr/bin/env python3
"""
LangSAM-based photo detector for NerdScan

This script uses the lang-segment-anything model to detect and segment photos in scanned images
based on text prompts. It's designed to be a lightweight alternative for the NerdScan project.
"""

import os
# Set environment variable for MPS fallback to CPU when needed
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from lang_sam import LangSAM


class LangSAMDetector:
    """
    A lightweight class to detect and segment photos in scanned images using the LangSAM model.
    """
    
    def __init__(self, text_prompt="photo", threshold=0.5, filter_full_image=True, debug=False):
        """
        Initialize the LangSAM detector.
        
        Args:
            text_prompt (str): The text prompt to use for detection. Default is "photo".
            threshold (float): Confidence threshold for detections (0-1).
            filter_full_image (bool): Whether to filter out detections that cover most of the image.
            debug (bool): Whether to print debug information.
        """
        self.model = LangSAM()
        self.text_prompt = text_prompt
        self.threshold = threshold
        self.filter_full_image = filter_full_image
        self.debug = debug
    
    def detect(self, image_path, save_masks=False):
        """
        Detect photos in the given image.
        
        Args:
            image_path (str): Path to the image file.
            save_masks (bool): Whether to save the raw masks for debugging.
        
        Returns:
            tuple: (boxes, raw_results) - List of detected photo regions and raw results from model
        """
        # Load the image
        try:
            image_pil = Image.open(image_path).convert("RGB")
            width, height = image_pil.size
            if self.debug:
                print(f"Image dimensions: {width}x{height}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return [], None
        
        # Run detection
        print(f"Running detection with prompt: '{self.text_prompt}'")
        results = self.model.predict([image_pil], [self.text_prompt])
        
        # Extract bounding boxes
        boxes = []
        raw_masks = None
        
        if results and len(results) > 0:
            result = results[0]
            masks = result.get("masks", [])
            scores = result.get("scores", [1.0] * len(masks))
            
            if self.debug:
                print(f"Raw results: {result.keys()}")
                print(f"Found {len(masks)} masks with scores: {scores}")
            
            # Save masks for debugging if requested
            if save_masks and len(masks) > 0:
                self.save_debug_masks(image_path, image_pil, masks)
                raw_masks = masks
            
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if score < self.threshold:
                    if self.debug:
                        print(f"  Skipping mask {i+1} with score {score} (below threshold {self.threshold})")
                    continue
                
                # Convert mask to bounding box
                mask_np = np.array(mask)
                y_indices, x_indices = np.where(mask_np)
                
                if len(y_indices) == 0 or len(x_indices) == 0:
                    if self.debug:
                        print(f"  Skipping mask {i+1}: empty mask")
                    continue
                    
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Calculate box dimensions
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                if self.debug:
                    print(f"  Mask {i+1}: Found box ({x_min}, {y_min}, {box_width}, {box_height})")
                    print(f"           - Coverage: {box_width * box_height / (width * height):.4f} of image")
                
                # Filter out full-image detections (if enabled)
                if self.filter_full_image:
                    img_area = width * height
                    box_area = box_width * box_height
                    coverage_ratio = box_area / img_area
                    
                    # Skip if the box covers more than 95% of the image
                    if coverage_ratio > 0.95:
                        print(f"  Filtered detection {i+1}: covers {coverage_ratio:.2f} of image")
                        continue
                
                # (x, y, width, height) format
                boxes.append((int(x_min), int(y_min), int(box_width), int(box_height)))
        else:
            print("No results returned from model")
        
        return boxes, raw_masks
    
    def save_debug_masks(self, image_path, image_pil, masks):
        """Save individual masks for debugging"""
        base_path = Path(image_path).stem
        mask_dir = Path("test_images") / f"{base_path}_masks"
        mask_dir.mkdir(exist_ok=True, parents=True)
        
        # Save original image for reference
        image_pil.save(mask_dir / "original.jpg")
        
        # Save each mask as a separate image
        for i, mask in enumerate(masks):
            mask_np = np.array(mask) * 255  # Convert to 0-255 range
            mask_img = Image.fromarray(mask_np.astype(np.uint8))
            mask_img.save(mask_dir / f"mask_{i+1}.png")
            
            # Also create a visualization with the mask overlaid on the image
            mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask_np > 0] = [0, 255, 0]  # Green mask
            
            # Convert PIL image to numpy array
            img_np = np.array(image_pil)
            
            # Create alpha blended overlay
            alpha = 0.5
            overlay = cv2.addWeighted(img_np, 1, mask_rgb, alpha, 0)
            
            # Save overlay
            cv2.imwrite(str(mask_dir / f"overlay_{i+1}.jpg"), overlay[:, :, ::-1])  # RGB to BGR
        
        print(f"Saved debug masks to {mask_dir}")
    
    def visualize(self, image_path, boxes, output_path=None):
        """
        Visualize the detected boxes on the image.
        
        Args:
            image_path (str): Path to the original image.
            boxes (list): List of bounding boxes in (x, y, w, h) format.
            output_path (str, optional): Path to save the visualization. If None, display only.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path} for visualization")
            return
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw each box
        for i, (x, y, w, h) in enumerate(boxes):
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add label
            label = f"Photo {i+1}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
        
        # Save or display the image
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to {output_path}")
        
        return vis_image


def main():
    """
    Test the LangSAM detector on a sample image.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect photos using LangSAM model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="photo", help="Text prompt for detection")
    parser.add_argument("--output", type=str, help="Path to output image with detections")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold (0-1)")
    parser.add_argument("--keep-full", action="store_true", help="Keep detections that cover the full image")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--save-masks", action="store_true", help="Save raw masks for debugging")
    args = parser.parse_args()
    
    # Create detector
    detector = LangSAMDetector(
        text_prompt=args.prompt,
        threshold=args.threshold,
        filter_full_image=not args.keep_full,
        debug=args.debug
    )
    
    # Detect photos
    boxes, _ = detector.detect(args.image, save_masks=args.save_masks)
    
    # Draw results
    if args.output:
        detector.visualize(args.image, boxes, args.output)
    
    # Print results
    print(f"Detected {len(boxes)} photos in {args.image}")
    for i, box in enumerate(boxes):
        print(f"  Photo {i+1}: {box}")


if __name__ == "__main__":
    main()
