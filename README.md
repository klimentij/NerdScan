# NerdScan 📷 ✨

<div align="center">

![NerdScan Logo](https://img.shields.io/badge/NerdScan-Photo%20Detection%20%26%20Extraction-blue?style=for-the-badge&logo=image)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Revive your old photos from scanned documents using AI-powered detection and extraction**

[Key Features](#key-features) | [Installation](#installation) | [Usage](#usage) | [Examples](#examples) | [How It Works](#how-it-works) | [Configuration](#configuration) | [Contributing](#contributing) | [License](#license)

</div>

---

## Key Features

- 🔍 **AI-Powered Detection**: Uses state-of-the-art AI object detection to find photos in scans
- 🖼️ **Smart Photo Extraction**: Precisely crops and saves individual photos from cluttered scans
- 📊 **Visual Feedback**: Creates clear visualizations of detection results for easy verification
- 📅 **Intelligent Dating**: Automatically adds EXIF metadata based on folder structure with sequential dates
- 🌲 **Flexible Organization**: Option to preserve original folder structure or use flat output
- 🔄 **Smart Filtering**: Removes overlapping or low-confidence detections for clean results
- 🚀 **User-Friendly CLI**: Beautiful command-line interface with progress tracking and rich output
- 🛠️ **Highly Configurable**: Fine-tune every aspect of detection to suit your needs

## Why NerdScan?

If you have old photo albums, scrapbooks, or document collections with multiple photos per page, NerdScan makes digitizing them effortless. Instead of manually cropping each photo, simply scan entire pages and let NerdScan handle the rest - it finds, extracts, and even dates your photos automatically!

## Installation

### Prerequisites

- Python 3.7+
- [Optional] ExifTool for enhanced metadata support
- [Optional] CUDA-capable GPU for faster processing

### Quick Install

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/NerdScan.git
   cd NerdScan
   ```

2. Create a virtual environment and install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

### Additional Tools

For comprehensive EXIF metadata handling, install ExifTool:

- **macOS**: `brew install exiftool`
- **Linux**: `apt-get install libimage-exiftool-perl`
- **Windows**: Download from https://exiftool.org/

## Usage

### Quick Start

Process all images in the input directory:

```bash
python main.py -i input -o output
```

This will:
1. Scan for images in the `input` directory
2. Extract detected photos to `output/crops`
3. Save visualizations to `output/visualizations`

### Command Line Options

```
python main.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input directory containing scanned images | "input" |
| `-o, --output` | Output directory for results | "output" |
| `--text-prompt` | Text prompt for AI detection | "a photo. a picture. a photograph." |
| `--single-image` | Process a single image instead of directory | None |
| `--preserve-structure` | Preserve folder structure in output | False |
| `--remove-overlaps` | Remove overlapping detection boxes | False |
| `--overlap-threshold` | Threshold for overlap detection | 5.0 |
| `--confidence-threshold` | Minimum confidence score to keep detections | 0.15 |
| `--seed` | Random seed for reproducibility | 42 |
| `--sample-size` | Number of random images to process | All |
| `--device` | Device to run model on ('cuda' or 'cpu') | Auto-detect |

## Examples

### Process a Single Image

```bash
python main.py --single-image path/to/scan.jpg
```

### Preserve Folder Structure

```bash
python main.py -i input -o output --preserve-structure
```

### Use a Custom Text Prompt

For older or vintage photos:
```bash
python main.py -i input -o output --text-prompt "an old photograph. a vintage photo."
```

For Polaroid photos:
```bash
python main.py -i input -o output --text-prompt "a polaroid photo. an instant camera picture."
```

### Adjust Confidence Threshold

More strict detection (fewer false positives):
```bash
python main.py -i input -o output --confidence-threshold 0.25
```

More lenient detection (fewer missed photos):
```bash
python main.py -i input -o output --confidence-threshold 0.10
```

### Process a Subset of Images

```bash
python main.py -i input -o output --sample-size 10
```

## How It Works

### Detection Process

1. **AI-Powered Detection**: Uses the Grounded Object Detection model from Hugging Face
2. **Text Prompting**: Leverages natural language prompts to find photos in scanned images
3. **Confidence Filtering**: Removes low-confidence detections to reduce false positives
4. **Overlap Handling**: Identifies and filters overlapping detections to avoid duplicates

### Photo Extraction

1. **Smart Cropping**: Precisely extracts each detected photo as a separate image
2. **Metadata Enhancement**: Sets EXIF date data based on folder names (e.g., "1979")
3. **Quality Preservation**: Saves high-quality JPG files with original color profiles

### Year Detection

1. **Automatic Year Extraction**: Detects years from folder names (e.g., "1979")
2. **Smart Dating**: Creates sequential dates within the year for multiple photos
3. **Comprehensive Metadata**: Sets dates in multiple formats for maximum compatibility

## Output Structure

```
output/
  ├── crops/             # Extracted photos
  │   ├── file1_001.jpg
  │   ├── file1_002.jpg
  │   └── ...
  └── visualizations/    # Detection visualizations
      ├── file1_visualization.jpg
      └── ...
```

With `--preserve-structure` enabled, the subfolder structure from the input directory will be maintained.

## Best Practices

1. **Folder Organization**: Name folders with years when possible (e.g., "1979") for automatic EXIF dating
2. **Scan Quality**: Higher quality scans produce better detection results
3. **Custom Prompts**: Use specific text prompts to improve detection for challenging cases
4. **Confidence Tuning**: Adjust confidence threshold based on your specific scans and needs

## Troubleshooting

### Common Issues

- **No detections**: Try lowering the confidence threshold (e.g., `--confidence-threshold 0.10`)
- **Too many false positives**: Increase the confidence threshold (e.g., `--confidence-threshold 0.25`)
- **Overlapping detections**: Use the `--remove-overlaps` flag
- **Specific photo types not detected**: Customize the text prompt (e.g., for Polaroids: `--text-prompt "a polaroid photo."`)

## Contributing

Contributions are welcome! Check out the [Contributing Guide](CONTRIBUTING.md) for details on how to:

- Submit bug reports and feature requests
- Set up the development environment
- Submit pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - For the object detection model
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - For model access
- [Rich](https://github.com/Textualize/rich) - For beautiful terminal output
- All contributors and users of NerdScan

- All contributors and users of NerdScan
