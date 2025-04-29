# NerdScan

A Python script to automatically detect and crop multiple photos from single scanned image files.

It processes images from an `input` directory and saves the cropped photos to an `output` directory, maintaining the original folder structure.

Optionally, if a photo's parent folder name in the `input` directory is a year (e.g., `input/1998/scan.jpg`), the script writes this year as the capture date (YYYY:01:01) into the EXIF metadata of the cropped photos.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd NerdScan
    ```

2.  **Create a virtual environment using uv:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows (Command Prompt):
        ```bash
        .venv\Scripts\activate.bat
        ```
    *   On Windows (PowerShell):
        ```bash
        .venv\Scripts\Activate.ps1
        ```

4.  **Install the required dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

1.  Place your scanned images inside the `input` directory. You can organize them into subdirectories (e.g., `input/1998/scan1.jpg`, `input/misc/scan2.png`).
2.  Run the script:
    ```bash
    uv run python scan_divider.py
    ```
3.  The cropped photos will be saved in the `output` directory, mirroring the structure of the `input` directory. If a parent folder in `input` is named with a year, that year will be added to the EXIF metadata of the corresponding cropped images.
