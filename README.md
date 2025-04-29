# NerdScan

A Python script to automatically detect and crop multiple photos from single scanned image files.

It processes images from an `input` directory and saves the cropped photos to an `output` directory, maintaining the original folder structure.

Optionally, if a photo's parent folder name in the `input` directory is a year (e.g., `input/1998/scan.jpg`), the script writes this year as the capture date (YYYY:01:01) into the EXIF metadata of the cropped photos.
