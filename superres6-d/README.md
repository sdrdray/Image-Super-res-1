# Image Denoising Tool

This project provides a simple command-line tool to denoise images using OpenCV's Non-local Means Denoising algorithm.

## Features
- Denoise color images using a state-of-the-art algorithm
- Easy to use from the command line
- Modular code for easy extension

## Installation

1. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python denoise.py input_image.jpg output_image.jpg
```
- `input_image.jpg`: Path to your noisy image
- `output_image.jpg`: Path where the denoised image will be saved

## Extending
You can add more denoising algorithms in `denoise.py` as needed. 