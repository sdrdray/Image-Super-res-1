import argparse
from utils import load_image, save_image
import cv2


def denoise_image(image):
    """Apply Non-local Means Denoising to a color image."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def main():
    parser = argparse.ArgumentParser(description="Denoise an image using Non-local Means Denoising.")
    parser.add_argument('input', help='Path to the input (noisy) image')
    parser.add_argument('output', help='Path to save the denoised image')
    args = parser.parse_args()

    image = load_image(args.input)
    denoised = denoise_image(image)
    save_image(args.output, denoised)
    print(f"Denoised image saved to {args.output}")


if __name__ == "__main__":
    main() 