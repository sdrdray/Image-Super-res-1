import cv2

def load_image(path):
    """Load an image from a file path as a numpy array (BGR)."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image

def save_image(path, image):
    """Save a numpy array image (BGR) to a file path."""
    cv2.imwrite(path, image) 