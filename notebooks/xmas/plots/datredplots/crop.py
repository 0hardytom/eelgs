import numpy as np

def crop(image: np.ndarray, n_pixels: int) -> np.ndarray:
    """
    Crops a specified number of pixels from the borders of an image.

    This function removes N pixels from the start and end of both the
    height (y-axis) and width (x-axis) of the image.

    Args:
        image: A 2D (grayscale) or 3D (e.g., RGB) NumPy array.
        n_pixels: The number of pixels to crop from each side (top, bottom,
                  left, and right).

    Returns:
        A new NumPy array representing the cropped image.

    Raises:
        ValueError: If n_pixels is negative or if the cropping amount
                    is larger than the image dimensions.
    """
    if n_pixels < 0:
        raise ValueError("Number of pixels to crop (n_pixels) cannot be negative.")

    # Get image dimensions
    if image.ndim < 2:
        raise ValueError(f"Input image must be at least 2D, but has {image.ndim} dimensions.")
    height, width = image.shape[:2]

    # Check if the crop amount is valid
    if 2 * n_pixels >= height or 2 * n_pixels >= width:
        raise ValueError(
            f"Cannot crop {n_pixels} pixels from each side of an image with "
            f"shape ({height}, {width}). The crop amount is too large."
        )

    # Perform the crop using slicing.
    # The slice for the y-axis is from n_pixels to height - n_pixels.
    # The slice for the x-axis is from n_pixels to width - n_pixels.
    cropped_image = image[n_pixels : height - n_pixels, n_pixels : width - n_pixels]

    return cropped_image

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Example with a 2D (grayscale) image
    print("--- 2D Example ---")
    # Create a 5x5 array
    grayscale_image = np.array([
        [ 1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ])

    print("Original image:\n", grayscale_image)
    print("Original shape:", grayscale_image.shape)

    # Crop 1 pixel from each border
    cropped_grayscale = crop(grayscale_image, n_pixels=1)
    print("\nCropped by 1 pixel:")
    print(cropped_grayscale)
    print("Cropped shape:", cropped_grayscale.shape)

    # 2. Example with a 3D (RGB) image
    print("\n--- 3D Example ---")
    # Create a 4x4 RGB image
    rgb_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
    print("Original shape:", rgb_image.shape)

    # Crop 1 pixel from each border
    cropped_rgb = crop(rgb_image, n_pixels=1)
    print("Cropped shape:", cropped_rgb.shape)

    # 3. Example with an invalid crop amount
    print("\n--- Invalid Crop Example ---")
    try:
        crop(grayscale_image, n_pixels=3)
    except ValueError as e:
        print(f"Caught expected error: {e}")
