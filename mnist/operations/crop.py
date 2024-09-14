import matplotlib.pyplot as plt
import numpy as np

def get_non_empty_bounds(image):
    """
    Given an image, determine the bounding-box that removes all empty pixels

    Params:
        image: np.array [H,W], image data as a numpy array

    Returns:
        tuple, (min_row, max_row, min_col, max_col)
    """

    # Create boolean masks for rows and cols, True if that row/col is not empty.
    non_empty_rows = np.any(image, axis=1)
    non_empty_cols = np.any(image, axis=0)

    # Use the boolean measks to find the min/max row/col that are not empty
    min_row = np.flatnonzero(non_empty_rows)[0]
    max_row = np.flatnonzero(non_empty_rows)[-1]
    min_col = np.flatnonzero(non_empty_cols)[0]
    max_col = np.flatnonzero(non_empty_cols)[-1]

    return (min_row, max_row, min_col, max_col)

def crop_non_empty_center(image, crop_height, crop_width):
    """
    Given an image and a desired crop size, crop the image around the center
    of the non-empty pixels.

    Params:
        image: np.array [H,W], image data as a numpy array
        crop_height: int, desired height of the crop
        crop_width: int, desired width of the crop

    Returns:
        np.array [crop_height, crop_width], the cropped image
    """
    bounds = get_non_empty_bounds(image)
    min_row, max_row, min_col, max_col = bounds

    # Calculate the center of the non-empty pixels
    non_empty_row_center = ( min_row + max_row ) // 2
    non_empty_col_center = ( min_col + max_col ) // 2

    # Use np.roll to move the center of the non-empty pixels to the center of the image
    im_center_row = int(image.shape[0] / 2)
    im_center_col = int(image.shape[1] / 2)
    row_shift = im_center_row - non_empty_row_center
    col_shift = im_center_col - non_empty_col_center
    image = np.roll(image, row_shift, axis=0)
    image = np.roll(image, col_shift, axis=1)

    # Crop the image around the center, using the desired crop size
    crop_min_row = im_center_row - (crop_height // 2)
    crop_max_row = crop_min_row + crop_height
    crop_min_col = im_center_col - (crop_width // 2)
    crop_max_col = crop_min_col + crop_width

    cropped_image = image[crop_min_row:crop_max_row, crop_min_col:crop_max_col]

    return cropped_image



if __name__ == '__main__':
    original = np.array([
        [0,0,0,0,0,0,0,0,],
        [0,0,1,1,0,0,0,0,],
        [0,0,1,0,0,0,0,0,],
        [0,0,1,0,1,0,0,0,],
        [0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,],
        [0,0,0,0,0,0,0,0,],
    ])

    cropped = crop_non_empty_center(original, 7, 5)

    plt.imshow(original, cmap='gray')
    plt.show()
    plt.imshow(cropped, cmap='gray')
    plt.show()

