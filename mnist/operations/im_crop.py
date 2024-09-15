import matplotlib.pyplot as plt
import numpy as np

def get_non_empty_bounds(image):
    """
    Given an image, determine the bounding-box that includes all non-empty pixels.

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

def im_crop(image):
    """
    Given an image (as a np array), crop it so that all empty rows and cols at the edges are removed.
    Like str.strip(), but for a numpy arr.

    Params:
        image: np.array [H,W], image data as a numpy array
        crop_height: int, desired height of the crop
        crop_width: int, desired width of the crop

    Returns:
        np.array [crop_height, crop_width], the cropped image
    """
    bounds = get_non_empty_bounds(image)
    min_row, max_row, min_col, max_col = bounds

    cropped_image = image[min_row:max_row+1, min_col:max_col+1]
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

    cropped = crop_non_empty(original)

    expected = np.array([
        [1,1,0,],
        [1,0,0,],
        [1,0,1,],
    ])

    assert np.array_equal(cropped, expected)
    print('Passed')

