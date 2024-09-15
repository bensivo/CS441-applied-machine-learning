import matplotlib.pyplot as plt
from skimage.transform import resize

from .im_to_binary import im_to_binary
from .im_crop import im_crop


def im_preprocess(image, height=20, width=20, threshold=50, debug=False):
    """
    Run the followign preprocessing steps on the given image:
        - Crop the image to remove empty rows and columns
        - Resize the image to the given dimensions
        - Convert the image to binary, using a threshold value

    Params:
        image: np.array [H,W], image data as a numpy array
        height: int, desired height of the image
        width: int, desired width of the image
        threshold: int, threshold value for binarization
        debug: bool, if True, show intermediate steps of the preprocessing

    Returns:
        np.array [height, width], the preprocessed image
    """
    im = im_crop(image)
    if debug:
        plt.imshow(im, cmap='gray')
        plt.show()

    im = resize(im, (height, width), preserve_range=True)
    if debug:
        plt.imshow(im, cmap='gray')
        plt.show()

    im = im_to_binary(im, threshold=threshold)
    if debug:
        plt.imshow(im, cmap='gray')
        plt.show()

    return im
