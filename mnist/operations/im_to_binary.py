def im_to_binary(im, threshold):
    """
    Convert an image to binary image using thresholding.
    
    Params:
        im: np.array, input image
        threshold: int, threshold value
    
    Returns:
        np.array, input image, with all values either True or False
            True = greater than or equal to threshold
            False = less than threshold
    """
    return im >= threshold
