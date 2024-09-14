import matplotlib.pyplot as plt

import numpy as np
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

from operations.load_mnist import load_mnist
from operations.im_to_binary import im_to_binary
from operations.crop import crop_non_empty_center


training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw = load_mnist()
 
print(training_data_raw.shape)
print(training_labels_raw.shape)
print(eval_data_raw.shape)
print(eval_labels_raw.shape)
 

im = training_data_raw[2]
plt.imshow(im, cmap='gray')
plt.show()

im_binary = im_to_binary(im, threshold=10)
plt.imshow(im_binary, cmap='gray')
plt.show()

im_cropped = crop_non_empty_center(im, crop_height=19, crop_width=19)
plt.imshow(im_cropped, cmap='gray')
plt.show()

