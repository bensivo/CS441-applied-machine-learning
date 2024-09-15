import numpy as np
from sklearn.naive_bayes import GaussianNB

from operations.load_mnist import load_mnist
from operations.im_preprocess import im_preprocess

# Load the MNIST dataset, as downloaded from download.py
#
# Data is stored in 4 numpy arrays:
#  - training_data_raw: [N,H,W], training images, as uint8 numpy array (0-255)
#  - training_labels_raw: [N], training image labels
#  - eval_data_raw: [N,H,W], eval images as uint8 numpy array (0-255)
#  - eval_labels_raw: [N], eval image labels
print('Loading MNIST data')
training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw = load_mnist()

# Run pre-processing on all images:
#   - Crop the image to remove empty rows and columns
#   - Resize the image to 20x20
#   - Convert the image to binary, using a threshold value
print('Pre-processing images')
width = 20
height = 20
binary_threshold = 50

training_data_processed = np.zeros((training_data_raw.shape[0], height, width))
eval_data_processed = np.zeros((eval_data_raw.shape[0], height, width))

for i in range(training_data_raw.shape[0]):
    image = training_data_raw[i]
    training_data_processed[i] = im_preprocess(
            image, 
            height=height, 
            width=width, 
            threshold=binary_threshold
    )

for i in range(eval_data_raw.shape[0]):
    image = eval_data_raw[i]
    eval_data_processed[i] = im_preprocess(
            image, 
            height=height, 
            width=width, 
            threshold=binary_threshold
    )


# Flatten the processed images into 1-dimensional feature vectors
print('Building feature vectors')
training_data_flattened = training_data_processed.reshape(training_data_processed.shape[0], -1)
eval_data_flattened = eval_data_processed.reshape(eval_data_processed.shape[0], -1)


# Fit a Naive Bayes classifier on the training data
print('Fitting Naive Bayes classifier')
classifier = GaussianNB()
classifier.fit(training_data_flattened, training_labels_raw)


# Predict the labels of the evaluation data
print('Making predictions')
predictions = classifier.predict(eval_data_flattened)


# Calculate accuracy
accuracy = np.mean(predictions == eval_labels_raw)
print(f'Accuracy: {accuracy}')
