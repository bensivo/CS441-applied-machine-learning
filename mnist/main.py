import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier

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


def train_and_eval(model_type, training_data, training_labels, eval_data, eval_labels):
    """
    Train a Naive Bayes classifier on the training data and evaluate it on the eval data.

    Params:
        nb_distribution: 'gaussian' or 'bernoulli'
        training_data: [N,H,W] numpy array of training images
        training_labels: [N] numpy array of training image labels
        eval_data: [N,H,W] numpy array of eval images
        eval_labels: [N] numpy array of eval image labels

    Returns:
        accuracy: the accuracy of the classifier on the eval data
    """
    # Flatten the original images into 1-dimensional feature vectors
    training_data_flattened = training_data.reshape(training_data.shape[0], -1)
    eval_data_flattened = eval_data.reshape(eval_data.shape[0], -1)

    if model_type == 'gaussian':
        classifier = GaussianNB()
    elif model_type == 'bernoulli':
        classifier = BernoulliNB()
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,  # These params are very low, but we're optimizing for speed, not performance
            max_depth=10,
            random_state=12345
        )
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    classifier.fit(training_data_flattened, training_labels)
    predictions = classifier.predict(eval_data_flattened)
    accuracy = np.count_nonzero(predictions == eval_labels) / eval_labels.shape[0]
    return accuracy


accuracy_gaussian_raw = train_and_eval('gaussian', training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw)
accuracy_bernoulli_raw = train_and_eval('bernoulli', training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw)
accuracy_random_forest_raw = train_and_eval('random_forest', training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw)
accuracy_gaussian_processed = train_and_eval('gaussian', training_data_processed, training_labels_raw, eval_data_processed, eval_labels_raw)
accuracy_bernoulli_processed = train_and_eval('bernoulli', training_data_processed, training_labels_raw, eval_data_processed, eval_labels_raw)
accuracy_random_forest_processed = train_and_eval('random_forest', training_data_processed, training_labels_raw, eval_data_processed, eval_labels_raw)

print('Accuracy (Gaussian, raw):', accuracy_gaussian_raw)
print('Accuracy (Bernoulli, raw):', accuracy_bernoulli_raw)
print('Accuracy (Random Forest, raw):', accuracy_random_forest_raw)
print('Accuracy (Gaussian, processed):', accuracy_gaussian_processed)
print('Accuracy (Bernoulli, processed):', accuracy_bernoulli_processed)
print('Accuracy (Random Forest, processed):', accuracy_random_forest_processed)


