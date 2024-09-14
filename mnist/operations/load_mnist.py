import numpy as np

def load_mnist():
    data = np.load('./mnist.npz')
    training_data_raw = data['training_data_raw']
    training_labels_raw = data['training_labels_raw']
    eval_data_raw = data['eval_data_raw']
    eval_labels_raw = data['eval_labels_raw']

    return training_data_raw, training_labels_raw, eval_data_raw, eval_labels_raw
