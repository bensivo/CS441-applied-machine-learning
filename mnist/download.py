import numpy as np
import os
import torchvision


"""
Download the MNIST dataset into a npz file
"""

should_download = not os.path.exists('mnist.npz')
training_data = torchvision.datasets.MNIST(
    'mnist',
    train=True,
    transform=None,
    target_transform=None,
    download=should_download,
)
eval_data = torchvision.datasets.MNIST(
    'mnist',
    train=False,
    transform=None,
    target_transform=None,
    download=should_download,
)

training_data_raw = training_data.data.numpy()
training_labels_raw = training_data.targets.numpy()
eval_data_raw = eval_data.data.numpy()
eval_labels_raw = eval_data.targets.numpy()

np.savez(
    './mnist.npz',        
    training_data_raw = training_data_raw,
    training_labels_raw=training_labels_raw,
    eval_data_raw=eval_data_raw,
    eval_labels_raw=eval_labels_raw,
)
