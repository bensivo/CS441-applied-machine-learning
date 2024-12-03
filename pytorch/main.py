import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

import os

from operations.download_cifar10 import download_cifar10
from operations.util import imshow
from operations.net import Net
from operations.nn_train import nn_train
from operations.nn_test import nn_test


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    trainset, trainloader, testset, testloader, classes = download_cifar10()

    nn_train(
        dataloader = trainloader,
        model_filepath = './cifar_net.pth'
    )

    nn_test(
        dataloader = testloader,
        model_filepath = './cifar_net.pth',
        classes = classes
    )


if __name__ == "__main__":
    main()
