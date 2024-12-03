# Pytorch

An implementation of an image classifier using pytorch, implemented for CS441 in UIUC's MCS program.


## Instructions
Go through the CIFAR-10 tutorial at https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, and ensure you can run the code. Modify the architecture that is offered in the CIFAR-10 tutorial to get the best accuracy you can. Anything better than about 93.5% will be comparable with current research.

Redo the same efforts for the MNIST digit data set.


## 1. Installation
```
python3 -m venv ./venv
source ./venv/bin/activate

pip install matplotlib numpy torch torchvision
```

## 2. Download Dataset
Download the dataset from kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

This will give you a file "diabetes.csv", put it in this folder.


## 3. Run the code
```
python main.py
```
