import torch
import torchvision
import torchvision.transforms as transforms


def download_cifar10():
    """
    Download the CIFAR10 dataset from torchvision

    Returns:
        trainset: torch.utils.data.Dataset
        trainloader: torch.utils.data.DataLoader
        testset: torch.utils.data.Dataset
        testloader: torch.utils.data.DataLoader
        classes: tuple
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes