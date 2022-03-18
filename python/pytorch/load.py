import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_dataset():
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x:torch.flatten(x))])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return mnist_trainset,mnist_testset