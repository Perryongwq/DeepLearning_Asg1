import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# from utils import save_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           save_images=False,
                           num_workers=4,
                           pin_memory=True):

    
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=valid_size,
        random_state=random_seed,
        shuffle=True,
        stratify=train_dataset.targets
    )

    train_dataset = Subset(train_dataset, train_idx)
    valid_dataset = Subset(valid_dataset, valid_idx)

    labels = []
    for i in range(len(train_dataset)):
        labels.append(train_dataset[i][-1])
    label_counts = dict(Counter(labels))
    num_classes = len(label_counts)
    num_sample_per_class = label_counts[list(label_counts.keys())[0]]
    for key in label_counts:
        assert label_counts[key] == num_sample_per_class

    labels = []
    for i in range(len(valid_dataset)):
        labels.append(valid_dataset[i][-1])
    label_counts = dict(Counter(labels))
    num_classes = len(label_counts)
    num_sample_per_class = label_counts[list(label_counts.keys())[0]]
    for key in label_counts:
        assert label_counts[key] == num_sample_per_class

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
    )
    # normalize = transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5],
    # )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return data_loader