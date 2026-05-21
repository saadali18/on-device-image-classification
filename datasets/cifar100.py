import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Number of classes:", len(train_dataset.classes))
    print("First 10 classes:", train_dataset.classes[:10])
    print("First 20 targets:", train_dataset.targets[:20])
    print("Target min:", min(train_dataset.targets))
    print("Target max:", max(train_dataset.targets))

    return train_loader, test_loader

