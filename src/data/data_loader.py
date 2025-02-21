import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10_data(batch_size=128):
    """
    Load and prepare CIFAR-10 dataset with appropriate transformations.
    We add some basic data augmentation for training to improve generalization.
    """
    # Training data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Add padding and random crop for robustness
        transforms.RandomHorizontalFlip(),     # Random flipping for augmentation
        transforms.ToTensor(),
        # CIFAR-10 normalization values (pre-computed mean and std)
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Test data only needs basic transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader