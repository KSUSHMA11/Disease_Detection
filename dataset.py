import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Creates and returns train, validation, and test dataloaders.
    Assumes standard directory structure: data_dir/train, data_dir/val, data_dir/test
    """
    
    # Standard PyTorch normalization for ImageNet pre-trained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Data augmentation for training to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation and testing only require resizing and normalization
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Specify paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # If standard split doesn't exist, we fallback to just one main directory using subsets
    if not os.path.exists(train_dir):
        print(f"Warning: Standard train/val split not found at {data_dir}. Using generic ImageFolder without split.")
        dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
        return {"all": DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)}, dataset.classes

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    
    # Handle optional test split
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = None

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    if test_loader:
        dataloaders['test'] = test_loader

    # Classes
    class_names = train_dataset.classes
    
    return dataloaders, class_names
