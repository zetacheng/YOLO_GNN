import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# In data.py
class CIFAR10DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_loaders(self):
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                       download=True, transform=self.transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                      download=True, transform=self.transform_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=2)
        return train_loader, test_loader
