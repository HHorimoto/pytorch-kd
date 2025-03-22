import torch
from torchvision import datasets

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomCrop, Normalize 
from PIL import Image
import pathlib

from src.utils.seeds import worker_init_fn

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./data/', is_train=True, download=True):
        if is_train:
            self.transform = Compose([
                RandomHorizontalFlip(),
                RandomCrop(32, padding=4),
                ToTensor(),
                Normalize(0.5, 0.5), 
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(0.5, 0.5), 
            ])

        self.data = datasets.CIFAR10(root=root, train=is_train, download=download, transform=self.transform)

        self.labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index][0]
        label = self.data[index][1]
        return image, label

def create_dataset(root='./data/', download=True, batch_size=64):

    train_dataset = CIFAR10Dataset(root=root, is_train=True, download=download)
    test_dataset = CIFAR10Dataset(root=root, is_train=False, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader