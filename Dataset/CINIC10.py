from torchvision.datasets import ImageFolder
import os


class CINIC10:
    """
        CINIC-10 Dataset.

    Args:
        root (string): Root directory of dataset where train, valid, and test folders exist.
        split (string): 'train', 'valid', or 'test' to specify the dataset split.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = ImageFolder(self.root, transform=self.transform, target_transform=self.target_transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

