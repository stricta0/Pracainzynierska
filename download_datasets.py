# download_datasets.py
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Datasets_Menadger:
    def __init__(self, dataset_path="/home/miku/PycharmProjects/Pracainzynierska/data"):
        self.dataset_path = dataset_path

    def download_mnist(self):
        torchvision.datasets.MNIST(
            root=self.dataset_path, train=True, download=True, transform=transforms.ToTensor()
        )
        torchvision.datasets.MNIST(
            root=self.dataset_path, train=False, download=True, transform=transforms.ToTensor()
        )
        print("MNIST downloaded.")

    def download_fashion_mnist(self):
        torchvision.datasets.FashionMNIST(
            root=self.dataset_path, train=True, download=True, transform=transforms.ToTensor()
        )
        torchvision.datasets.FashionMNIST(
            root=self.dataset_path, train=False, download=True, transform=transforms.ToTensor()
        )
        print("Fashion-MNIST downloaded.")

    def download_emnist_balanced(self):
        torchvision.datasets.EMNIST(
            root=self.dataset_path, split="balanced", train=True, download=True, transform=transforms.ToTensor()
        )
        torchvision.datasets.EMNIST(
            root=self.dataset_path, split="balanced", train=False, download=True, transform=transforms.ToTensor()
        )
        print("EMNIST Balanced downloaded.")

    def download_emnist_byclass(self):
        torchvision.datasets.EMNIST(
            root=self.dataset_path, split="byclass", train=True, download=True, transform=transforms.ToTensor()
        )
        torchvision.datasets.EMNIST(
            root=self.dataset_path, split="byclass", train=False, download=True, transform=transforms.ToTensor()
        )
        print("EMNIST ByClass downloaded.")

    def download_all(self):
        self.download_mnist()
        self.download_fashion_mnist()
        self.download_emnist_balanced()
        self.download_emnist_byclass()

    def get_loaders(
            self,
            dataset_name: str,
            data_root: str = None,
            batch_size: int = 128,
            num_workers: int = 2,
            pin_memory: bool = True,
            download: bool = False,
    ):
        """
        Zwraca: (train_loader, test_loader, num_classes)
        - data_root: ścieżka do katalogu z danymi (może być absolutna)
        - download=True pobierze dane, jeśli ich nie ma
        """
        if data_root is None:
            data_root = self.dataset_path

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        if dataset_name == "mnist":
            train_ds = torchvision.datasets.MNIST(data_root, train=True, download=download, transform=transform)
            test_ds = torchvision.datasets.MNIST(data_root, train=False, download=download, transform=transform)
            num_classes = 10

        elif dataset_name == "fashion":
            train_ds = torchvision.datasets.FashionMNIST(data_root, train=True, download=download, transform=transform)
            test_ds = torchvision.datasets.FashionMNIST(data_root, train=False, download=download, transform=transform)
            num_classes = 10

        elif dataset_name == "emnist_balanced":
            train_ds = torchvision.datasets.EMNIST(data_root, split="balanced", train=True, download=download,
                                                   transform=transform)
            test_ds = torchvision.datasets.EMNIST(data_root, split="balanced", train=False, download=download,
                                                  transform=transform)
            num_classes = 47

        elif dataset_name == "emnist_byclass":
            train_ds = torchvision.datasets.EMNIST(data_root, split="byclass", train=True, download=download,
                                                   transform=transform)
            test_ds = torchvision.datasets.EMNIST(data_root, split="byclass", train=False, download=download,
                                                  transform=transform)
            num_classes = 62

        else:
            raise ValueError("dataset_name must be: mnist | fashion | emnist_balanced | emnist_byclass")

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader, num_classes


if __name__ == "__main__":
    data_downolader = Datasets_Menadger()
    #data_downolader.download_all()
    print("trzeba wywolac download_all jak chcesz wszystko pobrac")
