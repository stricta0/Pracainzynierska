# download_datasets.py
import torchvision
import torchvision.transforms as transforms

def download_mnist():
    torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    print("MNIST downloaded.")

def download_fashion_mnist():
    torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    print("Fashion-MNIST downloaded.")

def download_emnist_balanced():
    torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=True, download=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=False, download=True, transform=transforms.ToTensor()
    )
    print("EMNIST Balanced downloaded.")

def download_emnist_byclass():
    torchvision.datasets.EMNIST(
        root="./data", split="byclass", train=True, download=True, transform=transforms.ToTensor()
    )
    torchvision.datasets.EMNIST(
        root="./data", split="byclass", train=False, download=True, transform=transforms.ToTensor()
    )
    print("EMNIST ByClass downloaded.")

if __name__ == "__main__":
    download_mnist()
    download_fashion_mnist()
    download_emnist_balanced()
    download_emnist_byclass()
    print("✅ All datasets downloaded successfully.")
