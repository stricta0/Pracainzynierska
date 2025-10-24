import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ---------- Model ----------
class MLP(nn.Module):
    def __init__(self, input_dim: int = 28*28, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 784)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def load_from_checkpoint(ckpt_path: str, force_cpu: bool = False):
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(ckpt_path, map_location=device)  # oczekuje: model_state, hidden, num_classes
    hidden = int(ckpt["hidden"])
    num_classes = int(ckpt["num_classes"])
    model = MLP(hidden=hidden, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, hidden, num_classes

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

# ---------- Loadery z data_root ----------
def get_loaders(
    dataset_name: str,
    data_root: str = "./data",
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
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    if dataset_name == "mnist":
        train_ds = torchvision.datasets.MNIST(data_root, train=True,  download=download, transform=transform)
        test_ds  = torchvision.datasets.MNIST(data_root, train=False, download=download, transform=transform)
        num_classes = 10

    elif dataset_name == "fashion":
        train_ds = torchvision.datasets.FashionMNIST(data_root, train=True,  download=download, transform=transform)
        test_ds  = torchvision.datasets.FashionMNIST(data_root, train=False, download=download, transform=transform)
        num_classes = 10

    elif dataset_name == "emnist_balanced":
        train_ds = torchvision.datasets.EMNIST(data_root, split="balanced", train=True,  download=download, transform=transform)
        test_ds  = torchvision.datasets.EMNIST(data_root, split="balanced", train=False, download=download, transform=transform)
        num_classes = 47

    elif dataset_name == "emnist_byclass":
        train_ds = torchvision.datasets.EMNIST(data_root, split="byclass", train=True,  download=download, transform=transform)
        test_ds  = torchvision.datasets.EMNIST(data_root, split="byclass", train=False, download=download, transform=transform)
        num_classes = 62

    else:
        raise ValueError("dataset_name must be: mnist | fashion | emnist_balanced | emnist_byclass")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, num_classes

def test_pt(pt_path, dataset):
    model, device, h, C = load_from_checkpoint(pt_path)
    train_loader, test_loader, num_classes = get_loaders(dataset_name=dataset, data_root="/home/miku/PycharmProjects/Pracainzynierska/data")
    test_loss, test_acc = evaluate(model, test_loader, device)
    return test_loss, test_acc

#print(test_pt("/home/miku/PycharmProjects/Pracainzynierska/checkpoints/mlp_mnist_h5000.pt", "mnist"))