# train_rrf.py
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# ---------- Utils ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Data ----------
def get_loaders(dataset_name: str, batch_size: int = 128):
    # Jednolita normalizacja (wystarczająca na start)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    if dataset_name == "mnist":
        train_ds = torchvision.datasets.MNIST("./data", train=True, download=False, transform=transform)
        test_ds  = torchvision.datasets.MNIST("./data", train=False, download=False, transform=transform)
        num_classes = 10

    elif dataset_name == "fashion":
        train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=False, transform=transform)
        test_ds  = torchvision.datasets.FashionMNIST("./data", train=False, download=False, transform=transform)
        num_classes = 10

    elif dataset_name == "emnist_balanced":
        train_ds = torchvision.datasets.EMNIST("./data", split="balanced", train=True, download=False, transform=transform)
        test_ds  = torchvision.datasets.EMNIST("./data", split="balanced", train=False, download=False, transform=transform)
        num_classes = 47

    elif dataset_name == "emnist_byclass":
        train_ds = torchvision.datasets.EMNIST("./data", split="byclass", train=True, download=False, transform=transform)
        test_ds  = torchvision.datasets.EMNIST("./data", split="byclass", train=False, download=False, transform=transform)
        num_classes = 62

    else:
        raise ValueError("dataset_name must be one of: mnist, fashion, emnist_balanced, emnist_byclass")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, num_classes


# ---------- Model ----------
class RandomReLUFeatures(nn.Module):
    """
    Random ReLU Features:
      - fc1: losowa, ZAMROŻONA projekcja do przestrzeni cech (ReLU)
      - fc2: trenowana warstwa wyjściowa do klasyfikacji
    """
    def __init__(self, input_dim: int = 28*28, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        # Losowa, zamrożona projekcja
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        with torch.no_grad():
            # He-like init dla ReLU; skala 1/sqrt(in) utrzymuje stabilność
            nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0 / np.sqrt(input_dim))
            nn.init.normal_(self.fc1.bias,   mean=0.0, std=1.0)
        for p in self.fc1.parameters():
            p.requires_grad = False  # zamrożenie

        # Trenowana klasyfikacja na cechach ReLU
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 784)
        x = x.flatten(1)
        # Losowe cechy + ReLU (bez uczenia fc1)
        with torch.set_grad_enabled(self.fc1.weight.requires_grad):
            x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits


def add_line_to_file(line, file_name="wyniki.txt"):
    """
    Dopisuje pojedynczą linię tekstu do pliku.
    Jeśli plik nie istnieje, zostanie utworzony automatycznie.

    Args:
        line (str): Tekst, który ma zostać dopisany.
        file_name (str): Nazwa pliku (domyślnie 'wyniki.txt').
    """
    if not line.endswith("\n"):
        line += "\n"
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(line)


# ---------- Train / Eval ----------
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


def train_one_run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    add_line_to_file(f"Using device: {device}, type: {device.type}", args.log_file)

    train_loader, test_loader, num_classes = get_loaders(args.dataset, args.batch_size)
    model = RandomReLUFeatures(hidden=args.hidden, num_classes=num_classes).to(device)
    add_line_to_file(f"Model: RandomReLUFeatures(hidden={args.hidden}, num_classes={num_classes})", args.log_file)

    # Trenujemy TYLKO parametry wymagające gradientu (czyli fc2)
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        momentum=float(args.momentum)
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    epoch_without_improvement = 0
    best_train_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running / len(train_loader.dataset)
        train_acc = correct / total if total > 0 else 0.0
        val_loss, val_acc = evaluate(model, test_loader, device)
        is_interpolation = (correct == total) and (train_loss < 1e-6)
        add_line_to_file(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:7f} | train_acc={train_acc*100:7f}% | "
            f"train_correct={correct} | train_total={total} | "
            f"val_loss={val_loss:.7f} | val_acc={val_acc*100:.7f}% | "
            f"interpolacja={is_interpolation} | soft_intepolacja={correct == total}",
            args.log_file
        )

        epoch_without_improvement += 1
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            epoch_without_improvement = 0
        if val_acc > best_acc:
            epoch_without_improvement = 0
            best_acc = val_acc
            ckpt_path = f"checkpoints/rrf_{args.dataset}_h{args.hidden}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": args.hidden,
                    "num_classes": num_classes,
                },
                ckpt_path
            )
        if epoch_without_improvement > args.patience:
            break

    add_line_to_file(f"Best val_acc: {best_acc*100:.2f}%", args.log_file)


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random ReLU Features on MNIST/Fashion/EMNIST")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion", "emnist_balanced", "emnist_byclass"])
    parser.add_argument("--hidden", type=int, default=100, help="neurons in hidden layer (random features)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=int, default=0.95)  # zachowuję te same argumenty
    parser.add_argument("--log_file", type=str, default="wyniki_hiden_100")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    train_one_run(args)
