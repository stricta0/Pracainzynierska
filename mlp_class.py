# train_mlp.py
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from download_datasets import Datasets_Menadger
from txt_loger import Loger

class MLP_Whole:
    def __init__(self, dataset_name="mnist", hidden=100, batch_size=128, epochs=10000, patience=1000, lr=0.001, seed=42, momentum=0.95, log_file="wyniki_bez_nazwy.txt", cpu=None):
        self.dataset_name = dataset_name
        self.hidden = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.momentum = momentum
        self.log_file_name= log_file
        self.cpu = cpu
        self.seed = seed

        self.args_dict = {
            "dataset": self.dataset_name, "hidden": self.hidden, "batch_size": self.batch_size, "epochs": self.epochs,
            "patience": self.patience, "lr": self.lr, "seed": self.seed, "momentum": self.momentum, "log_file": self.log_file_name, "cpu": self.cpu,
        }

        self.loger = Loger(file_name=self.log_file_name)
        self.datasets_menadger = Datasets_Menadger()

    # ---------- Model ----------
    class MLP(nn.Module):
        def __init__(self, input_dim: int = 28 * 28, hidden: int = 128, num_classes: int = 10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, num_classes)

        def forward(self, x):
            # x: (B, 1, 28, 28) -> (B, 784)
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            logits = self.fc2(x)
            return logits

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def evaluate(self, model, loader, device):
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

    def train_model(self):
        self.set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and not self.cpu else "cpu")
        self.loger.add_line_to_file(f"Using device: {device}, type: {device.type}")
        train_loader, test_loader, num_classes = self.datasets_menadger.get_loaders(dataset_name=self.dataset_name, batch_size=self.batch_size)
        model = self.MLP(hidden=self.hidden, num_classes=num_classes).to(device)
        self.loger.add_line_to_file(f"Model: MLP(hidden={self.hidden}, num_classes={num_classes})")
        self.loger.add_line_to_file(self.loger.get_args_log_line(self.args_dict))

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()

        os.makedirs("checkpoints", exist_ok=True)
        best_acc = 0.0
        epoch_without_improvement = 0
        best_train_acc = 0.0
        best_train_loss = 100000.0
        best_epoch = 0
        for epoch in range(1, self.epochs + 1):
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
            val_loss, val_acc = self.evaluate(model, test_loader, device)
            is_interpolation = (correct == total) and (train_loss < 1e-6)
            self.loger.add_line_to_file(
                f"[Epoch {epoch:02d}] train_loss={train_loss:7f} | train_acc={train_acc * 100:7f}% | train_correct={correct} | train_total={total} | val_loss={val_loss:.7f} | val_acc={val_acc * 100:.7f}% | interpolacja={is_interpolation} | soft_intepolacja={correct == total}",
                )

            epoch_without_improvement += 1
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                epoch_without_improvement = 0
            if val_acc > best_acc:
                epoch_without_improvement = 0
                best_acc = val_acc
                best_epoch = epoch
                ckpt_path = f"checkpoints/mlp_{self.dataset_name}_h{self.hidden}.pt"
                torch.save({"model_state": model.state_dict(),
                            "hidden": self.hidden,
                            "num_classes": num_classes}, ckpt_path)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epoch_without_improvement = 0
            if epoch_without_improvement > self.patience:
                break
        self.loger.add_line_to_file(f"Best val_acc: {best_acc * 100:.2f}%, on epoch: {best_epoch}")

    def kompresuj(self, procent_kompresji):

# ---------- CLI ----------
if __name__ == "__main__":
    mlp = MLP_Whole(dataset_name="mnist", hidden=100, batch_size=128, epochs=10000, patience=1000, lr=0.001, seed=42, momentum=0.95, log_file="wyniki_bez_nazwy.txt", cpu=None)
    mlp.train_model()
