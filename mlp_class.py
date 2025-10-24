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

from txt_loger import Loger

class MLP_Whole:
    def __init__(self, dataset, hidden, batch_size):
        self.dataset = dataset
        self.hidden = hidden
        self.args = args
        self.log_file_name = args.log_file
        self.loger = Loger(file_name=self.log_file_name)

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

    def train_model(self, args=None):
        if args is None:
            args = self.args

        self.set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.loger.add_line_to_file(f"Using device: {device}, type: {device.type}")
        train_loader, test_loader, num_classes = get_loaders(args.dataset, args.batch_size)
        model = self.MLP(hidden=args.hidden, num_classes=num_classes).to(device)
        self.loger.add_line_to_file(f"Model: MLP(hidden={args.hidden}, num_classes={num_classes})")
        self.loger.add_line_to_file(self.loger.get_args_log_line(args))

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        criterion = nn.CrossEntropyLoss()

        os.makedirs("checkpoints", exist_ok=True)
        best_acc = 0.0
        epoch_without_improvement = 0
        best_train_acc = 0.0
        best_train_loss = 100000.0
        best_epoch = 0
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
            val_loss, val_acc = self.evaluate(model, test_loader, device)
            is_interpolation = (correct == total) and (train_loss < 1e-6)
            add_line_to_file(
                f"[Epoch {epoch:02d}] train_loss={train_loss:7f} | train_acc={train_acc * 100:7f}% | train_correct={correct} | train_total={total} | val_loss={val_loss:.7f} | val_acc={val_acc * 100:.7f}% | interpolacja={is_interpolation} | soft_intepolacja={correct == total}",
                args.log_file)

            epoch_without_improvement += 1
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                epoch_without_improvement = 0
            if val_acc > best_acc:
                epoch_without_improvement = 0
                best_acc = val_acc
                best_epoch = epoch
                ckpt_path = f"checkpoints/mlp_{args.dataset}_h{args.hidden}.pt"
                torch.save({"model_state": model.state_dict(),
                            "hidden": args.hidden,
                            "num_classes": num_classes}, ckpt_path)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epoch_without_improvement = 0
            if epoch_without_improvement > args.patience:
                break
        add_line_to_file(f"Best val_acc: {best_acc * 100:.2f}%, on epoch: {best_epoch}", args.log_file)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST/Fashion/EMNIST")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion", "emnist_balanced", "emnist_byclass"])
    parser.add_argument("--hidden", type=int, default=100, help="neurons in hidden layer")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=int, default=0.95)
    parser.add_argument("--log_file", type=str, default="wyniki_hiden_100")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    mlp = MLP_Whole(args)

    mlp.train_one_run(args)
