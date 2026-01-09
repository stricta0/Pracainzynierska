# MLP_Whole.py
# Podklasa MyModel – dwuwarstwowy MLP (Linear → ReLU → Linear)
# Zawiera przykładowe uruchomienia treningu i kompresji w __main__.

from __future__ import annotations

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from myModel import MyModel, TrainConfig


class MLP1_ByMyModel(MyModel):
    class Net(nn.Module):
        def __init__(self, in_dim: int = 28 * 28, hidden: int = 128, num_classes: int = 10):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, num_classes)
        def forward(self, x):
            # x: (B, 1, 28, 28) → (B, 784)
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    def __init__(self, cfg: TrainConfig, hidden: int = 128):
        self.hidden = hidden
        super().__init__(cfg)

    # wymagane przez MyModel
    def build_model(self) -> nn.Module:
        return MLP1_ByMyModel.Net(hidden=self.hidden, num_classes=self.num_classes)

    # uzupełnienia do logów 1:1
    def extra_log_context(self) -> str:
        return f"hidden={self.hidden}"

    def model_desc(self) -> str:
        return f"MLP(hidden={self.hidden}, num_classes={self.num_classes})"


# ─────────────────────────────────────────────────────────────────────────────
# Przykładowe użycie z CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP_Whole – trening i kompresja iteracyjna")

    # — Hiperparametry i IO —
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=6000, type=int)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_file", default="mlp_train.txt", type=str)
    parser.add_argument("--cpu", action="store_true", help="wymuś CPU")
    parser.add_argument("--model_name", default="MLP", type=str)

    # — Kompresja —
    parser.add_argument("--compress", action="store_true", help="wykonaj kompresję iteracyjną")
    parser.add_argument("--use_cache", action="store_true", help="wykorzystaj cashowanie na gpu")
    parser.add_argument("--C", default=90.0, type=float, help="docelowy łączny stopień kompresji w % (np. 90)")
    parser.add_argument("--step", default=10.0, type=float, help="rozmiar pojedynczego kroku w % z pozostałych (np. 10)")
    parser.add_argument("--log_dir", default="kompresja_mlp", type=str)

    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        momentum=args.momentum,
        seed=args.seed,
        log_file=args.log_file,
        cpu=True if args.cpu else None,
        model_name=args.model_name,
        use_cache=True if args.use_cache else False,
    )
    print("inicjalizacja modelu")
    model = MLP1_ByMyModel(cfg, hidden=args.hidden)

    # Przykład 1: sam trening (logi w args.log_file)


    # Przykład 2: (opcjonalnie) kompresja iteracyjna z logami do katalogu
    if args.compress:
        # katalog na logi kompresji
        print(f"Rozpoczynam kompresje do katalogu {args.log_dir}")
        steps, alive_after = model.iterative_compression(
            total_compression_goal=args.C,
            step_size=args.step,
            log_dir=args.log_dir,
            include_bias_report=True,
        )
        print(f"Kompresja: kroki={steps}, alive_after={alive_after:.4f}%")
    else:
        model.train_model()