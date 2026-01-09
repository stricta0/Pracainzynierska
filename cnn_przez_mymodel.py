# cnn_przez_mymodel.py
from __future__ import annotations
import argparse
import math
import numpy as np
import torch
import torch.nn as nn

from myModel import MyModel, TrainConfig  # dopasuj wielkość liter do nazwy pliku!


class CNN_ByMyModel(MyModel):
    """
    Proste CNN na szkielecie MyModel:
      [Wejście x: (B, C, H, W)]
        -> Conv2d( C -> c1, 3x3, pad=1 ) + ReLU
        -> Conv2d( c1 -> c2, 3x3, pad=1 ) + ReLU
        -> MaxPool(2x2)
        -> (opcjonalnie Dropout)
        -> Flatten
        -> Linear( c2 * (H/2) * (W/2) -> num_classes )

    Pruning: tniemy wagi konwolucji i główki liniowej (biasy wg flagi w MyModel).
    """

    # ──────────────────────────────────────────────────────────────────────
    # 1) Sieć właściwa
    # ──────────────────────────────────────────────────────────────────────
    class Net(nn.Module):
        def __init__(
            self,
            in_ch: int,
            in_h: int,
            in_w: int,
            num_classes: int,
            ch1: int = 32,
            ch2: int = 64,
        ):
            """
            Args:
                in_ch      : liczba kanałów wejścia (np. 1 dla MNIST/EMNIST)
                in_h, in_w : wysokość/szerokość wejścia (np. 28x28)
                num_classes: liczba klas
                ch1, ch2   : kanały w warstwach konwolucyjnych
            """
            super().__init__()
            # Blok konwolucyjny (zachowuje rozmiar HxW dzięki padding=1)
            self.conv1 = nn.Conv2d(in_ch, ch1, kernel_size=3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # H,W -> H/2, W/2

            # Po 2×Conv(3x3, pad=1) + MaxPool(2x2): rozmiar map cech = (ch2, H/2, W/2)
            out_h = in_h // 2
            out_w = in_w // 2
            feat_dim = ch2 * out_h * out_w

            self.head = nn.Linear(feat_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(1)  # (B, feat_dim)
            logits = self.head(x)
            return logits

    # ──────────────────────────────────────────────────────────────────────
    # 2) Kapsuła MyModel: konfiguracja i hooki
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        cfg: TrainConfig,
        ch1: int = 32,
        ch2: int = 64,
        ):
        """
        Args:
            cfg         : obiekt TrainConfig
            ch1, ch2    : liczba kanałów w Conv1/Conv2
        """
        self.ch1 = int(ch1)
        self.ch2 = int(ch2)
        super().__init__(cfg)  # -> ustawia device/seed/loadery/num_classes i zawoła build_model()

    # Pomocniczo wyczytujemy rozmiar wejścia z pierwszego batcha
    def _infer_chw(self) -> tuple[int, int, int]:
        try:
            x, _ = next(iter(self.train_loader))
            # oczekujemy (B, C, H, W) – jeśli (B, D), to założymy 1x√D x √D (fallback)
            if x.dim() >= 4:
                _, C, H, W = x.shape[:4]
                return int(C), int(H), int(W)
            else:
                D = int(np.prod(list(x.shape)[1:]))
                side = int(math.sqrt(D))
                return 1, side, side  # fallback
        except Exception:
            return 1, 28, 28  # bezpieczny fallback dla MNIST/EMNIST

    def build_model(self) -> nn.Module:
        """
        Składamy CNN:
          Conv-ReLU-Conv-ReLU-MaxPool-()-Flatten-Linear
        """
        C, H, W = self._infer_chw()
        return CNN_ByMyModel.Net(
            in_ch=C,
            in_h=H,
            in_w=W,
            num_classes=self.num_classes,
            ch1=self.ch1,
            ch2=self.ch2,
        )

    def extra_log_context(self) -> str:
        return f"ch1={self.ch1} | ch2={self.ch2}"

    def get_prunable_layers(self):
        """
        Tniemy wagi warstw konwolucyjnych i liniowych (biasy wg self.prune_bias z MyModel).
        """
        targets = []
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                targets.append((m, "weight"))
        return targets


# ─────────────────────────────────────────────────────────────────────────────
# CLI – jak w RFF/MLP: trening + (opcjonalnie) iteracyjna kompresja
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN_ByMyModel – trening i kompresja iteracyjna")

    # Hiperparametry i IO
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_file", default="cnn_train2.txt", type=str)
    parser.add_argument("--cpu", action="store_true", help="wymuś CPU")
    parser.add_argument("--model_name", default="CNN", type=str)

    # Architektura
    parser.add_argument("--ch1", default=32, type=int)
    parser.add_argument("--ch2", default=64, type=int)

    # Kompresja
    parser.add_argument("--compress", action="store_true", help="wykonaj kompresję iteracyjną")
    parser.add_argument("--use_cache", action="store_true", help="wykorzystaj cashowanie na gpu")
    parser.add_argument("--C", default=60.0, type=float, help="docelowy łączny stopień kompresji w % (np. 90)")
    parser.add_argument("--step", default=10.0, type=float, help="rozmiar pojedynczego kroku w % z pozostałych (np. 10)")
    parser.add_argument("--log_dir", default="kompresja_cnn2", type=str)

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

    model = CNN_ByMyModel(
        cfg,
        ch1=args.ch1,
        ch2=args.ch2,
    )

    # Sam trening lub trening + kompresja iteracyjna (jak w Twoim API)
    if args.compress:
        steps, alive_after = model.iterative_compression(
            total_compression_goal=args.C,
            step_size=args.step,
            log_dir=args.log_dir,
            include_bias_report=True,
        )
        print(f"Kompresja: kroki={steps}, alive_after={alive_after:.4f}%")
    else:
        model.train_model()
