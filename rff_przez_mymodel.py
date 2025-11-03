# rff_przez_mymodel.py
from __future__ import annotations
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myModel import MyModel, TrainConfig

class RFF_ByMyModel(MyModel):
    """
    Random Fourier Features (RFF) na Twoim szkielecie MyModel.

    Architektura:
      [Wejście x] --(zamrożona warstwa RFF)--> [φ(x) ∈ R^N] --(trenowalna główka Linear)--> [logity]

    • Warstwa RFF (pierwsza) generuje N losowych cech:
      Wagi w_k i przesunięcia b_k są BUFORAMI (buffers) – nie uczymy ich, ale przenoszą się na GPU/CPU,
      zapisują w checkpointach.

    • Główka (druga warstwa) to Linear(N → num_classes) – JEDYNA część trenowana SGD/CE.
      Pruning dotyczy wyłącznie główki (jak w get_prunable_layers()).
    """

    # ──────────────────────────────────────────────────────────────────────
    # 1) Warstwa RFF: losowe, nieuczone cechy Fouriera
    # ──────────────────────────────────────────────────────────────────────
    class RFFLayer(nn.Module):
        def __init__(self, in_dim: int, n_features: int, sigma: float = 5.0):
            """
            Warstwa losowa, nieuczona dla RFF

            Args:
                in_dim     : liczba cech wejściowych (np. 28*28 dla obrazków 28x28)
                n_features : N – liczba losowych cech (im większe N, tym „bogatsza” aproksymacja jądra RBF)
                sigma      : parametr szerokości jądra RBF (większe sigma -> mniejsze wariancje wag)

            Losowanie:
                każdy wektor wk = [wk1, wk2, ..., wkd] jest losowany niezależnie
                każde wkj jest próbkowane z rozkładu nomrlanego o średniej 0 i odchyleniu standardowym (std) 1/sigma

            Uwaga: W i b rejestrujemy jako BUFFERS (nie trainable)
            """
            super().__init__()
            self.in_dim = int(in_dim)
            self.n_features = int(n_features)
            self.sigma = float(sigma)

            # std = 1/sigma  (bo Var = 1/sigma^2)
            std = 1.0 / max(self.sigma, 1e-12)  # zabezpieczenie przed dzieleniem przez 0
            # Macierz W: [n_features, in_dim], wektor faz b: [n_features]
            # Macierz W - losowa macierz o wymiarach self.n_features na self.in_dim z losowymi liczbami o średniej = 0 i odchyleniu = 1 pomnożona przez std
            W = torch.randn(self.n_features, self.in_dim) * std
            #torch.empty(self.n_features) - tworzy wektor długości n_features - .uniform_(0.0, 2.0 * pi) - losowanie z rozkładu jednostajnego na przedziale 0, 2pi (caly cykl cosinusa)
            b = torch.empty(self.n_features).uniform_(0.0, 2.0 * math.pi)

            # Rejestracja jako BUFFERS (nie parametry trenowalne)
            # tworzy tez self.W i self.b
            self.register_buffer("W", W)  # kształt: [N, D]
            self.register_buffer("b", b)  # kształt: [N]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            dla self.n_features = 3 i in_dim = 2
            z = [
                sqrt(2) * cos(x1*w11 + x2*w12 + b1),
                sqrt(2) * cos(x1*w21 + x2*w22 + b2),
                sqrt(2) * cos(x1*w31 + x2*w32 + b3)
                ]
            """
            z = x.matmul(self.W.t())
            z = z + self.b
            return math.sqrt(2.0) * torch.cos(z)

    # ──────────────────────────────────────────────────────────────────────
    # 2) Cała sieć: RFF (zamrożone) + główka Linear (trenowalna)
    # ──────────────────────────────────────────────────────────────────────
    class Net(nn.Module):
        def __init__(self, in_dim: int, n_features: int, num_classes: int, sigma: float = 5.0):
            """
            Args:
                in_dim     : wymiar wejścia (D)
                n_features : liczba cech RFF (N)
                num_classes: liczba klas wyjściowych
                sigma      : parametr jądra RBF
            """
            super().__init__()
            #model z 1 wartswa
            self.rff = RFF_ByMyModel.RFFLayer(in_dim=in_dim, n_features=n_features, sigma=sigma)
            # Uwaga: to JEDYNA trenowalna warstwa (jej wagi i bias będą optymalizowane)
            self.head = nn.Linear(n_features, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() > 2:
                x = x.flatten(1)
            # Stałe cechy RFF (bezgrad) + ewentualny dropout
            feats = self.rff(x)
            # Główka liniowa daje logity do CrossEntropyLoss
            logits = self.head(feats)
            return logits

    # ──────────────────────────────────────────────────────────────────────
    # 3) Kapsuła MyModel: konfiguracja i hooki
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        cfg: TrainConfig,
        n_features: int = 5000,
        sigma: float = 5.0,
        in_dim_override: int | None = None,
    ):
        """
        Args:
            cfg            : obiekt TrainConfig z Twojego MyModel
            n_features     : N – ile cech RFF generować
            sigma          : szerokość jądra (im mniejsze, tym wyższe częstotliwości)
            in_dim_override: umożliwia ręczne wymuszenie wymiaru wejścia (gdy loader
                             nie pozwala łatwo go odczytać lub dane nie są obrazkami)
        """
        self.n_features = int(n_features)
        self.sigma = float(sigma)
        self._in_dim_override = in_dim_override
        super().__init__(cfg)  # -> ustawi device, seed, loadery, num_classes i wywoła build_model()

    # --- Hooki wymagane przez MyModel ---

    def _infer_input_dim(self) -> int:
        """
        Próba odczytu wymiaru wejścia (D) z pierwszej paczki z loadera.
        Jeśli się nie uda, używamy domyślnego 28*28 (zgodne z MNIST/EMNIST).
        """
        try:
            x, _ = next(iter(self.train_loader))
            # x.shape: [B, C, H, W] lub [B, D]; bierzemy iloczyn od indeksu 1
            return int(np.prod(list(x.shape)[1:]))
        except Exception:
            return 28 * 28

    def build_model(self) -> nn.Module:
        """
        Składamy sieć: (RFF zamrożone) + (Linear head trenowalna).
        num_classes jest ustawiane w MyModel po get_loaders().
        """
        in_dim = self._in_dim_override if self._in_dim_override is not None else self._infer_input_dim()
        return RFF_ByMyModel.Net(
            in_dim=in_dim,
            n_features=self.n_features,
            num_classes=self.num_classes,
            sigma=self.sigma,
        )

    def extra_log_context(self) -> str:
        """
        Dodatkowy kontekst do linii argumentów w loggerze (pojawia się w plikach logów).
        """
        return f"RFF | n_features={self.n_features} | sigma={self.sigma}"

    def get_prunable_layers(self):
        """
        Zwracamy listę (moduł, 'weight') do cięcia. Chcemy ciąć TYLKO główkę Linear.
        Dlaczego nie tniemy RFF?
          • warstwa RFF nie ma trenowalnych wag (W, b to buffers),
          • celem jest badanie kompresji klasyfikatora nad stałymi cechami.
        """
        targets = []
        for name, m in self.model.named_modules():
            # Prunujemy wyłącznie macierz wag główki.
            # (biasów domyślnie nie tniemy — w Twoim MyModel jest to kontrolowane flagą prune_bias)
            if isinstance(m, nn.Linear) and name.endswith("head"):
                targets.append((m, "weight"))
        return targets


# ─────────────────────────────────────────────────────────────────────────────
# Przykładowe użycie z CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFF_ByMyModel – trening i kompresja iteracyjna")

    # — Hiperparametry i IO —
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--n_features", default=20000, type=int, help="liczba cech RFF (N)")
    parser.add_argument("--sigma", default=5.0, type=float, help="parametr jądra RBF (W ~ N(0, 1/sigma^2 I))")

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)        # zwykle trochę większe niż dla MLP
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_file", default="rff_train.txt", type=str)
    parser.add_argument("--cpu", action="store_true", help="wymuś CPU")
    parser.add_argument("--model_name", default="RFF", type=str)

    # — Kompresja —
    parser.add_argument("--compress", action="store_true", help="wykonaj kompresję iteracyjną")
    parser.add_argument("--C", default=60.0, type=float, help="docelowy łączny stopień kompresji w % (np. 90)")
    parser.add_argument("--step", default=10.0, type=float, help="rozmiar pojedynczego kroku w % z pozostałych (np. 10)")
    parser.add_argument("--log_dir", default="kompresja_rff", type=str)

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
    )

    model = RFF_ByMyModel(cfg, n_features=args.n_features, sigma=args.sigma)

    # Przykład 1: sam trening (logi w args.log_file)


    # Przykład 2: (opcjonalnie) kompresja iteracyjna z logami do katalogu
    if args.compress:
        steps, alive_after = model.kompresja_iteracyjna(
            calkowity_stopien_kompresji=args.C,
            rozmiar_kroku=args.step,
            log_dir=args.log_dir,
            include_bias_report=True,
        )
        print(f"Kompresja: kroki={steps}, alive_after={alive_after:.4f}%")
    else:
        model.train_model()