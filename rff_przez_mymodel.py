# rff_przez_mymodel.py
from __future__ import annotations
import argparse
import torch
import torch.nn as nn

from myModel import MyModel, TrainConfig

class RFF_ByMyModel(MyModel):
    """
    Random Fourier Features (RFF) na szkielecie MyModel – WERSJA "real 2N".

    ZGODNE z definicją w pracy:
      - φ(x; v) = e^{-i <v, x>}  (definicja zespolona)
      - Autorzy traktują H_N jako klasę REALNĄ z 2N parametrami poprzez
        wzięcie części rzeczywistej i urojonej osobno.
      - Implementujemy to wprost jako konkatenację [cos(<v,x>), sin(<v,x>)]
        (bez fazy b, bez √2), z v ~ N(0, I).

    Architektura:
      [Wejście x] --(zamrożona warstwa RFF: [cos, sin])--> [φ(x) ∈ R^{2N}]
        --(trenowalna główka Linear)--> [logity ∈ R^{num_classes}]

    Uwaga:
      - Warstwa RFF jest NIEUCZONA (W w bufferach).
      - Główka Linear jest jedyną trenowalną częścią (zgodne z "fixed first layer").
    """

    # ──────────────────────────────────────────────────────────────────────
    # 1) Warstwa RFF: losowe, nieuczone cechy Fouriera (realna reprezentacja 2N)
    # ──────────────────────────────────────────────────────────────────────
    class RFFLayer(nn.Module):
        def __init__(self, in_dim: int, n_features: int):
            """
            Args:
                in_dim     : D – liczba cech wejściowych (np. 28*28 dla 28x28)
                n_features : N – liczba wektorów v_k (później zwracamy 2N cech: [cos, sin])

            Losowanie:
                v_k ~ N(0, I_D)  (standard normal, BEZ skali/sigma)
            """
            super().__init__()
            self.in_dim = int(in_dim)
            self.n_features = int(n_features)

            # Macierz W: [N, D], v_k ~ N(0, I)
            W = torch.randn(self.n_features, self.in_dim)
            self.register_buffer("W", W)  # nie-trenowalne, ale zapisują się w checkpointach

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Zwraca realne cechy RFF: [cos(<v,x>), sin(<v,x>)] ∈ R^{2N}
            Bez fazy b, bez czynnika √2, dokładnie jak "Re/Im osobno" w definicji.
            """
            if x.dim() > 2:
                x = x.flatten(1)  # [B, D] # B - batch size, D - 28*28
            # W to macierz N x D, to tranformacji D x N, teraz mnożymy macież x * W.t(), a więc [B, D] * [D, N] = [B, N]
            z = x.matmul(self.W.t())         # [B, N], z = <x, v_k> # mnożenie x * W^t  (transpozycja)
            feats = torch.cat([torch.cos(z),  # [B, N]
                               torch.sin(z)], # [B, N]
                              dim=1)          # [B, 2N]
            return feats


    # ──────────────────────────────────────────────────────────────────────
    # 2) Cała sieć: RFF (zamrożone) + główka Linear (trenowalna)
    # ──────────────────────────────────────────────────────────────────────
    class Net(nn.Module):
        def __init__(self, in_dim: int, n_features: int, num_classes: int):
            """
            Args:
                in_dim     : D – wymiar wejścia
                n_features : N – liczba wektorów v_k (warstwa zwróci 2N cech)
                num_classes: liczba klas wyjściowych (logity są realne)
            """
            super().__init__()
            self.rff = RFF_ByMyModel.RFFLayer(in_dim=in_dim, n_features=n_features)
            # Wyjście RFF ma rozmiar 2N (Re i Im osobno):
            self.head = nn.Linear(2 * n_features, num_classes, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.rff(x)       # [B, 2N], real
            logits = self.head(feats) # [B, C], real — OK dla CrossEntropyLoss
            return logits

    # ──────────────────────────────────────────────────────────────────────
    # 3) Kapsuła MyModel: konfiguracja i hooki
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        cfg: TrainConfig,
        n_features: int = 5000,
        in_dim: int = 28 * 28,
    ):
        """
        Args:
            cfg            : obiekt TrainConfig z Twojego MyModel
            n_features     : N – ile losowych kierunków v_k (cech będzie 2N)
            in_dim_override: wielkość wejścia
        """
        self.n_features = int(n_features)
        self._in_dim = in_dim
        super().__init__(cfg)  # ustawia device, seed, loadery, num_classes i wywoła build_model()

    def build_model(self) -> nn.Module:
        """Składamy sieć: (RFF zamrożone) + (Linear head trenowalna)."""
        return RFF_ByMyModel.Net(
            in_dim=self._in_dim,
            n_features=self.n_features,
            num_classes=self.num_classes,
        )

    def extra_log_context(self) -> str:
        return f"RFF(real-2N) | n_features(N)={self.n_features} | output_feats=2N={2*self.n_features}"

    def get_prunable_layers(self):
        """
        Tniemy WYŁĄCZNIE główkę Linear (RFF są nieuczone).
        """
        targets = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and name.endswith("head"):
                targets.append((m, "weight"))
        return targets


# ─────────────────────────────────────────────────────────────────────────────
# Przykładowe użycie z CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFF_ByMyModel – trening i (opcjonalnie) kompresja")

    # — Hiperparametry i IO —
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--n_features", default=20000, type=int, help="liczba kierunków v_k (N) => cech będzie 2N")

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--patience", default=1000, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_file", default="rff_train.txt", type=str)
    parser.add_argument("--cpu", action="store_true", help="wymuś CPU")
    parser.add_argument("--model_name", default="RFF", type=str)

    # — Kompresja —
    parser.add_argument("--compress", action="store_true", help="wykonaj kompresję iteracyjną")
    parser.add_argument("--use_cache", action="store_true", help="wykorzystaj cache na GPU (jeśli wspierane)")
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
        use_cache=True if args.use_cache else False,
    )

    model = RFF_ByMyModel(cfg, n_features=args.n_features)

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
