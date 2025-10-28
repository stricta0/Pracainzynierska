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
import torch.nn.utils.prune as prune
import math

from download_datasets import Datasets_Menadger
from txt_loger import Loger

class MLP_Whole:
    # -------------------- INIT --------------------
    def __init__(self, dataset_name="mnist", hidden=100, batch_size=128, epochs=10000, patience=1000, lr=0.001, seed=42, momentum=0.95, log_file="wyniki_bez_nazwy.txt", cpu=None):
        #parametry treningu
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
        self.set_seed(self.seed)
        #slownik z argumentami - to jest do logowania
        self.args_dict = {
            "dataset": self.dataset_name, "hidden": self.hidden, "batch_size": self.batch_size, "epochs": self.epochs,
            "patience": self.patience, "lr": self.lr, "seed": self.seed, "momentum": self.momentum, "log_file": self.log_file_name, "cpu": self.cpu,
        }

        # Dodatkowe moduły (pliki w pythonie)
        self.loger = Loger(file_name=self.log_file_name)
        self.datasets_menadger = Datasets_Menadger()
        # --- do pruningu ---
        self.theta0 = {}       # kopie wag z inicjalizacji: {module: {"weight": tensor, "bias": tensor?}}
        self.prune_bias = False  # na start nie tniemy biasów (jak w wielu replikacjach LTH)

        # inicjalizacje, loadery, stale itd
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.cpu else "cpu")
        self.train_loader, self.test_loader, self.num_classes = self.datasets_menadger.get_loaders(dataset_name=self.dataset_name,
                                                                                    batch_size=self.batch_size)
        self.model = self.MLP(hidden=self.hidden, num_classes=self.num_classes).to(self.device)

    # -------------------- HELPERS --------------------
    def update_args_dict(self):
        self.args_dict = {
            "dataset": self.dataset_name, "hidden": self.hidden, "batch_size": self.batch_size, "epochs": self.epochs,
            "patience": self.patience, "lr": self.lr, "seed": self.seed, "momentum": self.momentum, "log_file": self.log_file_name, "cpu": self.cpu,
        }

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # -------------------- MODEL --------------------
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

    @torch.no_grad()
    def evaluate(self, loader=None):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        total, correct, total_loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
        return total_loss / total, correct / total

    def train_model(self):
        self.set_seed(self.seed)

        self.loger.add_line_to_file(f"Using device: {self.device}, type: {self.device.type}")
        self.loger.add_line_to_file(f"Model: MLP(hidden={self.hidden}, num_classes={self.num_classes})")
        self.update_args_dict()
        self.loger.add_line_to_file(self.loger.get_args_log_line(self.args_dict))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()

        os.makedirs("checkpoints", exist_ok=True)
        best_acc = 0.0
        epoch_without_improvement = 0
        best_train_acc = 0.0
        best_train_loss = 100000.0
        best_epoch = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running = 0.0

            correct, total = 0, 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = running / len(self.train_loader.dataset)
            train_acc = correct / total if total > 0 else 0.0
            val_loss, val_acc = self.evaluate()
            is_interpolation = (correct == total) and (train_loss < 1e-6)
            self.loger.add_line_to_file(
                f"[Epoch {epoch:02d}] train_loss={train_loss:7f} | train_acc={train_acc * 100:7f}% | train_correct={correct} | train_total={total} | val_loss={val_loss:.7f} | val_acc={val_acc * 100:.7f}% | interpolacja={is_interpolation} | soft_intepolacja={correct == total}",
                )

            #early stoping
            epoch_without_improvement += 1
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                epoch_without_improvement = 0
            if val_acc > best_acc:
                epoch_without_improvement = 0
                best_acc = val_acc
                best_epoch = epoch
                ckpt_path = f"checkpoints/mlp_{self.dataset_name}_h{self.hidden}.pt"
                torch.save({"model_state": self.model.state_dict(),
                            "hidden": self.hidden,
                            "num_classes": self.num_classes}, ckpt_path)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epoch_without_improvement = 0
            if epoch_without_improvement > self.patience:
                break

        self.loger.add_line_to_file(f"Best val_acc: {best_acc * 100:.2f}%, on epoch: {best_epoch}")

    # -------------------- THETA (zapis wag) --------------------
    def snapshot_theta0(self, include_bias = True):
        """
        Tworzy kopię „θ0” (snapshot bieżących parametrów) do self.theta0.
        Działa zarówno przed pruningiem, jak i PO (obsługuje weight_orig/bias_orig).
        PRZECHOWUJE wg NAZW paramów (state_dict-like), więc jest stabilne przy zapisie/odczycie.
        theta0 = {
             "fc1.weight": model.fc1.weight,
             "fc1.bias": model.fc1.bias,
             "fc2.weight": model.fc2.weight,
             "fc2.bias": model.fc2.bias,
        }
        """
        self.theta0 = {}  # reset
        with torch.no_grad():
            for module_name, module in self.model.named_modules():
                # interesują nas tylko warstwy z parametrem 'weight' (np. Linear)
                if hasattr(module, "weight") or hasattr(module, "weight_orig"):
                    # rozstrzygnij, skąd brać „prawdziwą” wagę (po pruningu jest weight_orig)
                    if hasattr(module, "weight_orig"):
                        w = module.weight_orig
                        key_w = f"{module_name}.weight" if module_name else "weight"
                    else:
                        w = module.weight
                        key_w = f"{module_name}.weight" if module_name else "weight"

                    self.theta0[key_w] = w.detach().clone()

                    if include_bias and (hasattr(module, "bias") or hasattr(module, "bias_orig")):
                        if hasattr(module, "bias_orig"):
                            b = module.bias_orig
                            key_b = f"{module_name}.bias" if module_name else "bias"
                        else:
                            b = module.bias
                            key_b = f"{module_name}.bias" if module_name else "bias"
                        if b is not None:
                            self.theta0[key_b] = b.detach().clone()

    @torch.no_grad()
    def reset_unpruned_to_theta0(self):
        """
        Resetuje TYLKO nieucięte parametry do wartości z self.theta0.
        - Jeżeli warstwa była prune’owana, korzysta z weight_orig/bias_orig i maski weight_mask/bias_mask.
        - Jeżeli nie była — kopiuje pełny tensor (nieużywane zwykle w iteracyjnych rundach, ale bezpieczne).
        """
        assert self.theta0 != {}, "Brak snapshotu θ0. Najpierw wywołaj snapshot_theta0()."

        for nazwa_warstwy, warstwa in self.model.named_modules():
            # --- wagi ---
            if hasattr(warstwa, "weight_orig") and hasattr(warstwa, "weight_mask"):
                #po puringu (dochodzi weight_mask)
                real_w = warstwa.weight_orig
                mask_w = warstwa.weight_mask.bool()
                key_w = f"{nazwa_warstwy}.weight" if nazwa_warstwy else "weight" #np. fc1.weight
                if key_w in self.theta0: #jesli self.theta0["fc1.weight"] istnieje
                    real_w[mask_w] = self.theta0[key_w].to(real_w.device)[mask_w] #wykonujemy kopie wag tylko tam gdzie w masce dostalismy boola (nie zostaly uciete przez puring)
                    # real_w.data[mask_w] = self.theta0[key_w].to(real_w.device)[mask_w] #wykonujemy kopie wag tylko tam gdzie w masce dostalismy boola (nie zostaly uciete przez puring) #TODO sprawdzic
            elif hasattr(warstwa, "weight"):
                # brak pruningu -> pełny reset
                key_w = f"{nazwa_warstwy}.weight" if nazwa_warstwy else "weight"
                if key_w in self.theta0:
                    warstwa.weight.data.copy_(self.theta0[key_w].to(warstwa.weight.device))

            # --- bias (jeśli istnieje) ---
            if hasattr(warstwa, "bias_orig") and hasattr(warstwa, "bias_mask") and warstwa.bias_orig is not None:
                real_b = warstwa.bias_orig
                mask_b = warstwa.bias_mask.bool()
                key_b = f"{nazwa_warstwy}.bias" if nazwa_warstwy else "bias"
                if key_b in self.theta0:
                    real_b.data[mask_b] = self.theta0[key_b].to(real_b.device)[mask_b]
            elif hasattr(warstwa, "bias") and warstwa.bias is not None:
                key_b = f"{nazwa_warstwy}.bias" if nazwa_warstwy else "bias"
                if key_b in self.theta0:
                    warstwa.bias.data.copy_(self.theta0[key_b].to(warstwa.bias.device))


    # -------------------- PURING LOGING -------------------
    def prune_report_to_str(self, model: nn.Module, include_bias: bool = False):
        """
        Loguje Pm oraz per-warstwa ile wag zostało (przez self.loger).
        [PRUNE] Pm=90.013% | fc1.w 3528000/3920000 (90.00%) | fc1.b 5000/5000 (100.00%) | fc2.w 45000/50000 (90.00%) | fc2.b 10/10 (100.00%)
        Pm - procent wag i biasów który został uciety
        fc1.w - (pierwsza warstwa) żywe_wagi / wszystkie_wagi (w %)
        fc1.b (pierwsza warstwa) żywa biasy / wszystkie biasy (w %)
        """
        alive_sum, total_sum = 0, 0
        lines = []

        for tag, warstwa in (("fc1", model.fc1), ("fc2", model.fc2)):
            a, T = self.alive_total(warstwa, "weight")
            lines.append(f"{tag}.w {a}/{T} ({100.0 * a / T:.2f}%)")
            alive_sum += a; total_sum += T

            if include_bias and (warstwa.bias is not None):
                ab, Tb = self.alive_total(warstwa, "bias")
                lines.append(f"{tag}.b {ab}/{Tb} ({100.0 * ab / Tb:.2f}%)")
                alive_sum += ab; total_sum += Tb

        Pm = 100.0 * alive_sum / max(1, total_sum)
        return f"PRUNE | Pm={Pm:.3f}% | " + " | ".join(lines)


    # -------------------- PURING --------------------
    def alive_total(self, warstwa: nn.Module, name: str) -> tuple[int, int]:
        """
        Zwraca (alive, total) dla danego modelu .
        W modelu to (ile wag żyje, ile jest wszystkich wag żywe + martwe)
        """
        m = getattr(warstwa, f"{name}_mask", None)
        if m is not None:
            return int(m.sum().item()), m.numel()
        t = getattr(warstwa, name)
        return t.numel(), t.numel()


    def usun_s_percent_wag_z_modelu(self, s_percent: float, include_bias = False):
        """
        Wycina dany s_percent wag z modelu z każdej warstwy (z biasami lub bez)
        """
        #asert na dobry s_percent
        s_percent = float(s_percent) / 100.0
        assert 0.0 < s_percent < 1.0, "s_percent musi być w (0, 100)."

        #ustawianie jakie wartstwy beda zmniejszane
        targetowane_warstwy_i_nazwy = [(self.model.fc1, "weight"), (self.model.fc2, "weight")]
        #dodawanie biasów warstw jeśli include_bias
        if include_bias:
            if self.model.fc1.bias is not None: targetowane_warstwy_i_nazwy.append((self.model.fc1, "bias"))
            if self.model.fc2.bias is not None: targetowane_warstwy_i_nazwy.append((self.model.fc2, "bias"))

        #przejdz sie po kazdej warstwie
        for warstwa, name in targetowane_warstwy_i_nazwy:
            alive, _ = self.alive_total(warstwa, name)
            if alive == 0:
                break
            ilosc_usuwanych_wag = max(1, int(s_percent * alive))
            prune.l1_unstructured(warstwa, name=name, amount=ilosc_usuwanych_wag)

    def kompresuj(self, s_percent: float, include_bias = False):
        """
        PRUNE (per-layer, s% z pozostałych) -> RESET(θ0 ocalałych) -> LOG(Pm).
        Wymaga: self.snapshot_theta0(model) zrobionego wcześniej.
        """
        if not getattr(self, "theta0", {}):
            raise RuntimeError("Brak θ0. Najpierw wywołaj snapshot_theta0(model).")

        # 1) prune
        self.usun_s_percent_wag_z_modelu(s_percent, include_bias=include_bias)

        # 2) reset ocalałych do θ0 (Lottery Ticket reset)
        self.reset_unpruned_to_theta0()



    def licz_kroki_do_kompresji(self, calkowity_stopien_kompresji_w_procentach: float,
                                rozmiar_kroku_w_procentach: float):
        """
        Zwraca minimalną liczbę iteracji n, aby przy cięciu s% z POZOSTAŁYCH po każdej rundzie
        osiągnąć (lub przekroczyć) docelową kompresję C%.

        Parametry:
          - calkowity_stopien_kompresji_w_procentach (C): 0 <= C < 100 (np. 90.0 oznacza 90% uciętych)
          - rozmiar_kroku_w_procentach (s): 0 < s < 100 (np. 20.0 oznacza „tnij 20% z pozostałych”)

        Zwraca:
          - n (int): liczba kroków.
        """
        calkowity_stopien_kompresji_w_procentach = float(calkowity_stopien_kompresji_w_procentach)
        rozmiar_kroku_w_procentach = float(rozmiar_kroku_w_procentach)

        if calkowity_stopien_kompresji_w_procentach <= 0:
            return 0
        if calkowity_stopien_kompresji_w_procentach >= 100:
            raise ValueError("Dokładnie 100% kompresji nie da się osiągnąć skończoną liczbą kroków.")
        if not (0.0 < rozmiar_kroku_w_procentach < 100.0):
            raise ValueError("rozmiar_kroku_w_procentach musi być w (0, 100).")

        docelowa_wartosc_procentowa = 100.0 - calkowity_stopien_kompresji_w_procentach  # ile ma zostać
        aktualna_wartosc_procentowa = 100.0                # start
        licznik = 0

        while aktualna_wartosc_procentowa > docelowa_wartosc_procentowa:
            licznik += 1
            krok_w_ulamku = rozmiar_kroku_w_procentach / 100.0
            aktualna_wartosc_procentowa = aktualna_wartosc_procentowa * (1 - krok_w_ulamku)

        return licznik, aktualna_wartosc_procentowa

    def kompresja_iteracyjna(self, calkowity_stopien_kompresji, rozmiar_kroku, log_dict_name):
        licznik, aktualna_wartosc_procentowa = self.licz_kroki_do_kompresji(calkowity_stopien_kompresji, rozmiar_kroku)
        self.snapshot_theta0()
        os.makedirs(log_dict_name, exist_ok=False)
        self.update_args_dict()
        self.loger.set_file_name(f"{log_dict_name}/kompresja.txt")
        self.loger.add_line_to_file(self.loger.get_args_log_line(self.args_dict))
        for i in range(licznik):
            self.loger.set_file_name(f"{log_dict_name}/trening_{i}.txt" )
            self.train_model()
            val_loss, val_acc = self.evaluate()
            self.loger.set_file_name(f"{log_dict_name}/kompresja.txt")
            self.loger.add_line_to_file(f"{i} | {self.prune_report_to_str(self.model, include_bias=True)} | val_loss: {val_loss} | val_acc: {val_acc}")
            self.kompresuj(rozmiar_kroku)
        self.loger.set_file_name(f"{log_dict_name}/trening_{licznik}.txt")
        self.train_model()
        val_loss, val_acc = self.evaluate()
        self.loger.set_file_name(f"{log_dict_name}/kompresja.txt")
        self.loger.add_line_to_file(f"{licznik} | {self.prune_report_to_str(self.model, include_bias=True)} | val_loss: {val_loss} | val_acc: {val_acc}")
        return licznik, aktualna_wartosc_procentowa

# ---------- CLI ----------
if __name__ == "__main__":
    mlp = MLP_Whole(dataset_name="mnist", hidden=5000, batch_size=128, epochs=1000, patience=100, lr=0.001, seed=42, momentum=0.95, log_file="z_snapshotem3.txt", cpu=None)
    print(mlp.kompresja_iteracyjna(90, 4, "kompresja_iteracyjna_wyniki_2"))
    print(mlp.kompresja_iteracyjna(90, 4, "kompresja_iteracyjna_wyniki_3"))
    print(mlp.kompresja_iteracyjna(90, 4, "kompresja_iteracyjna_wyniki_4"))
    print(mlp.kompresja_iteracyjna(90, 4, "kompresja_iteracyjna_wyniki_5"))
