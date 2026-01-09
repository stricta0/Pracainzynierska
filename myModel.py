from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from txt_logger import Logger
from download_datasets import DatasetsManager


@dataclass
class TrainConfig:
    dataset_name: str = "mnist"
    batch_size: int = 128
    epochs: int = 10_000
    patience: int = 1_000
    lr: float = 1e-3
    momentum: float = 0.95
    seed: int = 42
    log_file: str = "wyniki_bez_nazwy.txt"
    cpu: Optional[bool] = None  # None → auto, True → CPU, False → CUDA (jeśli dostępne)
    model_name: str = "NoNameSpecified"
    use_cache: bool = True


class MyModel:
    """
    Klasa bazowa łącząca wspólne elementy: trenowanie, ewaluacja, LTH-pruning,
    snapshot θ0, reset nieuciętych wag oraz **LOGOWANIE identyczne z Twoim**.

    Hooki do nadpisania w podklasach:
      - build_model()  → zwróć nn.Module (wymagane)
      - get_prunable_layers() → lista (warstwa, "weight") do pruningu
      - build_optimizer(), build_criterion(), get_loaders(), extra_log_context(), model_desc()
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.loger = Logger(file_name=cfg.log_file)
        self.datasets_menadger = DatasetsManager()
        self.theta0: Dict[str, torch.Tensor] = {}
        self.prune_bias: bool = False  # czy ciąć biasy przy kompresji (domyślnie nie)

        # [CACHE] ADDED — wskaźniki cache (brak alokacji pamięci na starcie)
        self.use_gpu_cashe = cfg.use_cache
        self._X_train = self._y_train = None
        self._X_test = self._y_test = None
        self._cache_device = None  # gdzie trzymamy cache (np. "cuda:0" lub "cpu")

        # urządzenie
        if cfg.cpu is True:
            self.device = torch.device("cpu")
        elif cfg.cpu is False:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(cfg.seed)

        # loadery i liczba klas (można nadpisać w podklasie)
        self.train_loader, self.test_loader, self.num_classes = self.get_loaders(
            dataset_name=cfg.dataset_name,
            batch_size=cfg.batch_size,
        )

        # model
        self.model: nn.Module = self.build_model().to(self.device)

        if self.use_gpu_cashe:
            self.load_data_to_cache()


    # ─── Hooki do nadpisania ────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum)

    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_loaders(self, dataset_name: str, batch_size: int):
        return self.datasets_menadger.get_loaders(dataset_name=dataset_name, batch_size=batch_size)

    def get_prunable_layers(self) -> List[Tuple[nn.Module, str]]:
        # domyślnie tnij Linear/Conv2d po "weight"
        targets: List[Tuple[nn.Module, str]] = []
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                targets.append((m, "weight"))
        return targets

    def extra_log_context(self) -> str:
        # np. "hidden=20" — podklasa może nadpisać
        return ""

    def model_desc(self) -> str:
        """Opis do wiersza: "Model: ...". Podklasa może nadpisać, by uzyskać np.
        "MLP(hidden=20, num_classes=10)". Domyślnie wypisuje nazwę klasy modelu
        i liczbę klas, jeśli znana.
        """
        name = self.model.__class__.__name__
        try:
            return f"{name}(num_classes={self.num_classes})"
        except Exception:
            return name

    # --- CACHE --------------------
    # [CACHE] ADDED
    def load_data_to_cache(self,
                           device: Optional[torch.device] = None,
                           verbose: bool = False) -> None:
        """
        Zrzuca wybrane split-y do pamięci (domyślnie na self.device).
        Transformacje z Datasetu wykonują się raz, a potem iterujemy po gotowych tensorach.
        """
        dev = device if device is not None else self.device
        self._cache_device = dev

        def _stack_dataset(ds):
            X_list, y_list = [], []
            for i in range(len(ds)):
                x_i, y_i = ds[i]  # transformy wykonają się RAZ
                X_list.append(x_i)
                y_list.append(y_i)
            X = torch.stack(X_list, dim=0)  # (N, C, H, W) lub (N, 1, 28, 28)
            y = torch.tensor(y_list, dtype=torch.long)
            return X.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

        ds = self.train_loader.dataset
        self._X_train, self._y_train = _stack_dataset(ds)
        ds = self.test_loader.dataset
        self._X_test, self._y_test = _stack_dataset(ds)

        if verbose:
            msg = "[CACHE] "
            if self._X_train is not None: msg += f"train={tuple(self._X_train.shape)} "
            if self._X_test is not None: msg += f"test={tuple(self._X_test.shape)} "
            msg += f"on device: {dev}"
            self.loger.add_line_to_file(msg)

    # ─── Pomocnicze ─────────────────────────────────────────────────────────
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _args_line(self) -> str:
        parts = [
            f"dataset={self.cfg.dataset_name}",
            self.extra_log_context(),
            f"batch_size={self.cfg.batch_size}",
            f"epochs={self.cfg.epochs}",
            f"patience={self.cfg.patience}",
            f"lr={self.cfg.lr}",
            f"seed={self.cfg.seed}",
            f"momentum={self.cfg.momentum}",
            f"log_file={self.cfg.log_file}",
            f"cpu={self.cfg.cpu}",
            f"model_name={self.cfg.model_name}",
            f"use_cache={self.cfg.use_cache}",
        ]
        # usuń puste fragmenty i sklej w stylu: a=b | c=d | ...
        parts = [p for p in parts if p]
        return " | ".join(parts)

    # ─── Ewaluacja ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader=None, batch_size: int = 256) -> Tuple[float, float]:
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        criterion = self.build_criterion()

        if not self.use_gpu_cashe:
            total, correct, total_loss = 0, 0, 0.0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)
            return total_loss / max(1, total), correct / max(1, total)

        Xc = None
        Yc = None
        if loader is self.test_loader:
            Xc = self._X_test
            Yc = self._y_test
        else:
            Xc = self._X_train
            Yc = self._y_train

        N = Xc.size(0)
        bs = batch_size
        total_loss, correct = 0.0, 0
        for s in range(0, N, bs):
            xb = Xc[s:s + bs]
            yb = Yc[s:s + bs]
            logits = self.model(xb)
            loss_b = criterion(logits, yb)
            total_loss += loss_b.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        return total_loss / max(1, N), correct / max(1, N)


    # ─── Trening (logi 1:1) ─────────────────────────────────────────────────
    def train_model(self, save_ckpt: bool = True, ckpt_name: Optional[str] = None):
        self.set_seed(self.cfg.seed)

        # >>> DOKŁADNIE jak w Twoim pliku: dwie linie nagłówka + args
        self.loger.add_line_to_file(f"Using device: {self.device}, type: {self.device.type}")
        self.loger.add_line_to_file(f"Model: {self.model_desc()}")
        self.loger.add_line_to_file(self._args_line())

        optimizer = self.build_optimizer()
        criterion = self.build_criterion()

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_train_acc = -1.0
        best_train_loss = float("inf")
        epoch_without_improvement = 0
        best_epoch = 0

        if ckpt_name is None:
            ckpt_name = f"checkpoints/{self.__class__.__name__.lower()}_{self.cfg.dataset_name}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        best_state_dict = None
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct, total = 0, 0

            if self.use_gpu_cashe:
                # [CACHE] ADDED — trening bez DataLoadera (zero H2D w trakcie)
                bs = int(self.cfg.batch_size)
                idx = torch.arange(self._X_train.size(0), device=self._X_train.device)
                perm = idx[torch.randperm(idx.numel(), device=idx.device)]
                for s in range(0, self._X_train.size(0), bs):
                    b = perm[s:s + bs]
                    x = self._X_train[b]  # już na GPU (lub gdziekolwiek jest cache)
                    y = self._y_train[b]

                    logits = self.model(x)
                    loss = criterion(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.detach().item() * x.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            else:
                # oryginalna ścieżka z DataLoaderem — BEZ ZMIAN
                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = criterion(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * x.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            # [CACHE] CHANGED — ujednolicenie licznika; przy cache 'total' to faktyczna liczba próbek
            denom = max(1, total if total > 0 else len(self.train_loader.dataset))
            train_loss = running_loss / denom
            train_acc = correct / max(1, total)

            val_loss, val_acc = self.evaluate()

            is_interpolation = (correct == total) and (train_loss < 1e-6)
            self.loger.add_line_to_file(
                f"[Epoch {epoch:02d}] "
                f"train_loss={train_loss:.6f} | "
                f"train_acc={train_acc*100:.6f}% | "
                f"train_correct={correct} | train_total={total} | "
                f"val_loss={val_loss:.7f} | val_acc={val_acc*100:.7f}% | "
                f"interpolacja={is_interpolation} | soft_intepolacja={correct == total}"
            )

            improved = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                improved = True
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                best_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if save_ckpt:
                    torch.save({
                        "model_state": self.model.state_dict(),
                        "num_classes": self.num_classes,
                    }, ckpt_name)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                improved = True
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                improved = True

            if improved:
                epoch_without_improvement = 0
            else:
                epoch_without_improvement += 1
                if epoch_without_improvement > self.cfg.patience:
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.loger.add_line_to_file(f"Best val_acc: {best_val_acc*100:.2f}%, on epoch: {best_epoch}")

    # ─── Snapshot θ0 ────────────────────────────────────────────────────────
    def snapshot_theta0(self, include_bias: bool = True):
        self.theta0 = {}
        with torch.no_grad():
            for module_name, module in self.model.named_modules():
                # weight
                if hasattr(module, "weight") or hasattr(module, "weight_orig"):
                    w = getattr(module, "weight_orig", None)
                    if w is None:
                        w = getattr(module, "weight", None)
                    if w is not None:
                        key_w = f"{module_name}.weight" if module_name else "weight"
                        self.theta0[key_w] = w.detach().clone()
                # bias
                if include_bias and (hasattr(module, "bias") or hasattr(module, "bias_orig")):
                    b = getattr(module, "bias_orig", None)
                    if b is None:
                        b = getattr(module, "bias", None)
                    if b is not None:
                        key_b = f"{module_name}.bias" if module_name else "bias"
                        self.theta0[key_b] = b.detach().clone()

    @torch.no_grad()
    def reset_unpruned_to_theta0(self):
        assert self.theta0, "Brak snapshotu θ0. Najpierw wywołaj snapshot_theta0()."
        for name, layer in self.model.named_modules():
            # weights
            if hasattr(layer, "weight_orig") and hasattr(layer, "weight_mask"):
                real_w = layer.weight_orig
                mask_w = layer.weight_mask.bool()
                key_w = f"{name}.weight" if name else "weight"
                if key_w in self.theta0:
                    real_w[mask_w] = self.theta0[key_w].to(real_w.device)[mask_w]
            elif hasattr(layer, "weight"):
                key_w = f"{name}.weight" if name else "weight"
                if key_w in self.theta0:
                    layer.weight.data.copy_(self.theta0[key_w].to(layer.weight.device))
            # bias
            if hasattr(layer, "bias_orig") and hasattr(layer, "bias_mask") and layer.bias_orig is not None:
                real_b = layer.bias_orig
                mask_b = layer.bias_mask.bool()
                key_b = f"{name}.bias" if name else "bias"
                if key_b in self.theta0:
                    real_b[mask_b] = self.theta0[key_b].to(real_b.device)[mask_b]
            elif hasattr(layer, "bias") and layer.bias is not None:
                key_b = f"{name}.bias" if name else "bias"
                if key_b in self.theta0:
                    layer.bias.data.copy_(self.theta0[key_b].to(layer.bias.device))

    # ─── Pomiar żywych/całkowitych ──────────────────────────────────────────
    def alive_total(self, layer: nn.Module, name: str) -> Tuple[int, int]:
        m = getattr(layer, f"{name}_mask", None)
        if m is not None:
            return int(m.sum().item()), m.numel()
        p = getattr(layer, name)
        return p.numel(), p.numel()

    # ─── Raport pruningu (format 1:1: "PRUN | fc1.w ... | fc1.b ... | ...") ─
    def prune_report_to_str(self, include_bias: bool = True) -> str:
        parts: List[str] = []
        # kolejność wg nazw atrybutów, by w MLP mieć fc1 → fc2
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and name:
                # weight
                a_w, T_w = self.alive_total(layer, "weight")
                parts.append(f"{name}.w {a_w}/{T_w} ({100.0*a_w/max(1,T_w):.2f}%)")
                # bias
                if include_bias and (layer.bias is not None):
                    a_b, T_b = self.alive_total(layer, "bias")
                    parts.append(f"{name}.b {a_b}/{T_b} ({100.0*a_b/max(1,T_b):.2f}%)")
        return "PRUN | " + " | ".join(parts)

    # ─── Pruning i reset ────────────────────────────────────────────────────
    def delete_s_percent_of_weights(self, s_percent: float, include_bias: bool = False):
        s = float(s_percent) / 100.0
        assert 0.0 < s < 1.0, "s_percent musi być w (0, 100)."

        targets = self.get_prunable_layers()
        if include_bias:
            for m, _ in list(targets):
                if getattr(m, "bias", None) is not None:
                    targets.append((m, "bias"))

        for layer, pname in targets:
            alive, _ = self.alive_total(layer, pname)
            if alive <= 0:
                continue
            k = max(1, int(s * alive))
            prune.l1_unstructured(layer, name=pname, amount=k)

    def compress(self, s_percent: float, include_bias: bool = False):
        if not self.theta0:
            raise RuntimeError("Brak θ0. Najpierw wywołaj snapshot_theta0().")
        self.delete_s_percent_of_weights(s_percent, include_bias=include_bias)
        self.reset_unpruned_to_theta0()

    # ─── Oblicz liczbę kroków do kompresji ──────────────────────────────────
    def count_steps_for_iterative_compression(self, total_compression_goal: float,
                                              step_size: float) -> Tuple[int, float]:
        C = float(total_compression_goal)
        s = float(step_size)
        if C <= 0:
            return 0, 100.0
        if C >= 100:
            raise ValueError("Dokładnie 100% kompresji nie da się osiągnąć skończoną liczbą kroków.")
        if not (0.0 < s < 100.0):
            raise ValueError("rozmiar_kroku_w_procentach musi być w (0, 100).")

        target_alive = 100.0 - C
        alive = 100.0
        n = 0
        step = s / 100.0
        while alive > target_alive:
            n += 1
            alive *= (1.0 - step)
        return n, alive

    # ─── Iteracyjna kompresja (logi 1:1) ────────────────────────────────────
    def iterative_compression(self, total_compression_goal: float, step_size: float,
                              log_dir: str, include_bias_report: bool = True) -> Tuple[int, float]:
        n_steps, alive_after = self.count_steps_for_iterative_compression(total_compression_goal, step_size)
        os.makedirs(log_dir, exist_ok=False)

        # Nagłówek kompresji
        self.loger.set_file_name(os.path.join(log_dir, "kompresja.txt"))
        self.loger.add_line_to_file(self._args_line() + f" | calkowity_oczekiwany_stopien_kompresji={total_compression_goal} | rozmiar_kroku={step_size}")

        # Snapshot θ0
        self.snapshot_theta0()

        # Iteracje: train → log raport (poziom z poprzedniego kroku) → prune
        for i in range(1, n_steps + 1):
            # trening
            self.loger.set_file_name(os.path.join(log_dir, f"trening_{i-1}.txt"))
            self.train_model()
            val_loss, val_acc = self.evaluate()

            # log raport dla kroku i (stan masek po poprzednim cięciu)
            self.loger.set_file_name(os.path.join(log_dir, "kompresja.txt"))
            self.loger.add_line_to_file(
                f"{i} | {self.prune_report_to_str(include_bias=include_bias_report)} | "
                f"val_loss: {val_loss:.7f} | val_acc: {val_acc:.7f}"
            )

            # prune + reset (przygotowuje stan pod kolejny cykl)
            self.compress(step_size, include_bias=self.prune_bias)

        # Końcowe domknięcie: jeszcze jeden trening po ostatnim prune
        self.loger.set_file_name(os.path.join(log_dir, f"trening_{n_steps}.txt"))
        self.train_model()
        val_loss_f, val_acc_f = self.evaluate()
        self.loger.set_file_name(os.path.join(log_dir, "kompresja.txt"))
        self.loger.add_line_to_file(
            f"{n_steps+1} | {self.prune_report_to_str(include_bias=include_bias_report)} | "
            f"val_loss: {val_loss_f:.7f} | val_acc: {val_acc_f:.7f}"
        )

        return n_steps, alive_after

