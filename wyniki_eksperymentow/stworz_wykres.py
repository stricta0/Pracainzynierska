from typing import Tuple, Iterable, Sequence, Mapping, Any, List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def is_int(s):
    try:
        int(s.strip())
        return True
    except ValueError:
        return False

def change_type_from_file_string(value):
    if value[-1] == "%":
        value = float(value[:-1]) / 100.0
    elif "." in value:
        value = float(value)
    elif value == "False":
        value = False
    elif value == "True":
        value = True
    elif is_int(value):
        value = int(value)
    return value

def get_all_files_paths_from_directory(
    directory_path: str,
    ignorowane_rozszerzenia: Optional[Iterable[str]] = None,
    ignore_file_names: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Zwraca listę *absolutnych* ścieżek do wszystkich plików w podanym katalogu (rekurencyjnie),
    z filtrami:
      - ignorowane_rozszerzenia: np. ["pdf", "jpg"]  -> pomija pliki o tych rozszerzeniach
      - ignore_file_names: np. ["readme.md", "wynik_koncowy.txt"] -> pomija pliki o tych nazwach
    Porównania są niewrażliwe na wielkość liter. Nazwy w ignore_file_names podaj BEZ ścieżek.

    :param directory_path: ścieżka do katalogu startowego
    :return: posortowana lista absolutnych ścieżek plików
    :raises ValueError: gdy ścieżka nie istnieje lub nie jest katalogiem
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Ścieżka '{directory_path}' nie jest katalogiem lub nie istnieje.")

    ignore_exts = {ext.lower().lstrip(".") for ext in (ignorowane_rozszerzenia or [])}
    ignore_names = {os.path.basename(str(n)).lower() for n in (ignore_file_names or [])}

    file_paths: List[str] = []
    for root, _, files in os.walk(directory_path, topdown=True, followlinks=False):
        for name in files:
            base = os.path.basename(name)
            base_l = base.lower()
            if base_l in ignore_names:
                continue
            ext = os.path.splitext(base_l)[1].lstrip(".")  # np. ".PDF" -> "pdf"
            if ext in ignore_exts:
                continue
            file_paths.append(os.path.abspath(os.path.join(root, base)))

    file_paths.sort()
    return file_paths

from typing import Sequence, Iterable

def make_stats_text(
    hidden_size_tab: Sequence[int],
    train_loss_tab: Sequence[float],
    val_loss_tab: Sequence[float],
    val_acc_tab: Sequence[float],
    *,
    title: str = "Statystyki vs. rozmiar warstwy ukrytej",
    show_raw: bool = False,
    loss_prec: int = 6,
    acc_prec: int = 4,
    include_acc_percent: bool = True,
    sep: str = " | ",
) -> str:
    """
    Buduje czytelny tekst do wklejenia do .txt:
    - (opcjonalnie) blok surowych tablic wejściowych,
    - Tabela oryginalna (kolejność wejściowa),
    - Tabela posortowana po val_acc (malejąco),
    - Tabela posortowana po val_loss (malejąco).
    """
    h = list(hidden_size_tab)
    tl = list(train_loss_tab)
    vl = list(val_loss_tab)
    va = list(val_acc_tab)

    n = len(h)
    if not (len(tl) == len(vl) == len(va) == n):
        raise ValueError("Wszystkie tablice muszą mieć taką samą długość.")

    def fmt_list(vals: Iterable, *, prec: int | None = None, ints: bool = False) -> str:
        if ints:
            return "[" + ", ".join(str(int(v)) for v in vals) + "]"
        if prec is None:
            return "[" + ", ".join(str(v) for v in vals) + "]"
        return "[" + ", ".join(f"{float(v):.{prec}f}" for v in vals) + "]"

    def build_rows(data):
        """data: iterable z krotek (hidden, train_loss, val_loss, val_acc)"""
        rows = []
        for hh, t, v, a in data:
            row = [
                str(int(hh)),
                f"{float(t):.{loss_prec}f}",
                f"{float(v):.{loss_prec}f}",
                f"{float(a):.{acc_prec}f}",
            ]
            if include_acc_percent:
                row.append(f"{float(a)*100:.2f}%")
            rows.append(row)
        return rows

    def fmt_table(rows, headers):
        """Zwraca listę linii tekstu tabeli (z nagłówkiem i kreską)."""
        widths = [
            max(len(hd), max((len(r[i]) for r in rows), default=0))
            for i, hd in enumerate(headers)
        ]
        def fmt_row(cells):
            return sep.join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
        sep_len = len(sep) * (len(headers) - 1)
        line_len = sum(widths) + sep_len
        lines = [fmt_row(headers), "-" * line_len]
        lines.extend(fmt_row(r) for r in rows)
        return lines

    headers = ["hidden", "train_loss", "val_loss", "val_acc"]
    if include_acc_percent:
        headers.append("val_acc_%")

    data = list(zip(h, tl, vl, va))

    # 1) oryginalna
    rows_orig = build_rows(data)
    # 2) sort po val_acc (malejąco)
    data_desc_acc = sorted(data, key=lambda t: t[3], reverse=True)
    rows_desc_acc = build_rows(data_desc_acc)
    # 3) sort po val_loss (malejąco)
    data_desc_loss = sorted(data, key=lambda t: t[2], reverse=True)
    rows_desc_loss = build_rows(data_desc_loss)

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    if show_raw:
        lines.append("Wejściowe tablice:")
        lines.append(f"hidden_size_tab : {fmt_list(h, ints=True)}")
        lines.append(f"train_loss_tab  : {fmt_list(tl, prec=loss_prec)}")
        lines.append(f"val_loss_tab    : {fmt_list(vl, prec=loss_prec)}")
        lines.append(f"val_acc_tab     : {fmt_list(va, prec=acc_prec)}")
        lines.append("")

    lines.append("Tabela (oryginalna kolejność):")
    lines.extend(fmt_table(rows_orig, headers))
    lines.append("")

    lines.append("Tabela (posortowana po val_acc ↓):")
    lines.extend(fmt_table(rows_desc_acc, headers))
    lines.append("")

    lines.append("Tabela (posortowana po val_loss ↓):")
    lines.extend(fmt_table(rows_desc_loss, headers))

    return "\n".join(lines)



def create_plot(
    x: Sequence[Mapping[str, Any]],
    y: Sequence[Mapping[str, Any]],
    plot_name: str,
    x_name: str,
    y_name: str,
    save: bool = True,
    save_name: str = "przykladowy_wykres",
    *,
    # --- estetyka (domyślne wartości można nadpisać) ---
    figsize: Tuple[float, float] = (7.0, 5.0),
    line_style: str = "-",
    line_width: float = 2.0,
    marker: str = "",            # np. "o", "s", "^"; pusty = bez markerów
    marker_size: float = 6.0,
    alpha: float = 1.0,
    grid: bool = True,
    grid_style: str = "--",
    grid_alpha: float = 0.3,
    title_size: int = 14,
    label_size: int = 12,
    tick_size: int = 10,
    # --- kolor domyślny dla serii bez 'color' ---
    default_color: str | None = None,  # None => cykl Matplotlib
    # --- zapis wektorowy ---
    vector_format: str = "pdf",  # "pdf" | "svg" | "eps"
    transparent: bool = False,
    tight_layout: bool = True,
    # --- ścieżka zapisu ---
    save_path: str | Path | None = None,  # katalog do zapisu (absolutny lub względny)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Rysuje wiele linii (x vs y) na jednym wykresie. x i y to listy słowników:
      {"tab": <tablica>, "name": "etykieta", "color": "blue"}
    Dla każdej serii i: parujemy x[i] z y[i].

    Zwraca (fig, ax). Opcjonalnie zapisuje wektorowo do <save_path>/<save_name>.<format>.

    Parametry zapisu:
    - save_path: katalog docelowy (utworzy się automatycznie). Gdy None -> bieżący katalog.
    - vector_format: "pdf" (domyślnie), "svg" lub "eps".
    """
    allowed_formats = {"pdf", "svg", "eps"}
    vf = vector_format.lower()
    if vf not in allowed_formats:
        raise ValueError(f"vector_format musi być jednym z {allowed_formats}, a jest '{vector_format}'.")

    if len(x) != len(y):
        raise ValueError(f"Liczba serii w x ({len(x)}) musi równać się liczbie serii w y ({len(y)}).")

    fig, ax = plt.subplots(figsize=figsize)

    for i, (xd, yd) in enumerate(zip(x, y), start=1):
        if not isinstance(xd, Mapping) or not isinstance(yd, Mapping):
            raise TypeError("Elementy x i y muszą być słownikami.")
        if "tab" not in xd or "tab" not in yd:
            raise KeyError("Każdy słownik musi zawierać klucz 'tab' z tablicą wartości.")

        xi = np.asarray(xd["tab"], dtype=float).ravel()
        yi = np.asarray(yd["tab"], dtype=float).ravel()
        if xi.shape != yi.shape:
            raise ValueError(f"Seria {i}: różna liczba punktów: len(x)={xi.size}, len(y)={yi.size}.")

        # etykieta i kolor: preferuj dane z 'y', potem z 'x', potem domyślne
        label = yd.get("name") or xd.get("name") or f"linia_{i}"
        color = yd.get("color", xd.get("color", default_color))

        plot_kwargs = dict(
            linestyle=line_style,
            linewidth=line_width,
            marker=(marker if marker else None),
            markersize=marker_size,
            alpha=alpha,
            label=label,
        )
        if color is not None:
            plot_kwargs["color"] = color  # jeśli None, użyj cyklu Matplotlib

        ax.plot(xi, yi, **plot_kwargs)

    ax.set_title(plot_name, fontsize=title_size)
    ax.set_xlabel(x_name, fontsize=label_size)
    ax.set_ylabel(y_name, fontsize=label_size)
    ax.tick_params(labelsize=tick_size)

    if grid:
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

    ax.legend()

    if tight_layout:
        fig.tight_layout()

    if save:
        # Ustal katalog docelowy
        out_dir = Path.cwd() if save_path is None else Path(save_path).expanduser().resolve()
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"save_path '{out_dir}' wskazuje na plik, a nie katalog.")
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = out_dir / f"{save_name}.{vf}"
        fig.savefig(
            filename,
            format=vf,
            bbox_inches="tight",
            transparent=transparent,
        )

    return fig, ax

def line_to_dict(line):
    res_dict = {}

    line_list = line.strip().split(" | ")
    epoch = int(line_list[0].split(" ")[1][:-1])
    train_loss = float(line_list[0].split(" ")[2].split("=")[1])
    line_list.pop(0)
    res_dict["epoch"] = epoch
    res_dict["train_loss"] = train_loss

    for element in line_list:
        element = element.strip()
        elements = element.split("=")
        value = elements[1]
        value = change_type_from_file_string(value)
        res_dict[elements[0]] = value

    return res_dict



def get_values_from_wynik_file(wynik_file="przyklad_wynik.txt", start_data_line_index=2):
    with open(wynik_file, "r") as f:
        wynik = f.read().strip().split("\n")

    start_data = wynik[start_data_line_index].strip().split(" | ")
    start_data_dict = {}
    for element in start_data:
        elements = element.split("=")
        start_data_dict[elements[0]] = change_type_from_file_string(elements[1])

    best_epoch = int(wynik[-1].strip().split(" ")[-1])

    best_line = wynik[best_epoch + start_data_line_index]
    best_line_dict = line_to_dict(best_line)
    return start_data_dict, best_line_dict

def add_file_to_plot_tabs(train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab, wynik_file="przyklad_wynik.txt", start_data_line_index=2):
    start_data_dict, best_line_dict = get_values_from_wynik_file(wynik_file, start_data_line_index)
    hidden_size_tab.append(start_data_dict["hidden"])
    train_loss_tab.append(best_line_dict["train_loss"])
    val_loss_tab.append(best_line_dict["val_loss"])
    val_acc_tab.append(best_line_dict["val_acc"])


def create_plot_from_directory(directory_path, save_path=None, algo_name=None,  start_data_line_index=2):
    if algo_name is None:
        algo_name = directory_path.split("/")[-2]
    train_loss_tab = []
    val_loss_tab = []
    val_acc_tab = []
    hidden_size_tab = []
    files = get_all_files_paths_from_directory(directory_path, ignorowane_rozszerzenia=["pdf"], ignore_file_names=["wynik_koncowy.txt"])
    start_data_dict, best_line_dict = get_values_from_wynik_file(files[0], start_data_line_index)
    data_set = start_data_dict["dataset"]
    files.sort(key=lambda x: get_values_from_wynik_file(x, start_data_line_index)[0]["hidden"])
    for path in files:
        add_file_to_plot_tabs(train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab, path, start_data_line_index)

    x = [
        {"tab": hidden_size_tab, "name": "hidden size"},
        {"tab": hidden_size_tab, "name": "hidden size"},
    ]
    y = [
        {"tab": train_loss_tab, "name": "train loss"},
        {"tab": val_loss_tab, "name": "validation loss"},
    ]
    additional_data = make_stats_text(
    hidden_size_tab, train_loss_tab, val_loss_tab, val_acc_tab,
    title=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}"
    )
    txt_file_path = Path(save_path) / "wynik_koncowy.txt"
    with open(txt_file_path, "w") as f:
        f.write(additional_data)


    fig, ax = create_plot(
        x, y,
        plot_name=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}",
        x_name="rozmiar warstwy ukrytej",
        y_name="wartosci metryk",
        vector_format="pdf",
        save=True, save_name=f"{algo_name}_{data_set}", save_path=save_path,
        marker="o"
    )

    return fig, ax, additional_data

def stworz_wszystkie_wykresy(
    wykresy_path: str | Path = "/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy",
    *,
    verbose: bool = True,
) -> Dict[str, List[str] | List[Tuple[str, str]]]:
    """
    Uruchamia create_plot_from_directory dla wszystkich katalogów pod wykresy_path,
    które zawierają *jakiekolwiek pliki*. Puste katalogi są ignorowane.

    Zwraca dict:
      {
        "processed":       [<ścieżki przetworzonych>],
        "skipped_empty":   [<ścieżki pustych katalogów>],
        "errors":          [(<ścieżka>, <komunikat błędu>), ...]
      }
    """
    root = Path(wykresy_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Ścieżka '{root}' nie jest katalogiem lub nie istnieje.")

    processed: List[str] = []
    skipped_empty: List[str] = []
    errors: List[Tuple[str, str]] = []

    # Przejdź po wszystkich podkatalogach
    for d in sorted([p for p in root.rglob("*") if p.is_dir()]):
        try:
            entries = list(d.iterdir())
        except Exception as e:
            msg = f"Nie można odczytać zawartości: {e}"
            errors.append((str(d), msg))
            if verbose:
                print(f"[ERR] {d}: {msg}")
            continue

        if not entries:
            # katalog pusty
            skipped_empty.append(str(d))
            if verbose:
                print(f"[SKIP] pusty: {d}")
            continue

        # Czy katalog zawiera *jakiekolwiek pliki*?
        has_file = any(ch.is_file() for ch in entries)
        if not has_file:
            # To raczej węzeł pośredni (zawiera kolejne katalogi). Idziemy dalej.
            continue

        # To katalog „z wykresem”: odpalamy generator
        try:
            create_plot_from_directory(str(d), save_path=str(d))
            processed.append(str(d))
            if verbose:
                print(f"[OK] {d}")
        except Exception as e:
            errors.append((str(d), str(e)))
            if verbose:
                print(f"[ERR] {d}: {e}")

    return {
        "processed": processed,
        "skipped_empty": skipped_empty,
        "errors": errors,
    }


def create_plot_from_directory_kompresja(directory_path, save_path=None, algo_name=None,  start_data_line_index=2):
    if algo_name is None:
        algo_name = directory_path.split("/")[-2]
    train_loss_tab = []
    val_loss_tab = []
    val_acc_tab = []
    hidden_size_tab = []
    files = get_all_files_paths_from_directory(directory_path, ignorowane_rozszerzenia=["pdf"], ignore_file_names=["wynik_koncowy.txt"])
    start_data_dict, best_line_dict = get_values_from_wynik_file(files[0], start_data_line_index)
    data_set = start_data_dict["dataset"]
    files.sort(key=lambda x: get_values_from_wynik_file(x, start_data_line_index)[0]["hidden"])
    for path in files:
        add_file_to_plot_tabs(train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab, path, start_data_line_index)

    x = [
        {"tab": hidden_size_tab, "name": "hidden size"},
        {"tab": hidden_size_tab, "name": "hidden size"},
    ]
    y = [
        {"tab": train_loss_tab, "name": "train loss"},
        {"tab": val_loss_tab, "name": "validation loss"},
    ]
    additional_data = make_stats_text(
    hidden_size_tab, train_loss_tab, val_loss_tab, val_acc_tab,
    title=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}"
    )
    txt_file_path = Path(save_path) / "wynik_koncowy.txt"
    with open(txt_file_path, "w") as f:
        f.write(additional_data)


    fig, ax = create_plot(
        x, y,
        plot_name=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}",
        x_name="rozmiar warstwy ukrytej",
        y_name="wartosci metryk",
        vector_format="pdf",
        save=True, save_name=f"{algo_name}_{data_set}", save_path=save_path,
        marker="o"
    )

    return fig, ax, additional_data
#create_plot_from_directory("/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy/mlp/mnist/wykres1_pierwsza_pr", save_path="/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy/mlp/mnist/wykres1_pierwsza_pr")


# wynik = stworz_wszystkie_wykresy(verbose=False)
# print(wynik["processed"])
# print("-"*100)
# print("ERRORS")
# print("-"*100)
# print(wynik["errors"])

create_plot_from_directory_kompresja("/home/miku/PycharmProjects/Pracainzynierska/kompresja_iteracyjna_wyniki_1", save_path="/home/miku/PycharmProjects/Pracainzynierska/kompresja_iteracyjna_wyniki_1")

