from typing import Tuple, Iterable, Sequence, Mapping, Any, List
import numpy as np
import matplotlib.pyplot as plt
import os

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

def get_all_files_paths_from_directory(directory_path: str) -> List[str]:
    """
    Zwraca listę *absolutnych* ścieżek do wszystkich plików w podanym katalogu (rekurencyjnie).
    Nie podąża za dowiązaniami symbolicznymi do katalogów.

    :param directory_path: Ścieżka do katalogu startowego.
    :return: Posortowana lista absolutnych ścieżek plików.
    :raises ValueError: Gdy ścieżka nie istnieje lub nie jest katalogiem.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Ścieżka '{directory_path}' nie jest katalogiem lub nie istnieje.")

    file_paths: List[str] = []
    for root, dirs, files in os.walk(directory_path, topdown=True, followlinks=False):
        for name in files:
            file_paths.append(os.path.abspath(os.path.join(root, name)))

    file_paths.sort()
    return file_paths


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
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Rysuje wiele linii (x vs y) na jednym wykresie. x i y to listy słowników:
      {"tab": <tablica>, "name": "etykieta", "color": "blue"}
    Dla każdej serii i: parujemy x[i] z y[i].

    Zwraca (fig, ax). Opcjonalnie zapisuje wektorowo do save_name.<format>.
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
        filename = f"{save_name}.{vf}"
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


def create_plot_from_directory(directory_path, algo_name=None,  start_data_line_index=2):
    if algo_name is None:
        algo_name = directory_path.split("/")[-2]
    train_loss_tab = []
    val_loss_tab = []
    val_acc_tab = []
    hidden_size_tab = []
    files = get_all_files_paths_from_directory(directory_path)
    start_data_dict, best_line_dict = get_values_from_wynik_file(files[0], start_data_line_index)
    data_set = start_data_dict["dataset"]
    files.sort(key=lambda x: get_values_from_wynik_file(x, start_data_line_index)[0]["hidden"])
    for path in files:
        print(path)
        add_file_to_plot_tabs(train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab, path, start_data_line_index)

    x = [
        {"tab": hidden_size_tab, "name": "hidden size"},
        {"tab": hidden_size_tab, "name": "hidden size"},
    ]
    y = [
        {"tab": train_loss_tab, "name": "train loss"},
        {"tab": val_loss_tab, "name": "validation loss"},
    ]
    print(f"hidden_size_tab : {hidden_size_tab}")
    print(f"train_loss_tab : {train_loss_tab}")
    print(f"val_loss_tab : {val_loss_tab}")
    print(f"val_acc_tab : {val_acc_tab}")
    fig, ax = create_plot(
        x, y,
        plot_name=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}",
        x_name="rozmiar warstwy ukrytej",
        y_name="wartosci metryk",
        vector_format="pdf",
        save=True, save_name=f"{algo_name}_{data_set}",
        marker="o"
    )

    return fig, ax
# # --- Przykład użycia ---
# x = [
#     {"tab": [0, 1, 2, 3], "name": "x dla kwadratu", "color": "blue"},
#     {"tab": [0, 1, 2, 3], "name": "x dla sześcianu", "color": "#ff7f0e"},
# ]
# y = [
#     {"tab": [0, 1, 4, 9],  "name": "y = x^2", "color": "blue"},
#     {"tab": [0, 1, 8, 27], "name": "y = x^3", "color": "#ff7f0e"},
# ]
#
# fig, ax = create_plot(
#     x, y,
#     plot_name="Porównanie funkcji potęgowych",
#     x_name="x", y_name="y",
#     vector_format="pdf",  # wektorowo
#     save=True, save_name="potegi"
# )



# print(get_all_files_paths_from_directory("mlp/mnist"))
# start_data, best_line_dict = get_values_from_wynik_file("mlp/mnist/wyniki_przyklad.txt")
# print(start_data)
# print(best_line_dict)

create_plot_from_directory("mlp/mnist")