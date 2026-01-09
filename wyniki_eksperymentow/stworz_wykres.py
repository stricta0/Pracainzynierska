from typing import Tuple, Iterable, Sequence, Mapping, Any, List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from sympy.physics.secondquant import Creator


class CreatorWykresow:
    # ============== HELPERS ===================
    def is_int(self, s):
        """
        sprawdza czy s jest intem
        """
        try:
            int(s.strip())
            return True
        except ValueError:
            return False

    def change_type_from_file_string(self, value):
        """
        zamienia value na float, int, bool lub string w zaleznosci od jak wyglada value
        """
        if value[-1] == "%":
            value = float(value[:-1]) / 100.0
        elif "." in value:
            all_numeryczne = "0123456789."
            if_float = True
            for znak in value:
                if znak not in all_numeryczne:
                    if_float = False
            if if_float:
                value = float(value)
        elif value == "False":
            value = False
        elif value == "True":
            value = True
        elif self.is_int(value):
            value = int(value)
        return value

    def get_all_files_paths_from_directory(self, directory_path, ignorowane_rozszerzenia = None, ignore_file_names = None):
        """
        Zwraca listę *absolutnych* ścieżek do wszystkich plików w podanym katalogu (rekurencyjnie),
        z filtrami:
          - ignorowane_rozszerzenia: np. ["pdf", "jpg"]  -> pomija pliki o tych rozszerzeniach
          - ignore_file_names: np. ["readme.md", "wynik_koncowy.txt"] -> pomija pliki o tych nazwach
        Porównania są niewrażliwe na wielkość liter. Nazwy w ignore_file_names podaj BEZ ścieżek.

        :param directory_path: ścieżka do katalogu startowego
        :param ignorowane_rozszerzenia: lista rozszerzen ktore ignorujemy
        :param ignore_file_names: lista ignorowanych plikow z katalogu
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

    # ============== TWORZNIE WYKRESOW ===================
    def create_plot( self,
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
        odwrocona_os_x: bool = False,
        max_na_osi_x: float = None,
        min_na_osi_x: float = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Rysuje wiele linii (x vs y) na jednym wykresie. x i y to listy słowników:
          {"tab": <tablica>, "name": "etykieta", "color": "blue"}
        gdzie color jest opcjonalny
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
        min_x, max_x = float("inf"), float("-inf")

        for i, (xd, yd) in enumerate(zip(x, y), start=1):
            if not isinstance(xd, Mapping) or not isinstance(yd, Mapping):
                raise TypeError("Elementy x i y muszą być słownikami.")
            if "tab" not in xd or "tab" not in yd:
                raise KeyError("Każdy słownik musi zawierać klucz 'tab' z tablicą wartości.")

            xi = np.asarray(xd["tab"], dtype=float).ravel()
            yi = np.asarray(yd["tab"], dtype=float).ravel()

            min_x = min(min_x, float(np.min(xi)))
            max_x = max(max_x, float(np.max(xi)))

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
        if odwrocona_os_x and np.isfinite(min_x) and np.isfinite(max_x):
            if max_na_osi_x is not None and min_na_osi_x is not None:
                ax.set_xlim(max_na_osi_x, min_na_osi_x)  # 100 po lewej, 0 po prawej
            else:
                ax.set_xlim(max_x, min_x)  # 100 po lewej, 0 po prawej
        elif max_na_osi_x is not None and min_na_osi_x is not None:
            ax.set_xlim(min_na_osi_x, max_na_osi_x)

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

    # =============== PRZETWARZANIE PLIKOW WYNIKOWYCH DOUBLE DESCENT =========================
    def line_to_dict_double_descent_plik(self, line):
        """
        bierze linike z pliku z wynikami eksperymentu double_descent
        i zwraca slownik z tymi wynikami

        @param line: linia z pliku w str

        returns line w formie slownika
        """
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
            value = self.change_type_from_file_string(value)
            res_dict[elements[0]] = value

        return res_dict


    def get_values_from_wynik_file_double_descent(self, wynik_file="przyklad_wynik.txt", start_data_line_index=2):
        """
        Wyciaga dane z pliku txt double_descent i zwraca w formie pary slownikow
        @param wynik_file: path do pliku z wynikami double_descent
        @param start_data_line_index: index gdzie zaczynaja sie dane w piku tekstowym

        returns slownik_argumentow_poczatkowych, slownik_wynikow_najlepszej_epoki
        slownik_argumentow_poczatkowych - to slownik z informacjami o trenowanym modelu (ilosc wag, epok itd)
        slownik_wynikow_najlepszej_epoki - slownik wynikow metryk dla najlpszej epoki modelu
        """
        with open(wynik_file, "r") as f:
            wynik = f.read().strip().split("\n")

        start_data = wynik[start_data_line_index].strip().split(" | ")
        start_data_dict = {}
        for element in start_data:
            elements = element.split("=")
            start_data_dict[elements[0]] = self.change_type_from_file_string(elements[1])
        best_epoch = int(wynik[-1].strip().split(" ")[-1])

        #TODO: best_epoch po vall_loss
        best_epoch_vall_loss = 0
        min_val_loss = 999999.9
        for linia in wynik:
            if linia[:6] != "[Epoch":
                continue
            linia_podzielona = linia.strip().split(" | ")
            epoka = int(linia_podzielona[0].split("]")[0].split(" ")[1])
            vall_loss = 999999.9
            for wartosc in linia_podzielona:
                if wartosc.split("=")[0] == "val_loss":
                    vall_loss = float(wartosc.split("=")[1])
            if min_val_loss > vall_loss:
                min_val_loss = vall_loss
                best_epoch_vall_loss = epoka
        best_line = wynik[best_epoch + start_data_line_index]
        best_line_vall_loss = wynik[best_epoch_vall_loss + start_data_line_index]

        best_line_dict = self.line_to_dict_double_descent_plik(best_line)
        best_line_dict_vall_loss = self.line_to_dict_double_descent_plik(best_line_vall_loss)
        #return start_data_dict, best_line_dict
        return start_data_dict, best_line_dict_vall_loss

    def add_file_to_plot_tabs_double_descent(self, train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab,sorting_key, wynik_file="przyklad_wynik.txt", start_data_line_index=2):
        """
        Dodaje dane z pliku tekstowego na temat pojedynczego eksperymtu double descent i
        dodaje je do wspolnych tablic - tabice sa inplace (faktycznie dodaje)
        @param train_loss_tab
        @param val_loss_tab
        @param val_acc_tab
        @param hidden_size_tab --- wspolne tablice z wynikami dla wszystkich eksperymentow
        @param wynik_file: path do pliku z przejscia double descent (do 1 pliku / 1 eksperymentu)
        @param start_data_line_index  indeks od ktorego zaczynaja sie dane w .txt

        returns NONE (dziala in place)
        """

        start_data_dict, best_line_dict = self.get_values_from_wynik_file_double_descent(wynik_file, start_data_line_index)
        hidden_size_tab.append(start_data_dict[sorting_key])
        train_loss_tab.append(best_line_dict["train_loss"])
        val_loss_tab.append(best_line_dict["val_loss"])
        val_acc_tab.append(best_line_dict["val_acc"])

    def make_stats_text_for_double_descent_tables(
            self,
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


    # =============== TWORZENIE WYKRESOW DOUBLE DESCENT =========================
    def create_plot_from_directory_double_descent(self, directory_path, save_path=None, algo_name=None, start_data_line_index=2):
        """
        Tworzy wykres z folderu dla eksperymentu double descent
        @param directory_path: sciezka do tego folderu
        @param save_path: folder do ktorego zapisujemy wyniki - ten sam co directory_path jesli nie podany
        @param algo_name: nazwa algorytmu ktory przeriabiamy (potrzebana do nazw itd) bazowo brana z nazwy folderu
        @param start_data_line_index: informacja o tym na ktorym indeksie zaczynaja sie dane w pliku (=2 oznacza pomin 2 pierwsze linie pliku)

        returns fig, ax, additional_data
        fig, ax - wykres w formie mathplotliba
        additional_data - tresc dodatkowego pliku .txt z informacjami
        """
        if algo_name is None:
            algo_name = directory_path.split("/")[-3]
        if save_path is None:
            save_path = directory_path
        train_loss_tab = []
        val_loss_tab = []
        val_acc_tab = []
        hidden_size_tab = []
        files = self.get_all_files_paths_from_directory(directory_path, ignorowane_rozszerzenia=["pdf"], ignore_file_names=["wynik_koncowy.txt"])
        start_data_dict, best_line_dict = self.get_values_from_wynik_file_double_descent(files[0], start_data_line_index)
        data_set = start_data_dict["dataset"]
        names_map = {"mlp" : "hidden", "rff" : "n_features", "cnn": "ch1"}
        sorting_key = names_map.get(algo_name.lower())
        files.sort(key=lambda x: self.get_values_from_wynik_file_double_descent(x, start_data_line_index)[0][sorting_key])
        for path in files:
            self.add_file_to_plot_tabs_double_descent(train_loss_tab, val_loss_tab, val_acc_tab, hidden_size_tab, sorting_key, path, start_data_line_index)
        x = [
            {"tab": hidden_size_tab, "name": "hidden size"},
            {"tab": hidden_size_tab, "name": "hidden size"},
        ]
        y = [
            {"tab": train_loss_tab, "name": "train loss"},
            {"tab": val_loss_tab, "name": "validation loss"},
        ]
        additional_data = self.make_stats_text_for_double_descent_tables(
        hidden_size_tab, train_loss_tab, val_loss_tab, val_acc_tab,
        title=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}"
        )
        txt_file_path = Path(save_path) / "wynik_koncowy.txt"
        with open(txt_file_path, "w") as f:
            f.write(additional_data)

        fig, ax = self.create_plot(
            x, y,
            plot_name=f"Wpływ rozmiaru warstwy ukrytej na metryki modelu {algo_name} na zbiorze {data_set}",
            x_name="rozmiar warstwy ukrytej",
            y_name="wartosci metryk",
            vector_format="pdf",
            save=True, save_name=f"{algo_name}_{data_set}", save_path=save_path,
            marker="o"
        )
        return fig, ax, additional_data

    def stworz_wszystkie_wykresy_double_descent(self, wykresy_path = "/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy", verbose = True):
        """
        Tworzy wszystkie wykresy dla double_descent z katalogu wyniki_eksperymentow/wykresy

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
                self.create_plot_from_directory_double_descent(str(d))
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


    # =============== PRZETWARZANIE PLIKOW WYNIKOWYCH KOMPRESJA =========================
    def get_wyniki_kompresja(self, file_path):
        """
        Odczttywanie plików wynikowych po kompresji
        @param file_path - sciezka do pliku po kompresji z ktorego mamy odczytac wartosci

        zwraca tablice lini zmienionych na slownik
        tab = [dict1, dict2, ...]
        dict2 = {'i': 1, 'operation_name': 'PRUN', 'fc1': {'w': {'aktualna': 3763200, 'orginalna_wielkosc': 3920000, 'aktualna_w_proc': 96.0}, 'b': {'aktualna': 5000, 'orginalna_wielkosc': 5000, 'aktualna_w_proc': 100.0}}, 'fc2': {'w': {'aktualna': 48000, 'orginalna_wielkosc': 50000, 'aktualna_w_proc': 96.0}, 'b': {'aktualna': 10, 'orginalna_wielkosc': 10, 'aktualna_w_proc': 100.0}}, 'val_loss': 0.15514711806178094, 'val_acc': 0.9562}
        """

        with open(file_path, "r") as f:
            plik = f.read().strip().split("\n")

        start_data = plik[0].strip().split(" | ")
        start_data_dict = {}
        for element in start_data:
            elements = element.split("=")
            start_data_dict[elements[0]] = self.change_type_from_file_string(elements[1])

        plik.pop(0)
        wartosci = [el.strip().split("|") for el in plik]
        wszystkie_pomiary = []
        for i in range(len(wartosci)):
            linia = wartosci[i]
            linia = [el.strip() for el in linia]

            wartosci[i] = linia
            wartosc_dict = {
                "i": int(linia[0]),
                "operation_name": linia[1],
            }


            for i in range(2, len(linia)):
                parametr = linia[i]
                parametr_name = parametr.strip().split(" ")[0].strip().strip(":")
                parametr_val = parametr.strip().split(" ")[1].strip().strip(":")
                if parametr_name == "val_loss" or parametr_name == "val_acc":
                    val = float(parametr_val)
                    wartosc_dict[parametr_name] = val
                    continue
                pozostalo_w_procentach = float(parametr.strip().split(" ")[2].strip().strip(":").strip("(").strip(")").strip("%"))
                nazwa_warstwy = parametr_name.strip().split(".")[0]
                typ_elementu = parametr_name.strip().split(".")[1] #b bias lub w wagi
                aktualna_wielkosc = int(parametr_val.split("/")[0])
                orginalna_wielkosc = int(parametr_val.split("/")[1])
                wyjsciowy_slownik = {"aktualna": aktualna_wielkosc, "orginalna_wielkosc": orginalna_wielkosc, "aktualna_w_proc": pozostalo_w_procentach}
                if nazwa_warstwy not in wartosc_dict:
                    wartosc_dict[nazwa_warstwy] = {typ_elementu : wyjsciowy_slownik}
                else:
                    wartosc_dict[nazwa_warstwy][typ_elementu] = wyjsciowy_slownik
            wszystkie_pomiary.append(wartosc_dict.copy())
        return start_data_dict, wszystkie_pomiary

    def stworz_wykres_kompresja(self, file_path, algo_name, save_path, reverse_kompresja=False):

        start_wyniki, wyniki = self.get_wyniki_kompresja(file_path)
        rozmiar_procentowy_wag = [wynik["fc1"]["w"]["aktualna_w_proc"] for wynik in wyniki]
        if reverse_kompresja:
            rozmiar_procentowy_wag = [100.0 - waga for waga in rozmiar_procentowy_wag]
        val_acc = [wynik["val_acc"] for wynik in wyniki]
        print(rozmiar_procentowy_wag)
        print(val_acc)
        x = [
            {"tab": rozmiar_procentowy_wag, "name": "Procenotwa wielkosc pozostalych wag"}
        ]
        y = [
            {"tab": val_acc, "name": "val accuracy"}
        ]
        x_name = "procent kompresji"
        if reverse_kompresja:
            x_name = "pozostaly procent modelu"
        fig, ax = self.create_plot(
            x, y,
            plot_name=f"Wpływ kompresji modelu na metryki modelu {algo_name} na zbiorze {start_wyniki["dataset"]}",
            x_name= x_name,
            y_name="wartosci metryk",
            vector_format="pdf",
            save=True, save_name=f"{algo_name}_{start_wyniki["dataset"]}", save_path=save_path,
            marker="o",
            odwrocona_os_x=True,
            max_na_osi_x=100.0,
            min_na_osi_x=0.0
        )
        print(f"stworzono wykres na pathie:{save_path}")
        return fig, ax

    #create_plot_from_directory("/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy/mlp/mnist/wykres1_pierwsza_pr", save_path="/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy/mlp/mnist/wykres1_pierwsza_pr")


    # wynik = stworz_wszystkie_wykresy_double_descent(verbose=False)
    # print(wynik["processed"])
    # print("-"*100)
    # print("ERRORS")
    # print("-"*100)
    # print(wynik["errors"])

    #stworz_wykres_kompresja("/home/miku/PycharmProjects/Pracainzynierska/kompresja_iteracyjna_wyniki_1/kompresja.txt", "mlp", "/home/miku/PycharmProjects/Pracainzynierska/kompresja_iteracyjna_wyniki_1")
    #get_wyniki_kompresja("/home/mikolaj/PycharmProjects/Pracainzynierska/kompresja_iteracyjna_wyniki_1/kompresja.txt")

if __name__ == "__main__":
    creator = CreatorWykresow()
    creator.stworz_wykres_kompresja("/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/kompresja/mlp3000/kompresja.txt", "mlp", "/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/kompresja/mlp3000/")

