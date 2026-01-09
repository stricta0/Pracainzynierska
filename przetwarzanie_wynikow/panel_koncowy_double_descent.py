from definicje_wykresow_double_descent import DefinicjeWykresowDoubleDescent
from creator_wykresow import KreatorWykresow
from file_manager import FileManager
from stworz_plik_statystyk_z_folderu import StatisticsFileBuilder
import json
import os
from pathlib import Path

class PanelKoncowyDoubleDescent:

    def __init__(self, folder_path, folder_na_wyniki_path, config_file_name="config.json"):
        with open(config_file_name) as json_file:
            self.config = json.load(json_file)

        self.folder_path = folder_path
        self.folder_na_wyniki_path = folder_na_wyniki_path

        self.nazwa_folderu_z_wykresami = self.config["double_descent"]["nazwa_folderu_z_wykresami"]
        self.nazwa_folderu_z_wynikami_analizy_txt = self.config["double_descent"]["nazwa_folderu_z_wynikami_w_plikach_txt"]

        self.ignore_file_names = self.config["double_descent"]["ignorowane"]["pliki"]
        self.ignorowane_rozszerzenia = self.config["double_descent"]["ignorowane"]["rozszerzenia"]
        self.ignorowane_foldery = list(set(self.config["double_descent"]["ignorowane"]["ignorowane_foldery"] + [self.nazwa_folderu_z_wykresami, self.nazwa_folderu_z_wynikami_analizy_txt]))
        self.names_for_size_in_model_map = self.config["double_descent"]["names_for_size_in_model_map"]
        self.lista_plikow = FileManager.get_files_list(folder_path, self.ignore_file_names, self.ignorowane_rozszerzenia, self.ignorowane_foldery, self.names_for_size_in_model_map)

    def stworz_wykresy_z_danych_folderu_double_descent_best_models(self, save_path=None, lista_plikow=None, yscale="linear", xscale="linear"):
        if lista_plikow is None:
            lista_plikow = self.lista_plikow
        if save_path is None:
            save_path = self.folder_path
        definicje_wykresow = DefinicjeWykresowDoubleDescent(lista_plikow)
        dane_wszystich_wykresow = definicje_wykresow.get_dane_do_wszystkich_wykresow_best_model()
        for dane_wykresu in dane_wszystich_wykresow:
            x = dane_wykresu["x"]
            y = dane_wykresu["y"]
            title = dane_wykresu["title"]
            type_name = dane_wykresu["type"]
            file_name = dane_wykresu["file_name"]
            best_at = dane_wykresu["best_at"]
            fig, ax = KreatorWykresow.create_plot(
                x, y,
                plot_name=title,
                x_name="rozmiar modelu",
                y_name="wartości metryk",
                vector_format="pdf",
                save=True, save_name=f"{file_name}_scale_y_{yscale}_x_{xscale}",
                save_path=f"{save_path}/{self.nazwa_folderu_z_wykresami}/best_models_at_{best_at}/{type_name}",
                marker="o",
                yscale=yscale, #"linear" | "log" | "symlog" | "logit"
                xscale=xscale,
                add_line_at_zero=False,
            )

    def stworz_wykresy_z_danych_folderu_double_descent_po_liczbie_epok(self, save_path=None, lista_plikow=None, yscale="linear",
                                                                       xscale="linear"):
        if lista_plikow is None:
            lista_plikow = self.lista_plikow
        if save_path is None:
            save_path = self.folder_path
        definicje_wykresow = DefinicjeWykresowDoubleDescent(lista_plikow)
        dane_wszystich_wykresow = definicje_wykresow.get_dane_do_wszystkich_wykresow_po_liczbie_epok()
        for dane_wykresu in dane_wszystich_wykresow:
            x = dane_wykresu["x"]
            y = dane_wykresu["y"]
            title = dane_wykresu["title"]
            type_name = dane_wykresu["type"]
            file_name = dane_wykresu["file_name"]
            y_name = dane_wykresu["y_name"]
            fig, ax = KreatorWykresow.create_plot(
                x, y,
                plot_name=title,
                x_name="ilość epok",
                y_name=y_name,
                vector_format="pdf",
                save=True, save_name=f"{file_name}_scale_y_{yscale}_x_{xscale}",
                save_path=f"{save_path}/{self.nazwa_folderu_z_wykresami}/liczba_epok/{type_name}",
                yscale=yscale,  # "linear" | "log" | "symlog" | "logit"
                xscale=xscale,
                add_line_at_zero=False,
            )

    def zapisz_statystyki_z_danych_folderu_double_descent(self, lista_plikow=None, save_path=None):
        if lista_plikow is None:
            lista_plikow = self.lista_plikow
        if save_path is None:
            save_path = self.folder_path

        create_statistic_file_obj = StatisticsFileBuilder(lista_plikow, self.names_for_size_in_model_map)

        best_val_acc_file, best_val_loss_file = create_statistic_file_obj.get_data_from_files()
        text_file_zwykla_kolejnosc_acc = create_statistic_file_obj.create_string_from_dict(best_val_acc_file)
        test_file_zwykla_kolejnosc_loss = create_statistic_file_obj.create_string_from_dict(best_val_loss_file)

        folder_na_pliki = os.path.join(save_path, self.nazwa_folderu_z_wynikami_analizy_txt)
        os.makedirs(folder_na_pliki, exist_ok=True)

        # zwykła kolejność oba modele
        zwykla_kolejnosc_acc_txt = os.path.join(folder_na_pliki, "zwykla_kolejnosc_acc.txt")
        zwykla_kolejnosc_loss_txt = os.path.join(folder_na_pliki, "zwykla_kolejnosc_loss.txt")
        with open(zwykla_kolejnosc_acc_txt, "w", encoding="utf-8") as f:
            f.write(text_file_zwykla_kolejnosc_acc)
        with open(zwykla_kolejnosc_loss_txt, "w", encoding="utf-8") as f:
            f.write(test_file_zwykla_kolejnosc_loss)

        # posortowane po val_acc
        # best val acc epoch
        posortowany = create_statistic_file_obj.sort_file_dict(best_val_acc_file, "val_acc", reverse=True)
        text_file_acc_sort_acc = create_statistic_file_obj.create_string_from_dict(posortowany)
        sorted_by_acc_model_best_acc_txt = os.path.join(folder_na_pliki, "sorted_by_acc_model_best_acc.txt")
        with open(sorted_by_acc_model_best_acc_txt, "w", encoding="utf-8") as f:
            f.write(text_file_acc_sort_acc)
        # best val loss epoch
        posortowany = create_statistic_file_obj.sort_file_dict(best_val_loss_file, "val_acc", reverse=True)
        text_file_acc_sort_acc = create_statistic_file_obj.create_string_from_dict(posortowany)
        sorted_by_acc_model_best_acc_txt = os.path.join(folder_na_pliki, "sorted_by_acc_model_best_val_loss.txt")
        with open(sorted_by_acc_model_best_acc_txt, "w", encoding="utf-8") as f:
            f.write(text_file_acc_sort_acc)


        # posortowane po val_loss
        # best val acc epoch
        posortowany = create_statistic_file_obj.sort_file_dict(best_val_acc_file, "val_loss", reverse=False)
        text_file_acc_sort_acc = create_statistic_file_obj.create_string_from_dict(posortowany)
        sorted_by_acc_model_best_acc_txt = os.path.join(folder_na_pliki, "sorted_by_val_loss_model_best_acc.txt")
        with open(sorted_by_acc_model_best_acc_txt, "w", encoding="utf-8") as f:
            f.write(text_file_acc_sort_acc)
        # best val loss epoch
        posortowany = create_statistic_file_obj.sort_file_dict(best_val_loss_file, "val_loss", reverse=False)
        text_file_acc_sort_acc = create_statistic_file_obj.create_string_from_dict(posortowany)
        sorted_by_acc_model_best_acc_txt = os.path.join(folder_na_pliki, "sorted_by_val_loss_model_best_val_loss.txt")
        with open(sorted_by_acc_model_best_acc_txt, "w", encoding="utf-8") as f:
            f.write(text_file_acc_sort_acc)

class DoubleDescentRunner:
    def przeprowadz_analize_double_descent_w_folderze(self, folder_path, save_wyniki_path):
        panel = PanelKoncowyDoubleDescent(folder_path, save_wyniki_path)
        panel.stworz_wykresy_z_danych_folderu_double_descent_best_models(yscale="linear", xscale="linear") #"linear" | "log" | "symlog" | "logit"
        panel.stworz_wykresy_z_danych_folderu_double_descent_best_models(yscale="log", xscale="linear")
        panel.stworz_wykresy_z_danych_folderu_double_descent_best_models(yscale="linear", xscale="log")
        panel.stworz_wykresy_z_danych_folderu_double_descent_best_models(yscale="log", xscale="log")

        panel.stworz_wykresy_z_danych_folderu_double_descent_po_liczbie_epok(yscale="linear", xscale="linear")
        panel.stworz_wykresy_z_danych_folderu_double_descent_po_liczbie_epok(yscale="linear", xscale="log")
        panel.stworz_wykresy_z_danych_folderu_double_descent_po_liczbie_epok(yscale="log", xscale="linear")
        panel.stworz_wykresy_z_danych_folderu_double_descent_po_liczbie_epok(yscale="log", xscale="log")
        panel.zapisz_statystyki_z_danych_folderu_double_descent()



    def przeprowadz_analize_double_descent_w_folderze_rekurencyjnie(self, folder_path, log_warn=False):
        """
        Przechodzi po folderze i wszystkich jego podfolderach.
        Dla każdego katalogu, który zawiera co najmniej jeden plik (nie katalog),
        wywołuje przeprowadz_analize_double_descent_w_folderze(folder_x, folder_x).

        :param folder_path: ścieżka startowa
        """
        root = Path(folder_path)
        print(f"przerabiam folder: {root}")
        if not root.exists():
            raise FileNotFoundError(f"Ścieżka nie istnieje: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"To nie jest katalog: {root}")

        # 1) Jeśli w bieżącym katalogu jest jakikolwiek plik – odpal analizę
        try:
            has_any_file = any(p.is_file() for p in root.iterdir())
        except PermissionError:
            if log_warn:
                print(f"[WARN] Analiza nie powiodła się dla '{root}': PermissionError")
            return

        if has_any_file:
            try:
                # save_wyniki_path ustawiamy na ten sam katalog
                self.przeprowadz_analize_double_descent_w_folderze(str(root), str(root))
            except Exception as e:
                # Nie przerywamy całej rekursji, tylko raportujemy i idziemy dalej
                if log_warn:
                    print(f"[WARN] Analiza nie powiodła się dla '{root}': {e}")

        # 2) Rekurencyjnie przejdź po podkatalogach (posortowane dla powtarzalności)
        try:
            subdirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda x: x.name.lower())
        except PermissionError:
            if log_warn:
                print(f"[WARN] Analiza nie powiodła się dla '{root}': PermissionError")
            return

        for sub in subdirs:
            # Opcjonalnie: pomiń linki symboliczne do katalogów, aby uniknąć pętli
            if sub.is_symlink():
                continue
            self.przeprowadz_analize_double_descent_w_folderze_rekurencyjnie(sub)



if __name__ == "__main__":
    folder_path = "/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/wykresy/"
    runner = DoubleDescentRunner()
    runner.przeprowadz_analize_double_descent_w_folderze_rekurencyjnie(folder_path, True)
