from odczytywanie_plikow_tekstowych import TextFileReader
from creator_wykresow import KreatorWykresow
import os

class PanelKoncowyLTH:
    def __init__(self):
        self.folder_path = None
        self.file_path = None
        self.statystyki, self.dane_z_pliku = None, None
        self.dane_po_procentach = None
        self.nazwa_folderu_z_wykresami = "wykresy"

    def analiza_folder_path(self, folder_path):
        self.folder_path = folder_path
        self.file_path = folder_path + "/kompresja.txt"
        self.statystyki, self.dane_z_pliku = TextFileReader.odczytaj_plik_kompresja(self.file_path)
        self.dane_po_procentach = self._get_data_in_percentage(self.dane_z_pliku)

        self._stworz_tabelki()
        self._stworz_wykresy()

    def _stworz_tabelki(self):
        head_tabelki = "procent_wielkosci | val_loss | val_acc"
        zawartosc_tabelki = []
        for wielkosc in self.dane_po_procentach:
            zawartosc_tabelki.append(f"{wielkosc} | {self.dane_po_procentach[wielkosc]["val_loss"]} | {self.dane_po_procentach[wielkosc]["val_acc"]}")

        tabelka_po_wielkosci = self._sort_tabelka(zawartosc_tabelki, 0, reverse=True)
        tabelka_po_val_loss = self._sort_tabelka(zawartosc_tabelki, 1)
        tabelka_po_val_acc = self._sort_tabelka(zawartosc_tabelki, 2, reverse=True)
        min_po_val_acc = self._znajdz_min_do_100(tabelka_po_val_acc)
        min_po_val_loss = self._znajdz_min_do_100(tabelka_po_val_loss)
        self._zapisz_do_pliku("tabela_po_wielkosci.txt", self._tabelka_to_str(head_tabelki, tabelka_po_wielkosci))
        self._zapisz_do_pliku("tabela_po_val_loss.txt", self._tabelka_to_str(head_tabelki, tabelka_po_val_loss) + f"min size lepszy bądź równy od 100%: {min_po_val_loss}")
        self._zapisz_do_pliku("tabela_po_val_acc.txt", self._tabelka_to_str(head_tabelki, tabelka_po_val_acc) + f"min size lepszy bądź równy od 100%: {min_po_val_acc}")

    def _stworz_wykresy(self):
        dane_do_wykresow = self._stworz_dane_do_wykresow()
        for dane_wykresu in dane_do_wykresow:
            x = dane_wykresu["x"]
            y = dane_wykresu["y"]
            title = dane_wykresu["title"]
            file_name = dane_wykresu["file_name"]

            yscales = ["linear", "log"]
            xscales = ["linear", "log"]

            for xscale in xscales:
                for yscale in yscales:
                    fig, ax = KreatorWykresow.create_plot(
                        x, y,
                        plot_name=title,
                        x_name="rozmiar modelu",
                        y_name="wartości metryk",
                        vector_format="pdf",
                        save=True, save_name=f"{file_name}_scale_y_{yscale}_x_{xscale}",
                        save_path=f"{self.folder_path}/{self.nazwa_folderu_z_wykresami}",
                        marker="o",
                        yscale=yscale, #"linear" | "log" | "symlog" | "logit"
                        xscale=xscale,
                        add_line_at_zero=False,
                        odwrocona_os_x=True
                    )

    def _stworz_dane_do_wykresow(self):
        wielkosci, val_loss, val_acc = self._get_dane_do_wykresow()
        x_poj = [{"tab": wielkosci, "name": "wielkość modelu w %"}]
        x_double = [
            {"tab": wielkosci, "name": "wielkość modelu w %"},
            {"tab": wielkosci, "name": "wielkość modelu w %"}
        ]
        y_val_loss = [{"tab": val_loss, "name": "strata testowa"}]
        y_val_acc = [{"tab": val_acc, "name": "dokładność testowa"}]
        y_val_loss_and_acc = [
            {"tab": val_loss, "name": "strata testowa"},
            {"tab": val_acc, "name": "dokładność testowa"}
        ]
        dane_do_wykresow = [
            {"x" : x_poj, "y" : y_val_loss,  "title": "Zależność straty testowej od procentowej wielkości modelu po kompresji", "file_name": "size_vs_val_loss"},
            {"x" : x_poj, "y" : y_val_acc,  "title": "Zależność dokładności testowej od procentowej wielkości modelu po kompresji", "file_name": "size_vs_val_acc"},
            {"x" : x_double, "y" : y_val_loss_and_acc, "title": "Zależność straty i dokładności testowej od procentowej wielkości modelu po kompresji", "file_name": "size_vs_val_loss_and_acc"}
        ]
        return dane_do_wykresow

    def _get_dane_do_wykresow(self):
        """
        x = [
            {"tab": hidden_size_tab, "name": "hidden size"},
            {"tab": hidden_size_tab, "name": "hidden size"},
        ]
        y = [
            {"tab": train_loss_tab, "name": "train loss"},
            {"tab": val_loss_tab,   "name": "validation loss"},
        ]
        :return:
        """
        wielkosci = []
        val_loss = []
        val_acc = []
        for wielkosc in self.dane_po_procentach:
            wielkosci.append(wielkosc)
            val_loss.append(self.dane_po_procentach[wielkosc]["val_loss"])
            val_acc.append(self.dane_po_procentach[wielkosc]["val_acc"])
        return wielkosci, val_loss, val_acc

    def _get_data_in_percentage(self, dane_z_pliku):
        dane_in_percentage = {}
        for index in dane_z_pliku:
            rozmiar = dane_z_pliku[index]["rozmiar"]
            val_acc = dane_z_pliku[index]["val_acc"]
            val_loss = dane_z_pliku[index]["val_loss"]
            dane_in_percentage[rozmiar] = {"val_acc": val_acc, "val_loss": val_loss}
        return dane_in_percentage

    def _znajdz_min_do_100(self, tabelka):
        min = 101.0
        for linia in tabelka:
            linia = linia.split(" | ")
            wielkosc = float(linia[0].strip())
            if wielkosc < min:
                min = wielkosc
            if wielkosc == 100.0:
                break
        return min

    def _zapisz_do_pliku(self, nazwa_pliku, tresc):
        sciezka = os.path.join(self.folder_path, nazwa_pliku)
        with open(sciezka, "w", encoding="utf-8") as f:
            f.write(tresc)

    def _tabelka_to_str(self, head, zawartosc):
        tabelka_str = f"{head}\n"

        for element in zawartosc:
            tabelka_str += f"{element}\n"
        return tabelka_str

    def _sort_tabelka(self, zawartosc_tabelki, index_sortowania, reverse=False):
        def primary(x: str) -> float:
            return float(x.split(" | ")[index_sortowania])

        def size(x: str) -> float:
            return float(x.split(" | ")[0])

        # 1) sort tie-break (procent) zawsze malejąco
        tmp = sorted(zawartosc_tabelki, key=size, reverse=False)
        # 2) sort główny (stabilny), wg reverse
        return sorted(tmp, key=primary, reverse=reverse)

class LTHRunner:
    def __init__(self, main_folder_path):
        self.main_folder_path = main_folder_path
        self.panel = PanelKoncowyLTH()

    def start_for_every_kompresja_subfolder(self):
        for name in os.listdir(self.main_folder_path):
            sub = os.path.join(self.main_folder_path, name)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "kompresja.txt")):
                self.panel.analiza_folder_path(sub)

if __name__ == "__main__":
    main_folder_path = "/home/miku/PycharmProjects/Pracainzynierska/wyniki_eksperymentow/kompresja"
    runnner = LTHRunner(main_folder_path)
    runnner.start_for_every_kompresja_subfolder()
