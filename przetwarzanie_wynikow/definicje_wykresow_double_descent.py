from odczytywanie_plikow_tekstowych import TextFileReader


class DefinicjeWykresowDoubleDescent:

    def __init__(self, files_list):
        self.files_list = files_list

        #co uznajemy za "size" modelu
        self.names_for_size_in_model_map = {"mlp" : "hidden", "mlp2": "hidden", "rff" : "n_features(N)", "cnn": "ch1"}

        self.size_tab = [] #tablica wielkosci modeli

        # TABLICE Z DANYMI Z FOLDERU double_descent_folder_path
        self.train_loss_tab_best_val_acc = [] # train_loss dla epoki o najwyższym val_acc
        self.val_loss_tab_best_val_acc = [] # vall_loss -||-
        self.val_acc_tab_best_val_acc = [] # vall_loss -||-

        self.train_loss_tab_best_val_loss = [] # train_loss dla epoki o najniższym val loss
        self.val_loss_tab_best_val_loss = [] # vall_loss -||-
        self.val_acc_tab_best_val_loss = []  # vall_loss -||-
        self._odczytaj_dane_z_plikow_tekstowych_wykresy_po_wielkosci_modelu() #zbieranie danych

    def _get_model_name_and_dataset(self, lista_plikow):
        plik = lista_plikow[0]
        start_info, _, _, _ = TextFileReader.odczytaj_treningowy_plik(plik)
        return start_info["model_name"], start_info["dataset"]

    # best model definiujemy na dwa sposoby: najniższy val_loss lub najwyższy val_acc
    def get_dane_do_wszystkich_wykresow_best_model(self):
        # dla kazdego modelu wyciagamy dwie epoki - najlepsza po accuracy i po val_loss - następnie definiujemy z tego wykresy
        if len(self.size_tab) == 0: #jesli nie wczytano folderu, zrob to
            self._odczytaj_dane_z_plikow_tekstowych_wykresy_po_wielkosci_modelu()
        assert len(self.size_tab) > 0, "size_tab jest pusty, nie udało się wczytać folderu"
        x = [
            {"tab": self.size_tab, "name": "model size"},
            {"tab": self.size_tab, "name": "model size"},
        ]
        x_poj = [
            {"tab": self.size_tab, "name": "model size"},
        ]

        y_valacc_best_val_acc = [
            {"tab": self.train_loss_tab_best_val_acc, "name": "strata treningowa"},
            {"tab": self.val_acc_tab_best_val_acc, "name": "dokładność testowa"},
        ]
        y_valacc_best_val_loss = [
            {"tab": self.train_loss_tab_best_val_loss, "name": "strata treningowa"},
            {"tab": self.val_acc_tab_best_val_loss, "name": "dokładność testowa"},
        ]
        y_valloss_best_val_acc = [
            {"tab": self.train_loss_tab_best_val_acc, "name": "strata treningowa"},
            {"tab": self.val_loss_tab_best_val_acc, "name": "strata testowa"},
        ]
        y_valloss_best_val_loss = [
            {"tab": self.train_loss_tab_best_val_loss, "name": "strata treningowa"},
            {"tab": self.val_loss_tab_best_val_loss, "name": "strata testowa"},
        ]

        y_valacc_best_val_acc_poj = [
            {"tab": self.val_acc_tab_best_val_acc, "name": "dokładność testowa"},
        ]
        y_valacc_best_val_loss_poj = [
            {"tab": self.val_acc_tab_best_val_loss, "name": "dokładność testowa"},
        ]
        y_valloss_best_val_acc_poj = [
            {"tab": self.val_loss_tab_best_val_acc, "name": "strata testowa"},
        ]
        y_valloss_best_val_loss_poj = [
            {"tab": self.val_loss_tab_best_val_loss, "name": "strata testowa"},
        ]

        y_train_loss_best_val_acc = [
            {"tab": self.train_loss_tab_best_val_acc, "name": "strata treningowa"},
        ]
        y_train_loss_best_val_loss = [
            {"tab": self.train_loss_tab_best_val_loss, "name": "strata treningowa"},
        ]


        model_name, dataset = self._get_model_name_and_dataset(self.files_list)
        dane_do_wykresu = [
            {
                "x": x,
                "y": y_valacc_best_val_acc,
                "title": f"Dokładność testowa i strata treningowa w zależności od rozmiaru modelu \n(checkpoint: najlepsza dokładność testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_acc_vs_train_loss_best_test_acc_model_{model_name}_{dataset}",
                "type": "acc",
                "best_at": "val_acc",
                "add_line_at_zero": True,
            },
            {
                "x": x,
                "y": y_valacc_best_val_loss,
                "title": f"Dokładność testowa i strata treningowa w zależności od rozmiaru modelu \n(checkpoint: najmniejsza strata testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_acc_vs_train_loss_best_test_loss_model_{model_name}_{dataset}",
                "type": "acc",
                "best_at": "val_loss",
            },
            {
                "x": x,
                "y": y_valloss_best_val_acc,
                "title": f"Strata testowa i treningowa w zależności od rozmiaru modelu \n(checkpoint: najlepsza dokładność testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_loss_vs_train_loss_best_test_acc_model_{model_name}_{dataset}",
                "type": "loss",
                "best_at": "val_acc",
                "add_line_at_zero": True,
            },
            {
                "x": x,
                "y": y_valloss_best_val_loss,
                "title": f"Strata testowa i treningowa w zależności od rozmiaru modelu \n(checkpoint: najmniejsza strata testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_loss_vs_train_loss_best_test_loss_model_{model_name}_{dataset}",
                "type": "loss",
                "best_at": "val_loss",
                "add_line_at_zero": True,
            },

            {
                "x": x_poj,
                "y": y_valacc_best_val_acc_poj,
                "title": f"Dokładność testowa w zależności od rozmiaru modelu \n(checkpoint: najlepsza dokładność testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_acc_best_test_acc_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_acc",
            },
            {
                "x": x_poj,
                "y": y_valacc_best_val_loss_poj,
                "title": f"Dokładność testowa w zależności od rozmiaru modelu \n(checkpoint: najmniejsza strata testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_acc_best_test_loss_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_loss",
            },
            {
                "x": x_poj,
                "y": y_valloss_best_val_acc_poj,
                "title": f"Strata testowa w zależności od rozmiaru modelu \n(checkpoint: najlepsza dokładność testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_loss_best_test_acc_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_acc",
                "add_line_at_zero": True,
            },
            {
                "x": x_poj,
                "y": y_valloss_best_val_loss_poj,
                "title": f"Strata testowa w zależności od rozmiaru modelu \n(checkpoint: najmniejsza strata testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_loss_best_test_loss_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_loss",
                "add_line_at_zero": True,
            },
            {
                "x": x_poj,
                "y": y_train_loss_best_val_acc,
                "title": f"Strata treningowa w zależności od rozmiaru modelu \n(checkpoint: najlepsza dokładność testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"train_loss_best_test_acc_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_acc",
                "add_line_at_zero": True,
            },
            {
                "x": x_poj,
                "y": y_train_loss_best_val_loss,
                "title": f"Strata treningowa w zależności od rozmiaru modelu \n(checkpoint: najmniejsza strata testowa), model: {model_name}, zbiór: {dataset}",
                "file_name": f"train_loss_best_test_loss_model_{model_name}_{dataset}",
                "type": "single",
                "best_at": "val_loss",
                "add_line_at_zero": True,
            },
        ]

        return dane_do_wykresu

    def get_dane_do_wszystkich_wykresow_po_liczbie_epok(self):
        # 1 linia = 1 wielkość modelu (1 plik tekstowy)
        # oś x - liczba epok
        # oś y - val_acc / val_loss
        if len(self.size_tab) == 0: #jesli nie wczytano folderu, zrob to
            self._odczytaj_dane_z_plikow_tekstowych_wykresy_po_wielkosci_modelu()
        assert len(self.size_tab) > 0, "size_tab jest pusty, nie udało się wczytać folderu"
        x = []
        y_acc = []
        y_loss = []
        # pojedynczy y to 1 linia dla danego modelu (pliku tekstowego) wyciagniete jego osiagi po epokach !
        for path in self.files_list:
            start_info, wartosci_na_epokach, max_val_acc_epoka_nr, min_train_loss_epoka_nr = TextFileReader.odczytaj_treningowy_plik(path)
            model_name = start_info["model_name"].lower()
            model_size_name = self.names_for_size_in_model_map[model_name]
            model_size = start_info[model_size_name] # to jest w gruncie rzeczy nazwa tej nitki w wykresie

            lista_epok = list(wartosci_na_epokach.keys()) # x tej lini
            lista_val_acc = [] # y tej lini val acc
            lista_val_loss = [] # y tej lini val loss
            for epoka in lista_epok:
                warotsci = wartosci_na_epokach[epoka]
                lista_val_acc.append(warotsci["val_acc"])
                lista_val_loss.append(warotsci["val_loss"])

            x.append({"tab": lista_epok.copy(), "name": "epoka"})
            y_acc.append({"tab": lista_val_acc.copy(), "name": f"size {model_size}", "size": float(model_size)})
            y_loss.append({"tab": lista_val_loss.copy(), "name": f"size {model_size}", "size": float(model_size)})

        model_name, dataset = self._get_model_name_and_dataset(self.files_list)
        dane_do_wykresu = [
            {
                "x": x,
                "y": y_acc,
                "title": f"Dokładność testowa w kolejnych epokach dla modeli o różnym rozmiarze\n model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_acc_vs_epoch_{model_name}_{dataset}",
                "type": "acc",
                "y_name": "dokładność testowa",
            },
            {
                "x": x,
                "y": y_loss,
                "title": f"Strata testowa w kolejnych epokach dla modeli o różnym rozmiarze\n model: {model_name}, zbiór: {dataset}",
                "file_name": f"test_loss_vs_epoch_{model_name}_{dataset}",
                "type": "loss",
                "y_name": "strata testowa",
            },
        ]

        return dane_do_wykresu

    def _odczytaj_dane_z_plikow_tekstowych_wykresy_po_wielkosci_modelu(self):
        self.size_tab.clear() #size tab jest ogolny bo wielkosci sa takie same niezaelznie jak patrzysz
        self.train_loss_tab_best_val_acc.clear()
        self.val_loss_tab_best_val_acc.clear()
        self.train_loss_tab_best_val_loss.clear()
        self.val_loss_tab_best_val_loss.clear()
        self.val_acc_tab_best_val_acc.clear()
        self.val_acc_tab_best_val_loss.clear()

        for path in self.files_list:
            start_info, wartosci_na_epokach, max_val_acc_epoka_nr, min_train_loss_epoka_nr = TextFileReader.odczytaj_treningowy_plik(path)
            model_name = start_info["model_name"].lower()
            model_size_name = self.names_for_size_in_model_map[model_name]
            model_size = start_info[model_size_name]
            self.size_tab.append(model_size)

            info_best_val_acc = wartosci_na_epokach[max_val_acc_epoka_nr]
            self.train_loss_tab_best_val_acc.append(info_best_val_acc["train_loss"])
            self.val_loss_tab_best_val_acc.append(info_best_val_acc["val_loss"])
            self.val_acc_tab_best_val_acc.append(info_best_val_acc["val_acc"])

            info_best_val_locc = wartosci_na_epokach[min_train_loss_epoka_nr]
            self.train_loss_tab_best_val_loss.append(info_best_val_locc["train_loss"])
            self.val_loss_tab_best_val_loss.append(info_best_val_locc["val_loss"])
            self.val_acc_tab_best_val_loss.append(info_best_val_locc["val_acc"])

