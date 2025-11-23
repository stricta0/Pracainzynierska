from odczytywanie_plikow_tekstowych import TextFileReader

class CreateStatisticsFile:

    def __init__(self, files_list, names_for_size_in_model_map):
        self.files_list = files_list
        self.names_for_size_in_model_map = names_for_size_in_model_map

    # def get_data_for_file(self):
    #     best_val_acc_file = "model_size | train_loss | val_loss | val_acc\n"
    #     best_val_loss_file = "model_size | train_loss | val_loss | val_acc\n"
    #
    #     for file in self.files_list:
    #         start_info, wartosci_na_epokach, max_val_acc_epoka_nr, min_train_loss_epoka_nr = TextFileReader.odczytaj_treningowy_plik(file)
    #         best_val_acc_file += f"{self._get_size(file)} | {wartosci_na_epokach[max_val_acc_epoka_nr]["train_loss"]} | {wartosci_na_epokach[max_val_acc_epoka_nr]["val_loss"]} | {wartosci_na_epokach[max_val_acc_epoka_nr]["val_acc"]}\n"
    #         best_val_loss_file += f"{self._get_size(file)} | {wartosci_na_epokach[min_train_loss_epoka_nr]["train_loss"]} | {wartosci_na_epokach[min_train_loss_epoka_nr]["val_loss"]} | {wartosci_na_epokach[min_train_loss_epoka_nr]["val_acc"]}\n"
    #
    #     return best_val_acc_file, best_val_loss_file

    def get_data_from_files(self):
        best_val_acc_file = {"model_size" : [], "train_loss": [], "val_loss": [], "val_acc": [], "epoch": []}
        best_val_loss_file = {"model_size" : [], "train_loss": [], "val_loss": [], "val_acc": [], "epoch": []}

        for file in self.files_list:
            start_info, wartosci_na_epokach, max_val_acc_epoka_nr, min_train_loss_epoka_nr = TextFileReader.odczytaj_treningowy_plik(file)

            best_val_acc_file["model_size"].append(self._get_size(file))
            best_val_acc_file["train_loss"].append(wartosci_na_epokach[max_val_acc_epoka_nr]["train_loss"])
            best_val_acc_file["val_loss"].append(wartosci_na_epokach[max_val_acc_epoka_nr]["val_loss"])
            best_val_acc_file["val_acc"].append(wartosci_na_epokach[max_val_acc_epoka_nr]["val_acc"])
            best_val_acc_file["epoch"].append(max_val_acc_epoka_nr)

            best_val_loss_file["model_size"].append(self._get_size(file))
            best_val_loss_file["train_loss"].append(wartosci_na_epokach[min_train_loss_epoka_nr]["train_loss"])
            best_val_loss_file["val_loss"].append(wartosci_na_epokach[min_train_loss_epoka_nr]["val_loss"])
            best_val_loss_file["val_acc"].append(wartosci_na_epokach[min_train_loss_epoka_nr]["val_acc"])
            best_val_loss_file["epoch"].append(min_train_loss_epoka_nr)
        return best_val_acc_file, best_val_loss_file

    def sort_file_dict(self, file_dict, key_name="val_acc", reverse=False):
        indekses = [i for i in range(len(file_dict["model_size"]))]
        posortowane_indeksy = sorted(indekses, key=lambda x: file_dict[key_name][x], reverse=reverse)

        def change_tab_with_indekses(tab, indekses):
            new_tab = []
            for indeks in indekses:
                new_tab.append(tab[indeks])
            return new_tab

        for key in file_dict.keys():
            file_dict[key] = change_tab_with_indekses(file_dict[key], posortowane_indeksy)
        return file_dict

    def create_string_from_dict(self, dict_file):
        # dodawanie pierwszej lini - na zasadzie model_size | train_loss | val_loss | val_acc
        best_val_acc_file = ""
        for key in dict_file:
            if best_val_acc_file == "":
                best_val_acc_file += f"{key}"
            else:
                best_val_acc_file += f" | {key}"
        best_val_acc_file += f"\n"

        # dodawanie danych
        for i in range(len(dict_file["model_size"])):
            line = ""
            for key in dict_file:
                if line == "":
                    line += f"{dict_file[key][i]}"
                else:
                    line += f" | {dict_file[key][i]}"
            line += f"\n"
            best_val_acc_file += line
        return best_val_acc_file

    def _get_size(self, plik):
        #names_for_size_in_model_map = {"mlp": "hidden", "mlp2": "hidden", "rff": "n_features", "cnn": "ch1"}
        start_info, _, _, _ = TextFileReader.odczytaj_treningowy_plik(plik)
        model_name = start_info["model_name"].lower()
        return int(start_info[self.names_for_size_in_model_map[model_name]])