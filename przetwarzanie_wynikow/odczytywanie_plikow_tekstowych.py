class TextFileReader:


    @staticmethod
    def odczytaj_treningowy_plik(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            plik = f.read().strip().split("\n")
            start_info = TextFileReader._read_start_data_plik_double_descent(plik)
            wartosci_na_epokach = TextFileReader._stworz_slownik_wynikow_double_descent(plik)
            max_val_acc_epoka_nr, min_vall_loss_epoka_nr = TextFileReader._find_best_val_acc_and_val_loss_epochs(wartosci_na_epokach)
        return start_info, wartosci_na_epokach, max_val_acc_epoka_nr, min_vall_loss_epoka_nr

    @staticmethod
    def odczytaj_plik_kompresja(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            plik = f.read().strip().split("\n")
        pierwsza_linia = plik.pop(0).strip().split("|")
        statystyki = {}
        for element in pierwsza_linia:
            try:
                element = element.strip().split("=")
                statystyki[element[0].strip()] = element[1].strip()
            except Exception as e:
                print(f"podczas odczytywania pliku doszlo do wyjatku: {e}")
        wszystkie_wartosci = {}
        for linia in plik:
            linia_copy = linia
            linia = linia.strip().split("|")
            index = int(linia.pop(0).strip())
            tryb = linia.pop(0).strip()
            val_acc = float(linia.pop().strip().split(":")[1].strip())
            val_loss = float(linia.pop().strip().split(":")[1].strip())

            rozmiary = linia
            rozmiary_koncowe = {}
            pierwszy_rozmiar = None
            for rozmiar in rozmiary:
                rozmiar = rozmiar.strip().split(" ")
                name = rozmiar[0].strip()
                fraction = {"licznik": int(rozmiar[1].strip().split("/")[0]), "mianownik": int(rozmiar[1].strip().split("/")[1])}
                in_percentage = float(rozmiar[2].strip()[1:-2])
                rozmiary_koncowe[name] = {"fraction": fraction, "in_percentage": in_percentage}
                if pierwszy_rozmiar is None:
                    pierwszy_rozmiar = in_percentage
            wszystkie_wartosci_obj = {
                "linia": linia_copy,
                "tryb": tryb,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "rozmiar": pierwszy_rozmiar,
                "rozmiary": rozmiary_koncowe,
            }
            wszystkie_wartosci[index] = wszystkie_wartosci_obj
        return statystyki, wszystkie_wartosci




    @staticmethod
    def _stworz_slownik_wynikow_double_descent(file, start_data_line_index=3):

        epoch = 1
        lines_for_epochs = {}
        for i in range(start_data_line_index, len(file) - 1):
            line = file[i]
            line_dict = TextFileReader._line_to_dict_double_descent_plik(line)
            lines_for_epochs[epoch] = line_dict
            epoch += 1
        return lines_for_epochs

    @staticmethod
    def _read_start_data_plik_double_descent(file, start_data_line_index=2):
        start_data = file[start_data_line_index].strip().split(" | ")
        start_data_dict = {}
        for element in start_data:
            elements = element.split("=")
            if len(elements) != 2:
                continue
            start_data_dict[elements[0]] = TextFileReader._cast_value_into_fitting_data_type(elements[1])
        return start_data_dict

    @staticmethod
    # =============== PRZETWARZANIE PLIKOW WYNIKOWYCH DOUBLE DESCENT =========================
    def _line_to_dict_double_descent_plik(line):
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
            value = TextFileReader._cast_value_into_fitting_data_type(value)
            res_dict[elements[0]] = value

        return res_dict

    @staticmethod
    def _is_int(s):
        """
        sprawdza czy s jest intem
        """
        try:
            int(s.strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def _cast_value_into_fitting_data_type(value):
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
        elif TextFileReader._is_int(value):
            value = int(value)
        return value

    @staticmethod
    def _find_best_val_acc_and_val_loss_epochs(slownik_po_epokach):
        max_val_acc = 0.0
        min_train_loss = float("inf")
        max_val_acc_epoka_nr = -1
        min_train_loss_epoka_nr = -1
        for epoka in slownik_po_epokach:
            if slownik_po_epokach[epoka]["val_acc"] > max_val_acc:
                max_val_acc = slownik_po_epokach[epoka]["val_acc"]
                max_val_acc_epoka_nr = epoka
            if slownik_po_epokach[epoka]["val_loss"] < min_train_loss:
                min_train_loss = slownik_po_epokach[epoka]["val_loss"]
                min_train_loss_epoka_nr = epoka
        return max_val_acc_epoka_nr, min_train_loss_epoka_nr


if __name__ == "__main__":
    plik_testowy = "mlp_fashion_hidden20.txt"
    txt_file_reader = TextFileReader()
    print(txt_file_reader.odczytaj_treningowy_plik(plik_testowy)[3])