import numpy as np

class Loger:
    def __init__(self, file_name):
        self.file_name = file_name

    def add_line_to_file(self, line):
        """
        Dopisuje pojedynczą linię tekstu do pliku.
        Jeśli plik nie istnieje, zostanie utworzony automatycznie.

        Args:
            line (str): Tekst, który ma zostać dopisany.
            file_name (str): Nazwa pliku (domyślnie 'wyniki.txt').
        """
        # upewnij się, że linia kończy się znakiem nowej linii
        if not line.endswith("\n"):
            line += "\n"

        with open(self.file_name, "a", encoding="utf-8") as f:
            f.write(line)

    @staticmethod
    def _fmt_arg(v):
        """Estetyczne formatowanie wartości do loga."""
        # liczby zmiennoprzecinkowe
        try:
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.6g}"
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                return str(int(v))
        except Exception:
            if isinstance(v, float):
                return f"{v:.6g}"
            if isinstance(v, int) and not isinstance(v, bool):
                return str(v)

        # iterowalne typu lista/tupla
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(map(str, v)) + "]"

        # None, bool, str i wszystko inne
        return str(v)

    @staticmethod
    def get_args_log_line(args, sep=" | "):
        items = vars(args).items()
        line = sep.join(f"{k}={Loger._fmt_arg(v)}" for k, v in items)

        return line