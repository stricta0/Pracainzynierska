from typing import List, Iterable, Optional
from odczytywanie_plikow_tekstowych import TextFileReader
import os

class FileMenadger:

    @staticmethod
    def _get_all_files_paths_from_directory(
            directory_path: str,
            ignorowane_rozszerzenia: Optional[Iterable[str]] = None,
            ignore_file_names: Optional[Iterable[str]] = None,
            ignorowane_foldery: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """
        Zwraca listę *absolutnych* ścieżek do wszystkich plików w podanym katalogu (rekurencyjnie),
        z filtrami:
          - ignorowane_rozszerzenia: np. ["pdf", "jpg"]  -> pomija pliki o tych rozszerzeniach
          - ignore_file_names: np. ["readme.md", "wynik_koncowy.txt"] -> pomija pliki o tych nazwach (bez ścieżek)
          - ignorowane_foldery: np. ["venv", "__pycache__"] -> nie wchodzi do tych katalogów podczas chodzenia po drzewie
        Porównania są niewrażliwe na wielkość liter.

        :param directory_path: ścieżka do katalogu startowego
        :param ignorowane_rozszerzenia: lista rozszerzeń, które ignorujemy (bez kropki)
        :param ignore_file_names: lista ignorowanych plików (same nazwy, bez ścieżek)
        :param ignorowane_foldery: lista nazw katalogów, do których nie wchodzimy
        :return: posortowana lista absolutnych ścieżek plików
        :raises ValueError: gdy ścieżka nie istnieje lub nie jest katalogiem
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Ścieżka '{directory_path}' nie jest katalogiem lub nie istnieje.")

        ignore_exts = {str(ext).lower().lstrip(".") for ext in (ignorowane_rozszerzenia or [])}
        ignore_names = {os.path.basename(str(n)).lower() for n in (ignore_file_names or [])}
        ignore_dirs = {os.path.basename(str(d)).lower() for d in (ignorowane_foldery or [])}

        file_paths: List[str] = []
        # topdown=True + modyfikacja `dirs[:]` pozwala przycinać rekurencję
        for root, dirs, files in os.walk(directory_path, topdown=True, followlinks=False):
            # PRUNING: usuń katalogi, do których nie chcemy wchodzić
            if ignore_dirs:
                dirs[:] = [d for d in dirs if d.lower() not in ignore_dirs]

            for name in files:
                base = os.path.basename(name)
                base_l = base.lower()

                # pomiń konkretne nazwy plików
                if base_l in ignore_names:
                    continue

                # pomiń po rozszerzeniu
                ext = os.path.splitext(base_l)[1].lstrip(".")  # ".PDF" -> "pdf"
                if ext in ignore_exts:
                    continue

                file_paths.append(os.path.abspath(os.path.join(root, base)))
        file_paths.sort()
        return file_paths

    @staticmethod
    def _get_sort_lista_plikow(lista_plikow, names_for_size_in_model_map):
        def get_size(plik, names_for_size_in_model_map):
            start_info, _, _, _ = TextFileReader.odczytaj_treningowy_plik(plik)
            model_name = start_info["model_name"].lower()
            return int(start_info[names_for_size_in_model_map[model_name]])
        return sorted(lista_plikow, key=lambda x : get_size(x, names_for_size_in_model_map), reverse=True)

    @staticmethod
    def get_fille_list(folder_path, ignore_file_names, ignorowane_rozszerzenia, ignorowane_foldery, names_for_size_in_model_map):
        lista_plikow = FileMenadger._get_all_files_paths_from_directory(folder_path, ignorowane_rozszerzenia=ignorowane_rozszerzenia,
                                                          ignore_file_names=ignore_file_names, ignorowane_foldery=ignorowane_foldery)
        return FileMenadger._get_sort_lista_plikow(lista_plikow, names_for_size_in_model_map)

