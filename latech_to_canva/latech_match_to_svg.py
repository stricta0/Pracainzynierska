#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys
import json


def _clean_math(latex_math_text: str) -> str:
    """
    Usuwa zewnętrzne \\[ \\] lub $$ $$ jeśli są obecne
    i zwraca sam środek wzoru.
    """
    s = latex_math_text.strip()

    # Usuń zewnętrzne \[ \]
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    # Usuń zewnętrzne $$ $$
    if s.startswith(r"$$") and s.endswith(r"$$"):
        s = s[2:-2].strip()

    return s


def latex_math_to_svg(latex_math_text: str, file_name: str) -> Path:
    """
    Kompiluje lateksowy wzór matematyczny do pliku SVG.

    :param latex_math_text: Tekst z LaTeX-em, np. r"\\[ a \\]" albo r"a"
    :param file_name: Nazwa pliku wyjściowego (bez rozszerzenia), np. "phi_real"
    :return: Ścieżka do wygenerowanego pliku SVG
    """
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    tex_path = base_dir / f"{file_name}.tex"
    dvi_path = base_dir / f"{file_name}.dvi"
    svg_path = base_dir / f"{file_name}.svg"
    aux_path = base_dir / f"{file_name}.aux"
    log_path = base_dir / f"{file_name}.log"

    # Wyczyść wzór – usuwamy zewnętrzne \[...\] / $$...$$
    inner_math = _clean_math(latex_math_text)

    # Zabezpieczenie pod % w Pythonowym %s
    inner_math_for_fmt = inner_math.replace("%", "%%")

    # Minimalny dokument LaTeX z klasą standalone – sami robimy \[...\]
    tex_content = r"""\documentclass[border=2pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
\[
%s
\]
\end{document}
""" % inner_math_for_fmt

    # Zapisujemy plik .tex
    tex_path.write_text(tex_content, encoding="utf-8")

    # 1. Kompilacja LaTeX -> DVI
    latex_cmd = ["latex", "-interaction=nonstopmode", tex_path.name]
    print("Uruchamiam:", " ".join(latex_cmd))
    latex_result = subprocess.run(
        latex_cmd,
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Zapisz log uruchomienia latex do logs/
    (logs_dir / f"{file_name}_latex_run.log").write_text(
        latex_result.stdout.decode(errors="ignore")
        + "\n\nSTDERR:\n"
        + latex_result.stderr.decode(errors="ignore"),
        encoding="utf-8",
    )

    if latex_result.returncode != 0:
        print("OSTRZEŻENIE: latex zwrócił kod != 0", file=sys.stderr)

    if not dvi_path.exists():
        raise FileNotFoundError(f"Nie odnaleziono pliku DVI: {dvi_path}")

    # 2. Konwersja DVI -> SVG
    dvisvgm_cmd = [
        "dvisvgm",
        dvi_path.name,
        "-n",  # --no-fonts: konwersja fontów na krzywe
        "-o",
        svg_path.name,
    ]
    print("Uruchamiam:", " ".join(dvisvgm_cmd))
    dvisvgm_result = subprocess.run(
        dvisvgm_cmd,
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Zapisz log uruchomienia dvisvgm do logs/
    (logs_dir / f"{file_name}_dvisvgm_run.log").write_text(
        dvisvgm_result.stdout.decode(errors="ignore")
        + "\n\nSTDERR:\n"
        + dvisvgm_result.stderr.decode(errors="ignore"),
        encoding="utf-8",
    )

    if dvisvgm_result.returncode != 0:
        print("Błąd podczas konwersji DVI -> SVG (dvisvgm):", file=sys.stderr)
        raise subprocess.CalledProcessError(
            dvisvgm_result.returncode, dvisvgm_cmd, dvisvgm_result.stdout, dvisvgm_result.stderr
        )

    if not svg_path.exists():
        raise FileNotFoundError(f"Nie odnaleziono pliku SVG: {svg_path}")

    # Przenieś latexowy plik .log do katalogu logs (jeśli istnieje)
    if log_path.exists():
        target_log = logs_dir / log_path.name
        try:
            log_path.replace(target_log)
        except Exception as e:
            print(f"OSTRZEŻENIE: nie udało się przenieść {log_path} do {target_log}: {e}", file=sys.stderr)

    # Sprzątanie plików pomocniczych
    for p in (aux_path, dvi_path):
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                print(f"OSTRZEŻENIE: nie udało się usunąć {p}: {e}", file=sys.stderr)

    print(f"OK, wygenerowano SVG: {svg_path}")
    return svg_path


def main():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.json"

    if not config_path.exists():
        print(f"Brak pliku konfiguracyjnego: {config_path}", file=sys.stderr)
        sys.exit(1)

    config_data = json.loads(config_path.read_text(encoding="utf-8"))

    # Oczekujemy prostego JSON-a:
    # {
    #   "file_name": "phi_real",
    #   "latex_math_text": "...wzór..."
    # }
    try:
        file_name = config_data["file_name"]
        latex_math_text = config_data["latex_math_text"]
    except KeyError as e:
        print(f"Brakuje klucza w config.json: {e}", file=sys.stderr)
        sys.exit(1)

    latex_math_to_svg(latex_math_text, file_name)


if __name__ == "__main__":
    main()
