#!/usr/bin/env bash
# Równoległe uruchamianie komend z pliku (domyślnie runs.txt) z twardym limitem -j (FIFO semafor)
# Użycie:
#   ./run_jobs.sh
#   ./run_jobs.sh -f myruns.txt
#   ./run_jobs.sh -j 6
#   ./run_jobs.sh -f runs.txt -j 6

set -euo pipefail

FILE="runs.txt"
JOBS=5

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      [[ $# -ge 2 ]] || { echo "Błąd: brak argumentu dla $1" >&2; exit 2; }
      FILE="$2"; shift 2 ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || { echo "Błąd: brak argumentu dla $1" >&2; exit 2; }
      JOBS="$2"; shift 2 ;;
    -h|--help)
      echo "Użycie: $0 [-f runs.txt] [-j równoległe_zadania]"
      exit 0 ;;
    *) echo "Nieznany argument: $1" >&2; exit 1 ;;
  esac
done

# --- walidacja pliku z ochroną na pustą nazwę i CR ---
if [[ -z "${FILE// }" ]]; then
  echo "Błąd: pusta nazwa pliku wejściowego (-f/--file)." >&2
  exit 1
fi
# Usuń końcowe \r z nazwy (gdyby przyszło z CRLF)
FILE="${FILE%$'\r'}"

if [[ ! -f "$FILE" ]]; then
  echo "Brak pliku: $FILE" >&2
  exit 1
fi

echo "Plik: $FILE"
echo "Równoległość: $JOBS"

# --- semafor FIFO ---
fifo=$(mktemp -u)
mkfifo "$fifo"
# Deskryptor 3 to nasza „miska z tokenami”
exec 3<>"$fifo"
rm -f "$fifo"
for ((i=0; i<JOBS; i++)); do printf '.' >&3; done

# Sprzątanie po Ctrl+C / SIGTERM
trap 'kill 0 2>/dev/null || true; exec 3>&- 3<&-; exit 130' INT TERM

# Otwórz plik na FD 9 i czytaj linia po linii (bez kruchego "done < $FILE")
exec 9<"$FILE"

# licznik zadań (ładne logi)
n=0
while IFS= read -r line <&9 || [[ -n "$line" ]]; do
  # usuń końcowe \r (CRLF → LF)
  line="${line%$'\r'}"
  # pomiń puste i komentarze
  [[ -z "${line// }" ]] && continue
  [[ "$line" =~ ^# ]] && continue

  # Pobierz token z semafora (blokuje, jeśli brak)
  IFS= read -r -n1 _tok <&3

  ((n+=1))
  {
    echo "[$n] Start: $line"
    # stdbuf: świeże logi
    stdbuf -oL -eL bash -lc "$line"
    rc=$?
    if (( rc != 0 )); then
      echo "[$n] Zakończone z kodem $rc: $line" >&2
    else
      echo "[$n] OK: $line"
    fi
    # oddaj token
    printf '.' >&3
  } &
done

# Zamknij FD 9, poczekaj na dzieci
exec 9<&-
wait
# zamknij semafor
exec 3>&- 3<&-
echo "Wszystkie zadania zakończone."
