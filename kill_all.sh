#!/usr/bin/env bash
# Zabija procesy Pythona pasujące do wzorca w pełnej komendzie.
# Użycie:
#   ./kill_all.sh                          # zabije wszystko z 'train_mlp.py' (po potwierdzeniu)
#   ./kill_all.sh -p "python train_mlp.py --hidden 50 --log_file wyniki_hidden_50"
#   ./kill_all.sh -p "train_mlp.py" -y     # bez pytania
#   ./kill_all.sh -p "train_mlp.py" -s KILL  # twardy kill -9 (uwaga!)

set -euo pipefail

PATTERN="train_mlp.py"
SIGNAL="TERM"
YES="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--pattern) PATTERN="$2"; shift 2 ;;
    -s|--signal)  SIGNAL="$2"; shift 2 ;;  # np. TERM/KILL/INT
    -y|--yes)     YES="true"; shift ;;
    -h|--help)
      echo "Użycie: $0 [-p wzorzec] [-s sygnał] [-y]"
      echo "Przykład: $0 -p \"python train_mlp.py --hidden 50 --log_file wyniki_hidden_50\""
      exit 0
      ;;
    *) echo "Nieznany argument: $1" >&2; exit 1 ;;
  esac
done

echo "Wzorzec: $PATTERN"
echo "Sygnał:  $SIGNAL"
echo "Znajdowane procesy:"
if ! pgrep -fa -- "$PATTERN"; then
  echo "Brak pasujących procesów."
  exit 0
fi

if [[ "$YES" != "true" ]]; then
  read -r -p "Zabić powyższe procesy? [y/N] " ans
  [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]] || { echo "Anulowano."; exit 0; }
fi

# Zabicie procesów
if pkill -"${SIGNAL}" -f -- "$PATTERN"; then
  echo "Wysłano SIG${SIGNAL} do procesów."
else
  echo "pkill nie znalazł procesów lub wystąpił błąd (być może już zniknęły)."
fi

sleep 0.5
if pgrep -fa -- "$PATTERN" >/dev/null; then
  echo "Pozostałe procesy:"
  pgrep -fa -- "$PATTERN" || true
  echo "Jeśli trzeba wymusić, użyj: $0 -p \"$PATTERN\" -s KILL -y"
else
  echo "Wszystkie procesy ubite."
fi
