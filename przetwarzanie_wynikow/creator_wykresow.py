from typing import Tuple, Sequence, Mapping, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl



class KreatorWykresow:
    """
    Przykład użycia (dwie serie: train i val):

        x = [
            {"tab": hidden_size_tab, "name": "hidden size"},
            {"tab": hidden_size_tab, "name": "hidden size"},
        ]
        y = [
            {"tab": train_loss_tab, "name": "train loss"},
            {"tab": val_loss_tab,   "name": "validation loss"},
        ]

        fig, ax = CreatorWykresow.create_plot(
            x, y,
            plot_name=f"Wpływ rozmiaru warstwy ukrytej na metryki {algo_name} ({data_set})",
            x_name="rozmiar warstwy ukrytej",
            y_name="wartości metryk",
            marker="o",
            # --- nowości ---
            # yscale / xscale: "linear" | "log" | "symlog" | "logit"
            yscale="symlog", yscale_linthresh=1e-3,  # symlog: liniowo blisko 0, logarytm dalej
            xscale="linear",
            add_line_at_zero=True
        )

    DOSTĘPNE SKALE (dla obu osi):
      - "linear"  : skala liniowa (domyślna)
      - "log"     : logarytmiczna; wymaga wszystkich wartości > 0
      - "symlog"  : „semi-log” z liniową strefą wokół 0; działa także dla wartości ujemnych
                    (parametry: linthresh>0 i linscale>0)
      - "logit"   : skala logit; typowo dla danych w (0,1); wymaga 0 < wartości < 1

    ZACHOWANA KOMPATYBILNOŚĆ:
      - Jeśli podasz logarytmic_scale_y=True lub logarytmic_scale_x=True i jednocześnie
        yscale/xscale pozostawisz domyślne ("linear"), to zostanie wymuszone "log".
      - Jeśli jawnie ustawisz yscale/xscale, to one mają pierwszeństwo nad flagami
        logarytmic_scale_*.
    """
    @staticmethod
    def create_plot(
        x: Sequence[Mapping[str, Any]],
        y: Sequence[Mapping[str, Any]],
        plot_name: str,
        x_name: str,
        y_name: str,
        save: bool = True,
        save_name: str = "przykladowy_wykres",
        *,
        # --- estetyka (domyślne wartości można nadpisać) ---
        figsize: Tuple[float, float] = (7.0, 5.0),
        line_style: str = "-",
        line_width: float = 2.0,
        marker: str = "",            # np. "o", "s", "^"; pusty = bez markerów
        marker_size: float = 6.0,
        alpha: float = 1.0,
        grid: bool = True,
        grid_style: str = "--",
        grid_alpha: float = 0.3,
        title_size: int = 14,
        label_size: int = 12,
        tick_size: int = 10,
        # --- kolor domyślny dla serii bez 'color' ---
        default_color: Optional[str] = None,  # None => cykl Matplotlib
        # --- zapis wektorowy ---
        vector_format: str = "pdf",  # "pdf" | "svg" | "eps"
        transparent: bool = False,
        tight_layout: bool = True,
        # --- ścieżka zapisu ---
        save_path: Optional[str | Path] = None,  # katalog do zapisu (absolutny lub względny)
        odwrocona_os_x: bool = False,
        max_na_osi_x: Optional[float] = None,
        min_na_osi_x: Optional[float] = None,
        # --- skale osi (NOWE) ---
        yscale: str = "linear",                   # "linear" | "log" | "symlog" | "logit"
        yscale_linthresh: float = 1e-3,           # tylko dla "symlog"
        yscale_linscale: float = 1.0,             # tylko dla "symlog"
        xscale: str = "linear",                   # "linear" | "log" | "symlog" | "logit"
        xscale_linthresh: float = 1e-3,           # tylko dla "symlog"
        xscale_linscale: float = 1.0,             # tylko dla "symlog"
        # --- linia zerowa ---
        add_line_at_zero: bool = False,           # True => pozioma przerywana linia na y=0.0
        # --- flagi wsteczne (zachowana kompatybilność) ---
        logarytmic_scale_y: bool = False,         # jeżeli yscale pozostaje "linear", to wymusi "log"
        logarytmic_scale_x: bool = False,          # jeżeli xscale pozostaje "linear", to wymusi "log"

        color_by_size: bool = False,  # True => kolor linii zależy od yd["size"]
        size_scale: str = "log",  # "log" | "linear"
        cmap_name: str = "RdYlGn_r",  # zielony -> czerwony (od małego do dużego)
        show_legend: bool = True,  # pozwala wyłączyć legendę
        colorbar_label: str = "rozmiar modelu",

        ignore_title: bool = False

    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Rysuje wiele linii (x vs y) na jednym wykresie. x i y to listy słowników:
          {"tab": <tablica>, "name": "etykieta", "color": "blue"}
        gdzie color jest opcjonalny.
        Dla każdej serii i: parujemy x[i] z y[i].

        Zwraca (fig, ax). Opcjonalnie zapisuje wektorowo do <save_path>/<save_name>.<format>.

        Parametry zapisu:
        - save_path: katalog docelowy (utworzy się automatycznie). Gdy None -> bieżący katalog.
        - vector_format: "pdf" (domyślnie), "svg" lub "eps".
        """

        # --- Walidacja formatu zapisu ---
        allowed_formats = {"pdf", "svg", "eps"}
        vf = vector_format.lower()
        if vf not in allowed_formats:
            raise ValueError(f"vector_format musi być jednym z {allowed_formats}, a jest '{vector_format}'.")

        # --- Walidacja długości wejść ---
        if len(x) != len(y):
            raise ValueError(f"Liczba serii w x ({len(x)}) musi równać się liczbie serii w y ({len(y)}).")

        # --- Obsługa kompatybilności: flagi logarytmiczne wymuszają 'log' jeśli nie ustawiono jawnej skali ---
        if yscale == "linear" and logarytmic_scale_y:
            yscale = "log"
        if xscale == "linear" and logarytmic_scale_x:
            xscale = "log"

        # --- Walidacja nazw skal ---
        valid_scales = {"linear", "log", "symlog", "logit"}
        if yscale not in valid_scales:
            raise ValueError(f"Nieznana skala Y: '{yscale}'. Dozwolone: {sorted(valid_scales)}")
        if xscale not in valid_scales:
            raise ValueError(f"Nieznana skala X: '{xscale}'. Dozwolone: {sorted(valid_scales)}")

        fig, ax = plt.subplots(figsize=figsize)
        min_x, max_x = float("inf"), float("-inf")

        # Zbierz wszystkie wartości do walidacji skal:
        all_y_values: List[np.ndarray] = []
        all_x_values: List[np.ndarray] = []

        sizes: List[float] = []
        if color_by_size:
            for yd in y:
                if "size" not in yd:
                    raise KeyError("color_by_size=True wymaga, aby każda seria y miała pole 'size' (liczbowe).")
                sizes.append(float(yd["size"]))

            vmin, vmax = float(np.min(sizes)), float(np.max(sizes))
            if size_scale == "log":
                if vmin <= 0:
                    raise ValueError("size_scale='log' wymaga, aby wszystkie rozmiary były > 0.")
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            elif size_scale == "linear":
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                raise ValueError("size_scale musi być 'log' albo 'linear'.")

            cmap = mpl.cm.get_cmap(cmap_name)


        for i, (xd, yd) in enumerate(zip(x, y), start=1):
            if not isinstance(xd, Mapping) or not isinstance(yd, Mapping):
                raise TypeError("Elementy x i y muszą być słownikami.")
            if "tab" not in xd or "tab" not in yd:
                raise KeyError("Każdy słownik musi zawierać klucz 'tab' z tablicą wartości.")

            xi = np.asarray(xd["tab"], dtype=float).ravel()
            yi = np.asarray(yd["tab"], dtype=float).ravel()

            all_x_values.append(xi)
            all_y_values.append(yi)

            min_x = min(min_x, float(np.min(xi)))
            max_x = max(max_x, float(np.max(xi)))

            if xi.shape != yi.shape:
                raise ValueError(f"Seria {i}: różna liczba punktów: len(x)={xi.size}, len(y)={yi.size}.")

            # etykieta i kolor: preferuj dane z 'y', potem z 'x', potem domyślne
            label = yd.get("name") or xd.get("name") or f"linia_{i}"
            if color_by_size:
                color = cmap(norm(float(yd["size"])))
            else:
                color = yd.get("color", xd.get("color", default_color))

            plot_kwargs = dict(
                linestyle=line_style,
                linewidth=line_width,
                marker=(marker if marker else None),
                markersize=marker_size,
                alpha=(0.6 if color_by_size else alpha),
                label=label,
            )
            if color is not None:
                plot_kwargs["color"] = color  # jeśli None, użyj cyklu Matplotlib

            ax.plot(xi, yi, **plot_kwargs)

        # --- Walidacje dla typów skal ---
        def _validate_for_log(vals: np.ndarray, axis_name: str):
            if vals.size and np.any(vals <= 0):
                raise ValueError(f"Skala 'log' na {axis_name} wymaga wartości > 0.")

        def _validate_for_logit(vals: np.ndarray, axis_name: str):
            if vals.size and (np.any(vals <= 0) or np.any(vals >= 1)):
                raise ValueError(f"Skala 'logit' na {axis_name} wymaga wartości w przedziale (0, 1).")

        if yscale == "log":
            _validate_for_log(np.concatenate(all_y_values) if all_y_values else np.array([]), "osi Y")
        if xscale == "log":
            _validate_for_log(np.concatenate(all_x_values) if all_x_values else np.array([]), "osi X")
        if yscale == "logit":
            _validate_for_logit(np.concatenate(all_y_values) if all_y_values else np.array([]), "osi Y")
        if xscale == "logit":
            _validate_for_logit(np.concatenate(all_x_values) if all_x_values else np.array([]), "osi X")

        # --- Ustawianie skal ---
        if yscale == "linear":
            pass
        elif yscale == "log":
            ax.set_yscale("log")
        elif yscale == "symlog":
            if yscale_linthresh <= 0 or yscale_linscale <= 0:
                raise ValueError("Dla 'symlog' parametry yscale_linthresh i yscale_linscale muszą być > 0.")
            ax.set_yscale("symlog", linthresh=yscale_linthresh, linscale=yscale_linscale)
        elif yscale == "logit":
            ax.set_yscale("logit")

        if xscale == "linear":
            pass
        elif xscale == "log":
            ax.set_xscale("log")
        elif xscale == "symlog":
            if xscale_linthresh <= 0 or xscale_linscale <= 0:
                raise ValueError("Dla 'symlog' parametry xscale_linthresh i xscale_linscale muszą być > 0.")
            ax.set_xscale("symlog", linthresh=xscale_linthresh, linscale=xscale_linscale)
        elif xscale == "logit":
            ax.set_xscale("logit")

        # --- Linia na poziomie y=0.0 (przerywana) ---
        if add_line_at_zero:
            ax.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
                zorder=5,  # wyżej niż siatka i większość elementów
            )

        # --- Opisy i estetyka ---
        if not ignore_title:
            ax.set_title(plot_name, fontsize=title_size)

        ax.set_xlabel(x_name, fontsize=label_size)
        ax.set_ylabel(y_name, fontsize=label_size)
        ax.tick_params(labelsize=tick_size)

        if grid:
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha)

        if show_legend and not color_by_size:
            ax.legend()

        if color_by_size:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # Matplotlib tego wymaga dla colorbar
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(colorbar_label, fontsize=label_size)
            cbar.ax.tick_params(labelsize=tick_size)

        # --- Zakres osi X / odwrócenie ---
        if odwrocona_os_x and np.isfinite(min_x) and np.isfinite(max_x):
            if max_na_osi_x is not None and min_na_osi_x is not None:
                ax.set_xlim(max_na_osi_x, min_na_osi_x)  # 100 po lewej, 0 po prawej
            else:
                ax.set_xlim(max_x, min_x)  # 100 po lewej, 0 po prawej
        elif max_na_osi_x is not None and min_na_osi_x is not None:
            ax.set_xlim(min_na_osi_x, max_na_osi_x)

        # --- Układ i zapis ---
        if tight_layout:
            fig.tight_layout()

        if save:
            out_dir = Path.cwd() if save_path is None else Path(save_path).expanduser().resolve()
            if out_dir.exists() and not out_dir.is_dir():
                raise ValueError(f"save_path '{out_dir}' wskazuje na plik, a nie katalog.")
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = out_dir / f"{save_name}.{vf}"
            fig.savefig(
                filename,
                format=vf,
                bbox_inches="tight",
                transparent=transparent,
            )

        return fig, ax
