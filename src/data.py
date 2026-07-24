"""
Carga y manejo del dataset crudo de marcha (.npy).

Estructura de cada .npy (segun c3d2np.py del repo original):
    X.shape == (num_frames, 19, 4)
    ultimo eje = [x, y, z, flag_validez]  ->  flag == -1 significa marcador invalido
    coordenadas en milimetros.

La etiqueta (clase) y el sujeto salen del PATH, no del array:
    base_folder / clase / sujeto / trial.npy

Criterio de calidad (ver EDA):
    - Los huecos en los BORDES (inicio/fin) son sujetos entrando/saliendo del
      volumen de captura; NO se tocan, los recorta la segmentacion de pasos (etapa B).
    - Los huecos INTERIORES cortos (<= max_gap) se interpolan linealmente.
    - Los huecos INTERIORES largos hacen que el trial no sea usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Orden FIJO de los 19 marcadores tal cual los escribe c3d2np.py.
# No reordenar: los .npy dependen de este orden.
MARKERS = [
    "C7", "LA", "RA", "REP", "LEP", "RUL", "LUL", "RASIS", "LASIS", "RPSIS",
    "LPSIS", "RGT", "LGT", "RLE", "LLE", "RCA", "LCA", "RFM", "LFM",
]
MARKER_IDX = {name: i for i, name in enumerate(MARKERS)}

INVALID_FLAG = -1


@dataclass
class Trial:
    """Un trial (una caminata): el array ya cargado + su metadata del path."""
    label: int       # clase 0-3 (form 1-4 del paper)
    subject: str     # id de sujeto (nombre de carpeta)
    trial_id: str    # stem del archivo
    X: np.ndarray    # (num_frames, 19, 4) en mm, cargado directo


def load_dataset(base_folder: str | Path) -> list[Trial]:
    """Carga todos los trials bajo base_folder/clase/sujeto/*.npy en memoria.

    La clase y el sujeto se leen del path, no del array. Costo en RAM:
    ~300 KB por trial -> ~350 MB para los 1139. Bien para una laptop; si en
    la etapa C aprieta la RAM, cambiar a streaming.
    """
    base_folder = Path(base_folder)
    paths = sorted(base_folder.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron .npy bajo {base_folder} con patron clase/sujeto/*.npy"
        )
    return [
        Trial(label=int(p.parent.parent.name), subject=p.parent.name,
              trial_id=p.stem, X=np.load(p))
        for p in paths
    ]


def valid_mask(X: np.ndarray) -> np.ndarray:
    """Mascara booleana (num_frames, 19): True donde el marcador es valido."""
    return X[..., 3] != INVALID_FLAG


def masked_coords(X: np.ndarray) -> np.ndarray:
    """Coordenadas xyz con NaN donde el marcador es invalido. Util para plots/estadistica."""
    xyz = X[..., :3].astype(float).copy()
    xyz[~valid_mask(X)] = np.nan
    return xyz


def _interior_invalid_runs(valid_1d: np.ndarray) -> list[tuple[int, int]]:
    """Runs (start, end) inclusive de frames invalidos INTERIORES de un marcador.

    Excluye las colas invalidas del inicio/fin (bordes). Si el marcador nunca
    es valido, no hay interior -> lista vacia.

    Unico punto donde se decide que frames son "interior" (analyze_gaps e
    interpolate_short_gaps parten de aqui, para no duplicar el criterio).
    """
    idx = np.where(valid_1d)[0]
    if len(idx) == 0:
        return []
    first, last = idx[0], idx[-1]
    runs = []
    run_start = None
    for i in range(first, last + 1):
        if not valid_1d[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
    return runs


def analyze_gaps(X: np.ndarray) -> tuple[int, int]:
    """Peor hueco de BORDE y peor hueco INTERIOR del trial (maximo entre marcadores).

    El borde es benigno (lo recorta la segmentacion de pasos); el interior es
    el que decide interpolar o descartar en interpolate_short_gaps.
    Devuelve (worst_edge, worst_interior).
    """
    vm = valid_mask(X)
    n_markers = X.shape[1]
    worst_edge, worst_interior = 0, 0
    for j in range(n_markers):
        valid_1d = vm[:, j]
        idx = np.where(valid_1d)[0]
        if len(idx) == 0:
            worst_edge = max(worst_edge, len(valid_1d))
            continue
        edge = int(idx[0] + (len(valid_1d) - 1 - idx[-1]))
        runs = _interior_invalid_runs(valid_1d)
        interior = max((e - s + 1 for s, e in runs), default=0)
        worst_edge = max(worst_edge, edge)
        worst_interior = max(worst_interior, interior)
    return worst_edge, worst_interior


def interpolate_short_gaps(
    X: np.ndarray, max_gap: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Interpola linealmente los huecos INTERIORES cortos (<= max_gap) por marcador.

    - No toca bordes (no hay como interpolar sin extrapolar).
    - No toca huecos interiores largos (> max_gap): se dejan invalidos.
    - Marca como validos (flag=1) solo los frames efectivamente rellenados.

    Devuelve (X_nuevo, filled_por_marcador) donde filled_por_marcador cuenta
    cuantos frames se fabricaron en cada marcador -> trazabilidad clinica.
    IMPORTANTE: no muta X; trabaja sobre una copia.
    """
    X = X.copy()
    vm = valid_mask(X)
    n_markers = X.shape[1]
    filled: dict[str, int] = {}

    for j in range(n_markers):
        valid_1d = vm[:, j]
        runs = _interior_invalid_runs(valid_1d)
        n_filled = 0
        for start, end in runs:
            length = end - start + 1
            if length > max_gap:
                continue  # hueco largo: no interpolar
            lo, hi = start - 1, end + 1  # frames validos que bracketean (existen: es interior)
            for c in range(3):  # x, y, z
                X[start:end + 1, j, c] = np.interp(
                    np.arange(start, end + 1), [lo, hi], [X[lo, j, c], X[hi, j, c]]
                )
            X[start:end + 1, j, 3] = 1  # marcar como validos
            n_filled += length
        if n_filled:
            filled[MARKERS[j]] = n_filled

    return X, filled
