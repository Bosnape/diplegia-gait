"""
Calculo de los 27 angulos articulares definidos en la Tabla 5 del paper
(Ferrari et al., 2019). Cada angulo se define por un triplete de marcadores
(I, II, III), siendo II el vertice.

El paper proyecta ademas cada angulo sobre los 3 planos del cuerpo -> 81 angulos
por frame. Aca dejamos el angulo 3D directo; la proyeccion a planos queda
pendiente porque requiere fijar el sistema de referencia del cuerpo (etapa de features).
"""
from __future__ import annotations

import numpy as np

try:
    from .data import MARKER_IDX  # como paquete:  python -m src.angles
except ImportError:
    from data import MARKER_IDX   # directo:        python src/angles.py

# Los 27 tripletes (Marker I, Marker II [vertice], Marker III) tal cual la Tabla 5.
# Transcritos del paper; verificar 1:1 antes de confiar en resultados finales.
ANGLE_TRIPLETS: list[tuple[str, str, str]] = [
    ("LGT", "LPSIS", "LLE"),
    ("LLE", "LGT", "LCA"),
    ("LCA", "LLE", "LFM"),
    ("LEP", "LA", "LUL"),
    ("LEP", "C7", "LUL"),
    ("LLE", "LASIS", "LFM"),
    ("LA", "C7", "LEP"),
    ("RGT", "RPSIS", "RLE"),
    ("RLE", "RGT", "RCA"),
    ("RCA", "RLE", "RFM"),
    ("REP", "RA", "RUL"),
    ("REP", "C7", "RUL"),
    ("RLE", "RASIS", "RFM"),
    ("RA", "C7", "REP"),
    ("LPSIS", "LGT", "RGT"),
    ("LASIS", "LGT", "RGT"),
    ("LPSIS", "LLE", "RLE"),
    ("C7", "LA", "RA"),
    ("C7", "LEP", "REP"),
    ("RPSIS", "LGT", "RGT"),
    ("RASIS", "LGT", "RGT"),
    ("RPSIS", "LLE", "RLE"),
    ("C7", "LUL", "RUL"),
    ("LASIS", "C7", "LPSIS"),
    ("RASIS", "C7", "RPSIS"),
    ("LA", "LASIS", "RASIS"),
    ("RA", "LASIS", "RASIS"),
]

# Version en indices para vectorizar.
ANGLE_TRIPLETS_IDX = np.array(
    [[MARKER_IDX[a], MARKER_IDX[b], MARKER_IDX[c]] for a, b, c in ANGLE_TRIPLETS]
)


def _angle_at_vertex(p_i: np.ndarray, p_v: np.ndarray, p_iii: np.ndarray) -> np.ndarray:
    """Angulo (radianes) en el vertice p_v entre los vectores v->i y v->iii.

    Acepta arrays con shape (..., 3); devuelve shape (...).
    Propaga NaN si algun marcador es invalido (viene como NaN).
    """
    u = p_i - p_v
    w = p_iii - p_v
    cos = np.sum(u * w, axis=-1) / (
        np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1)
    )
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def compute_angles(xyz: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Calcula los 27 angulos 3D por frame.

    Args:
        xyz: (num_frames, 19, 3). Usar masked_coords() para que los invalidos sean NaN.
        degrees: si True devuelve grados, si no radianes.

    Returns:
        (num_frames, 27) con NaN donde algun marcador del triplete es invalido.
    """
    i, v, k = ANGLE_TRIPLETS_IDX[:, 0], ANGLE_TRIPLETS_IDX[:, 1], ANGLE_TRIPLETS_IDX[:, 2]
    ang = _angle_at_vertex(xyz[:, i, :], xyz[:, v, :], xyz[:, k, :])
    return np.degrees(ang) if degrees else ang
