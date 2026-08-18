"""
features.py — Etapa B del pipeline de clasificación de diplejía.

Implementa las Tasks 1-4 del preprocesamiento de Ferrari et al. (2019),
"Gait-Based Diplegia Classification Using LSTM Networks".

Pipeline:
  Task 1+2  → segment_trial():           recorta bordes, estima T
  Task 3    → compute_projected_angles(): 81 ángulos por frame (27 × 3 planos)
  Task 4a   → compute_mlp_features():    vector 1620-D para MLP
  Task 4b   → compute_lstm_sequences():  ventanas (75, 81) para LSTM

Uso típico:
    from src.data import load_dataset, masked_coords, interpolate_short_gaps
    from src.features import process_trial, build_dataset

    trials = load_dataset('data/diplegia')
    result = build_dataset(trials)
    # result['mlp']  → dict con arrays X, y, subject
    # result['lstm'] → dict con arrays X, y, subject
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

try:
    from .data import MARKER_IDX, masked_coords
    from .angles import ANGLE_TRIPLETS_IDX
except ImportError:
    from data import MARKER_IDX, masked_coords
    from angles import ANGLE_TRIPLETS_IDX


# ──────────────────────────────────────────────────────────────────────────────
# Constantes del pipeline (ver §3.2 del paper)
# ──────────────────────────────────────────────────────────────────────────────

FS_ORIGINAL: int   = 100   # fps de captura VICON original
FS: int            = 50    # fps tras subsampling ×2 (paper §3.2)
SUBSAMPLE_FACTOR   = FS_ORIGINAL // FS   # = 2

# Detección de foot-strikes
VERT_AXIS: int         = 2      # índice del eje vertical (Z = altura sobre el suelo en VICON)
MIN_STEP_GAP_SEC: float = 0.30  # separación mínima entre foot-strikes del mismo pie (s)
MIN_PROMINENCE: float   = 2.0   # prominencia mínima del mínimo (mm)

# Parámetros FFT – Task 4a
N_COEFF: int = 20    # primeros 20 coeficientes armónicos del paso (paper §3.2)

# Parámetros secuencias – Task 4b
SEQ_LEN:  int = 75   # longitud de cada ventana temporal (frames)
SEQ_STEP: int = 15   # desplazamiento entre ventanas (frames)
MAX_SEQ:  int = 45   # máximo de secuencias por trial

# Planos anatómicos: cada tupla indica (nombre, eje_a_eliminar)
# Al proyectar, se elimina una coordenada y se trabaja en 2D
PLANES: list[tuple[str, int]] = [
    ("sagital",      0),   # proyección YZ: eliminar X (dimensión lateral)
    ("frontal",      2),   # proyección XY: eliminar Z (dimensión vertical)
    ("transversal",  1),   # proyección XZ: eliminar Y (dimensión A-P)
]


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 + 2: Segmentación de pasos
# ──────────────────────────────────────────────────────────────────────────────

def _find_foot_strikes(pos_1d: np.ndarray, min_gap: int) -> np.ndarray:
    """
    Detecta foot-strikes como mínimos locales en la posición vertical de un talón.

    Un foot-strike (contacto del talón con el suelo) se manifiesta como un mínimo
    en la coordenada vertical del marcador de talón: el marcador desciende hasta
    el suelo y luego sube durante el swing.

    Args:
        pos_1d:   Posición vertical del marcador (num_frames,), en mm.
        min_gap:  Separación mínima entre mínimos consecutivos, en frames.

    Returns:
        Índices de frames correspondientes a foot-strikes detectados.
    """
    peaks, _ = find_peaks(
        -pos_1d,             # invertimos para buscar mínimos con find_peaks
        distance=min_gap,
        prominence=MIN_PROMINENCE,
    )
    return peaks


def detect_steps(
    xyz: np.ndarray,
    fs: float = FS,
    vert_axis: int = VERT_AXIS,
    min_step_gap_sec: float = MIN_STEP_GAP_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta foot-strikes de ambos pies usando los marcadores de talón.

    Usa RCA (talón derecho) y LCA (talón izquierdo), detectando mínimos locales
    en la coordenada vertical indicada por ``vert_axis``.

    Args:
        xyz:              Coordenadas (num_frames, 19, 3) en mm.
                          Los NaN en marcadores inválidos son ignorados por find_peaks.
        fs:               Frecuencia de muestreo (fps).
        vert_axis:        Índice del eje vertical en la última dimensión de xyz.
        min_step_gap_sec: Separación mínima entre foot-strikes del mismo pie (s).

    Returns:
        Tupla ``(r_strikes, l_strikes)`` con los índices de frame de cada pie.
    """
    min_gap = max(1, int(min_step_gap_sec * fs))

    # Posición vertical de cada talón; reemplazar NaN con interpolación lineal
    # para que find_peaks no se vea afectado por datos inválidos
    rca_v = xyz[:, MARKER_IDX['RCA'], vert_axis].copy()
    lca_v = xyz[:, MARKER_IDX['LCA'], vert_axis].copy()

    for v in (rca_v, lca_v):
        nans = np.isnan(v)
        if nans.any() and not nans.all():
            idx = np.arange(len(v))
            v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
        elif nans.all():
            v[:] = 0.0   # marcador completamente inválido → sin strikes

    r_strikes = _find_foot_strikes(rca_v, min_gap)
    l_strikes = _find_foot_strikes(lca_v, min_gap)

    return r_strikes, l_strikes


def segment_trial(
    xyz: np.ndarray,
    fs: float = FS,
    vert_axis: int = VERT_AXIS,
    min_steps: int = 2,
) -> dict | None:
    """
    Recorta el trial para conservar solo los frames con pasos completos.

    Task 1 del paper: estima el período T promedio de un paso.
    Task 2 del paper: selecciona la región con número entero de pasos,
                      eliminando los bordes (pasos parciales).

    Los "bordes" son los frames antes del primer foot-strike y después del último:
    en esos instantes el paciente está entrando o saliendo del volumen de captura.

    Args:
        xyz:       Coordenadas ya interpoladas (num_frames, 19, 3) en mm.
        fs:        Frecuencia de muestreo en fps.
        vert_axis: Índice del eje vertical.
        min_steps: Número mínimo de pasos para considerar el trial válido.

    Returns:
        Diccionario con:
          - ``xyz_cropped``  → (n_valid_frames, 19, 3)
          - ``n_steps``      → número de pasos detectados (int)
          - ``T_frames``     → período medio de un paso en frames (float)
          - ``r_strikes``    → foot-strikes pie derecho (índices globales, pre-recorte)
          - ``l_strikes``    → foot-strikes pie izquierdo (índices globales, pre-recorte)
          - ``start_frame``  → primer frame válido (int)
          - ``end_frame``    → último frame válido (int)
        O ``None`` si el trial no tiene suficientes pasos.
    """
    r_strikes, l_strikes = detect_steps(xyz, fs=fs, vert_axis=vert_axis)

    all_strikes = np.sort(np.concatenate([r_strikes, l_strikes]))

    # Necesitamos al menos min_steps intervalos entre strikes consecutivos
    if len(all_strikes) < min_steps + 1:
        return None

    start_frame = int(all_strikes[0])
    end_frame   = int(all_strikes[-1])

    if end_frame - start_frame < 20:   # al menos 20 frames válidos
        return None

    # Número de pasos = número de intervalos entre strikes (ambos pies juntos)
    n_steps = len(all_strikes) - 1

    if n_steps < min_steps:
        return None

    # T = duración total de la región válida / número de pasos
    T_frames = (end_frame - start_frame) / n_steps

    return {
        'xyz_cropped': xyz[start_frame : end_frame + 1],
        'n_steps':     n_steps,
        'T_frames':    T_frames,
        'r_strikes':   r_strikes,
        'l_strikes':   l_strikes,
        'start_frame': start_frame,
        'end_frame':   end_frame,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task 3: 81 ángulos proyectados (27 ángulos × 3 planos anatómicos)
# ──────────────────────────────────────────────────────────────────────────────

def _project_plane(xyz: np.ndarray, drop_axis: int) -> np.ndarray:
    """
    Proyecta las coordenadas 3D en un plano anatómico eliminando un eje.

    Args:
        xyz:       (num_frames, 19, 3) coordenadas completas.
        drop_axis: Índice del eje a eliminar (0, 1 ó 2).

    Returns:
        (num_frames, 19, 2) coordenadas proyectadas en el plano.
    """
    axes = [i for i in range(3) if i != drop_axis]
    return xyz[:, :, axes]


def _angle_2d_batch(
    p_i: np.ndarray,
    p_v: np.ndarray,
    p_iii: np.ndarray,
) -> np.ndarray:
    """
    Calcula el ángulo en el vértice p_v en 2D, de forma vectorizada.

    Acepta arrays con shape (..., 2). Propaga NaN si algún punto tiene NaN.

    Args:
        p_i:   Posición del marcador I   → (..., 2)
        p_v:   Posición del vértice      → (..., 2)
        p_iii: Posición del marcador III → (..., 2)

    Returns:
        Ángulo en grados → shape (...,)
    """
    u = p_i   - p_v    # vector vértice → I
    w = p_iii - p_v    # vector vértice → III

    norm_u = np.linalg.norm(u, axis=-1)
    norm_w = np.linalg.norm(w, axis=-1)

    # Evitar división por cero (ocurre cuando dos marcadores coinciden)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.sum(u * w, axis=-1) / (norm_u * norm_w)

    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_projected_angles(xyz: np.ndarray) -> np.ndarray:
    """
    Calcula 81 ángulos por frame proyectando las coordenadas 3D en los
    tres planos anatómicos y calculando los 27 ángulos de la Tabla 5 en cada uno.

    Task 3 del paper:
    "projecting 3D coordinates of markers to the three human body's planes.
    Then, the projected coordinates were processed to extract 27 scalar angles
    in each plane (Table 5) [...] Consequently, 81 angles per frame were generated."

    Los tres planos y sus proyecciones (eje eliminado):
      - Sagital     (YZ): elimina X (eje 0) — movimiento adelante/atrás
      - Frontal     (XY): elimina Z (eje 2) — movimiento vertical
      - Transversal (XZ): elimina Y (eje 1) — movimiento A-P

    Args:
        xyz: (num_frames, 19, 3) coordenadas en mm. NaN donde el marcador es inválido.

    Returns:
        (num_frames, 81) ángulos en grados.
        El orden de columnas es: [27_sagital | 27_frontal | 27_transversal].
        Se propaga NaN si algún marcador del triplete es inválido.
    """
    n_frames  = xyz.shape[0]
    n_angles  = ANGLE_TRIPLETS_IDX.shape[0]   # 27
    n_planes  = len(PLANES)                   # 3
    result    = np.full((n_frames, n_planes * n_angles), np.nan)

    i_idx = ANGLE_TRIPLETS_IDX[:, 0]   # índices del marcador I   (27,)
    v_idx = ANGLE_TRIPLETS_IDX[:, 1]   # índices del vértice      (27,)
    k_idx = ANGLE_TRIPLETS_IDX[:, 2]   # índices del marcador III (27,)

    for plane_num, (plane_name, drop_axis) in enumerate(PLANES):
        # Proyectar coordenadas al plano (eliminar un eje)
        xy = _project_plane(xyz, drop_axis)  # (num_frames, 19, 2)

        # Extraer posiciones de los tres marcadores de cada triplete
        # Forma resultante: (num_frames, 27, 2)
        p_i   = xy[:, i_idx, :]
        p_v   = xy[:, v_idx, :]
        p_iii = xy[:, k_idx, :]

        angles = _angle_2d_batch(p_i, p_v, p_iii)   # (num_frames, 27)

        col_start = plane_num * n_angles
        col_end   = col_start + n_angles
        result[:, col_start:col_end] = angles

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Relleno de NaN residual (bordes con validez dispareja entre marcadores)
# ──────────────────────────────────────────────────────────────────────────────

def _fill_nan_columns(angles_81: np.ndarray) -> np.ndarray:
    """
    Reemplaza NaN por el promedio de cada columna (ángulo).

    `segment_trial` recorta el trial usando los foot-strikes de los talones
    (RCA/LCA), pero no garantiza que el resto de los marcadores sean válidos
    en todo ese rango: un marcador con un borde inválido más largo que el de
    los talones deja NaN dentro de la región recortada. Tanto la FFT (Task 4a)
    como las secuencias del LSTM (Task 4b) necesitan un array sin NaN.
    """
    angles_filled = angles_81.copy()
    for col in range(angles_filled.shape[1]):
        col_data = angles_filled[:, col]
        nan_mask = np.isnan(col_data)
        if nan_mask.any() and not nan_mask.all():
            col_mean = np.nanmean(col_data)
            angles_filled[nan_mask, col] = col_mean
        elif nan_mask.all():
            angles_filled[:, col] = 0.0
    return angles_filled


# ──────────────────────────────────────────────────────────────────────────────
# Task 4a: FFT → vector de features para MLP (1620-D)
# ──────────────────────────────────────────────────────────────────────────────

def compute_mlp_features(
    angles_81: np.ndarray,
    n_steps: int,
    n_coeff: int = N_COEFF,
) -> np.ndarray | None:
    """
    Aplica FFT y extrae los primeros ``n_coeff`` coeficientes asociados a los
    armónicos del período de un paso, generando el vector de features para MLP.

    Task 4a del paper:
    "the fast Fourier transform (FFT) algorithm was applied to a sequence
    encompassing multiple (say N) steps. Then, one coefficient every N was
    extracted from the FFT output vector, in order to analyze only those
    harmonics associated with the fundamental of a single step. Only the
    first 20 coefficients selected in this way were preserved; the first
    selected coefficient was not normalized, whereas all the other ones
    were normalized to the amplitude of the fundamental."

    Los coeficientes que corresponden a los armónicos del paso son los de los
    índices N, 2N, 3N, ..., 20N del vector FFT (donde N = n_steps).
    La amplitud del fundamental es |FFT[N]|; los coeficientes 2..20 se dividen
    por ese valor. El resultado final tiene forma (n_coeff × 81,) = (1620,).

    Args:
        angles_81: (num_frames, 81) ángulos del trial ya recortado.
        n_steps:   Número de pasos enteros en el trial (N del paper).
        n_coeff:   Número de coeficientes armónicos a extraer (paper: 20).

    Returns:
        Vector 1-D de (n_coeff × 81,) = (1620,) features.
    """
    angles_filled = _fill_nan_columns(angles_81)

    # Aplicar FFT a lo largo del eje temporal para cada uno de los 81 ángulos
    # fft_out shape: (num_freq_bins, 81), valores complejos
    fft_out = np.fft.rfft(angles_filled, axis=0)
    fft_len = fft_out.shape[0]

    # Extraer coeficientes en posiciones N, 2N, ..., n_coeff*N
    harmonic_indices = np.arange(1, n_coeff + 1) * n_steps   # [N, 2N, ..., 20N]

    # Extraer magnitudes de FFT; si algún índice supera los bins disponibles, se rellena con 0
    coeff_matrix = np.zeros((n_coeff, 81), dtype=float)
    for k, idx in enumerate(harmonic_indices):
        if idx < fft_len:
            coeff_matrix[k, :] = np.abs(fft_out[idx, :])
        else:
            coeff_matrix[k, :] = 0.0

    # Normalización: el primer coeficiente (k=1, fundamental) queda sin normalizar;
    # los coeficientes 2..20 se dividen por la amplitud del fundamental.
    fundamental = coeff_matrix[0, :]                     # (81,)
    norm_factor  = np.where(fundamental == 0, 1.0, fundamental)

    # coeff_matrix[0] queda igual; coeff_matrix[1:] se normaliza
    coeff_matrix[1:, :] = coeff_matrix[1:, :] / norm_factor[np.newaxis, :]

    # Aplanar en un vector 1-D: (n_coeff × 81,) = (1620,)
    return coeff_matrix.flatten()



# ──────────────────────────────────────────────────────────────────────────────
# Task 4b: Secuencias temporales para LSTM
# ──────────────────────────────────────────────────────────────────────────────

def compute_lstm_sequences(
    angles_81: np.ndarray,
    window:   int = SEQ_LEN,
    stride:   int = SEQ_STEP,
    max_seq:  int = MAX_SEQ,
) -> np.ndarray | None:
    """
    Divide la secuencia de ángulos en ventanas solapadas para el LSTM.

    Task 4b del paper:
    "simply gathers sequences of 75 elements (where each element is composed
    of 81 3D angles), displaced by 15 elements and for a maximum of 45
    sequences per trial."

    Args:
        angles_81: (num_frames, 81) ángulos del trial ya recortado.
        window:    Longitud de cada ventana en frames (paper: 75).
        stride:    Desplazamiento entre ventanas en frames (paper: 15).
        max_seq:   Máximo de secuencias a generar por trial (paper: 45).

    Returns:
        Array de shape (n_seq, window, 81) con n_seq ≤ max_seq, o
        ``None`` si el trial tiene menos frames que una ventana.
    """
    n_frames = angles_81.shape[0]

    if n_frames < window:
        return None

    angles_filled = _fill_nan_columns(angles_81)

    sequences = []
    start = 0
    while start + window <= n_frames and len(sequences) < max_seq:
        seq = angles_filled[start : start + window, :]    # (75, 81)
        sequences.append(seq)
        start += stride

    if not sequences:
        return None

    return np.stack(sequences, axis=0)   # (n_seq, 75, 81)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline completo: procesar un solo trial
# ──────────────────────────────────────────────────────────────────────────────

def process_trial(
    trial,                      # instancia de src.data.Trial
    fs: float = FS,
    vert_axis: int = VERT_AXIS,
    subsample: bool = True,
) -> dict | None:
    """
    Ejecuta el pipeline completo de la Etapa B para un trial.

    Pasos:
      1. Subsampling ×2 (100 fps → 50 fps), si ``subsample=True``.
      2. Obtener coordenadas con NaN en marcadores inválidos.
      3. Segmentación de pasos (Tasks 1+2): recortar bordes.
      4. Calcular 81 ángulos proyectados (Task 3).
      5. Features FFT para MLP (Task 4a).
      6. Secuencias para LSTM (Task 4b).

    Args:
        trial:     Objeto ``Trial`` de ``src.data`` (ya interpolado en EDA).
        fs:        Frecuencia de muestreo después del subsampling.
        vert_axis: Eje vertical para detección de pasos.
        subsample: Si True, aplica ``xyz = xyz[::SUBSAMPLE_FACTOR]`` primero.

    Returns:
        Diccionario con:
          - ``'label'``           → int, clase 0-3
          - ``'subject'``         → str, ID del paciente
          - ``'trial_id'``        → str
          - ``'n_steps'``         → int
          - ``'T_frames'``        → float
          - ``'mlp_features'``    → np.ndarray (1620,)
          - ``'lstm_sequences'``  → np.ndarray (n_seq, 75, 81) o None
          - ``'n_lstm_seq'``      → int
        O ``None`` si el trial se descarta (insuficientes pasos o frames).
    """
    # 1. Subsampling ×2: de 100 fps a 50 fps (paper §3.2)
    X = trial.X if not subsample else trial.X[::SUBSAMPLE_FACTOR]

    # 2. Coordenadas con NaN en marcadores inválidos
    xyz = masked_coords(X)   # (num_frames, 19, 3) con NaN donde flag == -1

    # 3. Segmentación de pasos: recortar bordes (Tasks 1+2)
    seg = segment_trial(xyz, fs=fs, vert_axis=vert_axis)
    if seg is None:
        return None   # trial descartado por insuficientes pasos

    xyz_crop = seg['xyz_cropped']   # región con pasos completos
    n_steps  = seg['n_steps']
    T_frames = seg['T_frames']

    # 4. Calcular 81 ángulos proyectados (Task 3)
    angles_81 = compute_projected_angles(xyz_crop)   # (n_valid_frames, 81)

    # 5. Features FFT para MLP (Task 4a)
    mlp_feat = compute_mlp_features(angles_81, n_steps=n_steps)
    if mlp_feat is None:
        return None   # trial demasiado corto para FFT

    # 6. Secuencias para LSTM (Task 4b)
    lstm_seqs = compute_lstm_sequences(angles_81)

    return {
        'label':          trial.label,
        'subject':        trial.subject,
        'trial_id':       trial.trial_id,
        'n_steps':        n_steps,
        'T_frames':       T_frames,
        'mlp_features':   mlp_feat,
        'lstm_sequences': lstm_seqs,
        'n_lstm_seq':     len(lstm_seqs) if lstm_seqs is not None else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline completo: procesar todos los trials y construir los datasets
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    trials: list,
    fs: float = FS,
    vert_axis: int = VERT_AXIS,
    subsample: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Procesa todos los trials y devuelve los arrays listos para MLP y LSTM.

    Args:
        trials:    Lista de objetos ``Trial`` (ya interpolados por el EDA).
        fs:        Frecuencia de muestreo post-subsampling.
        vert_axis: Eje vertical para detección de pasos.
        subsample: Aplicar subsampling ×2 antes del procesamiento.
        verbose:   Imprimir progreso y resumen.

    Returns:
        Diccionario con dos entradas:
          ``'mlp'``:
            - ``X``       → (N_trials, 1620)      features FFT
            - ``y``       → (N_trials,)            etiquetas 0-3
            - ``subject`` → (N_trials,)            IDs de pacientes

          ``'lstm'``:
            - ``X``       → (N_seq_total, 75, 81) secuencias temporales
            - ``y``       → (N_seq_total,)         etiquetas 0-3
            - ``subject`` → (N_seq_total,)         IDs de pacientes
    """
    mlp_X, mlp_y, mlp_subj   = [], [], []
    lstm_X, lstm_y, lstm_subj = [], [], []

    n_total    = len(trials)
    n_ok       = 0
    n_discard  = 0

    for i, trial in enumerate(trials):
        if verbose and (i % 100 == 0):
            print(f"  Procesando trial {i+1}/{n_total}...", end='\r')

        result = process_trial(trial, fs=fs, vert_axis=vert_axis,
                               subsample=subsample)

        if result is None:
            n_discard += 1
            continue

        n_ok += 1

        # --- MLP ---
        mlp_X.append(result['mlp_features'])
        mlp_y.append(result['label'])
        mlp_subj.append(result['subject'])

        # --- LSTM ---
        if result['lstm_sequences'] is not None:
            seqs = result['lstm_sequences']          # (n_seq, 75, 81)
            lstm_X.append(seqs)
            lstm_y.extend([result['label']]  * len(seqs))
            lstm_subj.extend([result['subject']] * len(seqs))

    if verbose:
        print(f"\n✓ Trials procesados: {n_ok}/{n_total} válidos "
              f"({n_discard} descartados)")
        if mlp_X:
            labels, counts = np.unique(mlp_y, return_counts=True)
            print("  Distribución MLP por clase:")
            for lbl, cnt in zip(labels, counts):
                print(f"    Form {lbl+1}: {cnt} trials")

    dataset: dict = {}

    if mlp_X:
        dataset['mlp'] = {
            'X':       np.stack(mlp_X, axis=0),        # (N, 1620)
            'y':       np.array(mlp_y, dtype=int),     # (N,)
            'subject': np.array(mlp_subj),             # (N,)
        }
    else:
        dataset['mlp'] = None

    if lstm_X:
        dataset['lstm'] = {
            'X':       np.concatenate(lstm_X, axis=0), # (M, 75, 81)
            'y':       np.array(lstm_y, dtype=int),    # (M,)
            'subject': np.array(lstm_subj),            # (M,)
        }
    else:
        dataset['lstm'] = None

    return dataset
