# Diplegia Gait Classification

Replicación de *Ferrari et al. (2019), "Gait-Based Diplegia Classification Using LSTM Networks"* (J. Healthcare Engineering) a partir del dataset crudo de marcha (marcadores VICON, formato `.npy`).

## Estado del dato

Los `.npy` **no** están en el repo original ni en este: se descargan del Google Drive enlazado en el [repo del dataset](https://github.com/lucabergamini/gait-analysis-dataset). Cada `.npy` es **un trial** con forma `(n_frames, 19, 4)`:

- 19 marcadores en orden fijo (ver `src/data.py::MARKERS`).
- Último eje `[x, y, z, flag]`; `flag == -1` → marcador inválido. Coordenadas en **mm**.
- La clase (0–3) y el sujeto salen del **path**: `data/<clase>/<sujeto>/<trial>.npy`.

> Dato crudo únicamente. Los 27 ángulos, la segmentación por pasos, la FFT y el split del paper **hay que reconstruirlos**. Conteos: paper = 174 pac / ~1038 trials; README del dataset = 178 pac / 1139 trials (incluye inválidos).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Descargar el dataset del Drive y descomprimir en data/diplegia/
```

## Estructura

```
diplegia-gait/
├── src/                 # Código compartido, importado por los notebooks
│   ├── data.py          # Carga .npy, parseo de path, máscaras de validez
│   ├── angles.py        # Los 27 ángulos de la Tabla 5 del paper
│   ├── features.py      # Segmentación de pasos + FFT (etapa B)
│   └── models.py        # MLP y LSTM (etapa C)
├── notebooks/           # Un notebook por etapa. Importan de src/, no duplican lógica
│   ├── 01_eda.ipynb        # Exploración y limpieza
│   ├── 02_features.ipynb   # Ángulos + FFT
│   ├── 03_mlp.ipynb        # Entrenamiento MLP
│   └── 04_lstm.ipynb       # Entrenamiento LSTM
└── data/                # Ignorado por git
```

## Reparto de tareas sugerido

| Etapa | Entregable | Depende de |
|---|---|---|
| **A. EDA + limpieza** | `01_eda.ipynb`, criterio de trials válidos en `data.py` | — |
| **B. Features** | `features.py` (pasos + FFT, 20 coefs) + `02_features.ipynb` | A |
| **C. Modelos** | `models.py` (MLP, LSTM) + `03/04_*.ipynb`, split patient-wise | B |
| **D. Evaluación** | matrices de confusión, top-1/top-2 vs Tabla 7–8 | C |

A y B son el camino crítico y son separables limpiamente entre dos personas (uno afina limpieza + segmentación de pasos, otro monta el pipeline de FFT/features y el esqueleto de modelos).