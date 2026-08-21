# Diplegia Gait Classification

Replicación de *Ferrari et al. (2019), "Gait-Based Diplegia Classification Using LSTM Networks"* (J. Healthcare Engineering) a partir del dataset crudo de marcha (marcadores VICON, formato `.npy`).

## El dataset crudo

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
│   ├── data.py          # Carga .npy, máscaras de validez, análisis de huecos e interpolación
│   ├── angles.py        # Los 27 ángulos de la Tabla 5 del paper
│   ├── features.py      # Segmentación de pasos, ángulos proyectados, features MLP (FFT) y LSTM (secuencias)
│   └── models.py        # MLP y LSTM en PyTorch, split patient-wise, evaluación por paciente
├── notebooks/           # Un notebook por etapa. Importan de src/, no duplican lógica
│   ├── 01_eda.ipynb        # Exploración y limpieza
│   ├── 02_features.ipynb   # Ángulos + FFT + secuencias, guarda data/features/*.npz
│   ├── 03_mlp_demo.ipynb   # Entrenamiento y evaluación del MLP
│   └── 04_lstm.ipynb       # Entrenamiento y evaluación del LSTM
└── data/                # Ignorado por git (dataset y data/features/*.npz se generan localmente)
```