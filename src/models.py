"""
models.py — Etapa C: Modelado MLP para clasificación de diplejía.

Basado en Ferrari et al. (2019), Secciones 3.2 (Task 5), 4 y 6.

Funcionalidades:
  - patient_wise_split(): Split 75/25 paciente a paciente por clase.
  - augment_form1(): Duplicación de datos de la Forma 1 en el train set.
  - DiplegiaMLP: Arquitectura PyTorch de 5 capas ocultas (Fig. 1).
  - evaluate_patient_level(): Agregación por paciente para Top-1 y Top-2 accuracy (Tabla 7).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    nn = object   # fallback para herencia de clase si no hay PyTorch




# ──────────────────────────────────────────────────────────────────────────────
# 1. Patient-wise Split & Data Augmentation (§3.2 Task 5)
# ──────────────────────────────────────────────────────────────────────────────

def patient_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Realiza un split patient-wise (75% train / 25% test) estratificado por clase.

    Asegura que los trials del mismo paciente NO aparezcan en train y test a la vez.

    Returns:
        train_idx, test_idx (arrays de índices enteros)
    """
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices  = []

    for label in np.unique(y):
        class_mask = (y == label)
        class_subjs = np.unique(subjects[class_mask])
        rng.shuffle(class_subjs)

        n_train_subjs = max(1, int(np.round(len(class_subjs) * train_ratio)))
        train_subjs = set(class_subjs[:n_train_subjs])
        test_subjs  = set(class_subjs[n_train_subjs:])

        for idx in np.where(class_mask)[0]:
            if subjects[idx] in train_subjs:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

    return np.array(train_indices), np.array(test_indices)


def augment_form1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Duplica los trials de la Forma 1 (label 0) en el conjunto de entrenamiento (Tabla 3).

    Returns:
        X_aug, y_aug, subj_aug
    """
    form1_mask = (y_train == 0)
    X_f1 = X_train[form1_mask]
    y_f1 = y_train[form1_mask]
    s_f1 = subj_train[form1_mask]

    X_aug = np.concatenate([X_train, X_f1], axis=0)
    y_aug = np.concatenate([y_train, y_f1], axis=0)
    s_aug = np.concatenate([subj_train, s_f1], axis=0)

    return X_aug, y_aug, s_aug


# ──────────────────────────────────────────────────────────────────────────────
# 2. Arquitecturas de Modelos (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

_BaseModel = nn.Module if HAS_TORCH else object


class DiplegiaMLP(_BaseModel):
    """
    Red MLP según Fig. 1 del paper (Ferrari et al. 2019).

    Entrada: (B, 1620)
    Capas: Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(64) -> Dropout(0.2)
           -> Dense(32) -> ReLU -> Dense(4) -> Softmax
    """
    def __init__(self, input_dim: int = 1620, num_classes: int = 4, dropout_rate: float = 0.2):
        if not HAS_TORCH:
            raise ImportError("PyTorch no está instalado en este entorno. Instala 'torch' para usar DiplegiaMLP.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiplegiaLSTM(_BaseModel):
    """
    Red Recurrente LSTM según Fig. 2 del paper (Ferrari et al. 2019).

    Entrada: (B, 75, 81)
    Capas: LSTM(hidden=32, batch_first=True) -> FC(1024) -> ReLU -> FC(496) -> ReLU
           -> FC(64) -> ReLU -> FC(32) -> ReLU -> FC(4) -> Softmax
    """
    def __init__(self, input_dim: int = 81, hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 496),
            nn.ReLU(),
            nn.Linear(496, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        # Tomar la última salida temporal (many-to-one) o hn[-1]
        last_out = out[:, -1, :]
        return self.fc(last_out)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Métricas de Evaluación Paciente a Paciente (Top-1 y Top-2)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_patient_level(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
    device: torch.device = None,
) -> dict:
    """
    Evalúa la precisión del modelo nivel Paciente (Top-1 y Top-2), como en Tabla 7.

    Para cada paciente en test:
      - Promedia las probabilidades predichas de todos sus trials/secuencias.
      - Top-1: La clase con mayor probabilidad coincide con la real.
      - Top-2: La clase real está entre las 2 con mayor probabilidad.

    Returns:
        Diccionario con métricas generales y por forma, más la matriz de confusión.
    """
    device = device or torch.device('cpu')
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()

    return _aggregate_patient_probs(probs, y_test, subj_test)


def evaluate_probs_patient_level(
    probs: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
) -> dict:
    """
    Evalúa matrices de probabilidad (N, 4) obtenidas por cualquier clasificador (Sklearn, PyTorch, etc).
    """
    return _aggregate_patient_probs(probs, y_test, subj_test)


def evaluate_epoch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subj: np.ndarray,
    criterion: nn.Module,
    device: torch.device = None,
) -> dict:
    """
    Evalúa el modelo en modo eval sobre (X, y, subj), devolviendo loss y
    accuracy a nivel secuencia junto con la agregación a nivel paciente
    (reutiliza _aggregate_patient_probs). Pensada para llamarse una vez por
    época dentro del loop de entrenamiento, sin repetir el forward pass ni
    duplicar la lógica de agregación por paciente.

    Returns:
        dict con 'loss' (float), 'seq_acc' (float) y 'patient' (dict, el
        mismo formato que devuelve evaluate_patient_level).
    """
    device = device or torch.device('cpu')
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        targets = torch.tensor(y, dtype=torch.long).to(device)
        logits = model(inputs)
        loss = criterion(logits, targets).item()
        seq_acc = (logits.argmax(dim=-1) == targets).float().mean().item()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    patient = _aggregate_patient_probs(probs, y, subj)
    return {'loss': loss, 'seq_acc': seq_acc, 'patient': patient}


def _aggregate_patient_probs(
    probs: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
) -> dict:
    patient_probs = {}
    patient_y     = {}

    for i, subj in enumerate(subj_test):
        if subj not in patient_probs:
            patient_probs[subj] = []
            patient_y[subj]     = y_test[i]
        patient_probs[subj].append(probs[i])

    unique_subjs = list(patient_probs.keys())
    t1_hits = []
    t2_hits = []
    y_true_list = []
    y_pred_list = []

    per_form_t1 = {0: [], 1: [], 2: [], 3: []}
    per_form_t2 = {0: [], 1: [], 2: [], 3: []}

    for subj in unique_subjs:
        avg_prob = np.mean(patient_probs[subj], axis=0)
        true_lbl = patient_y[subj]
        top2_preds = np.argsort(avg_prob)[::-1][:2]

        top1_correct = (top2_preds[0] == true_lbl)
        top2_correct = (true_lbl in top2_preds)

        t1_hits.append(top1_correct)
        t2_hits.append(top2_correct)

        per_form_t1[true_lbl].append(top1_correct)
        per_form_t2[true_lbl].append(top2_correct)

        y_true_list.append(true_lbl)
        y_pred_list.append(top2_preds[0])

    overall_t1 = np.mean(t1_hits)
    overall_t2 = np.mean(t2_hits)

    form_t1_acc = {f: np.mean(per_form_t1[f]) if per_form_t1[f] else 0.0 for f in range(4)}
    form_t2_acc = {f: np.mean(per_form_t2[f]) if per_form_t2[f] else 0.0 for f in range(4)}

    cm = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1, 2, 3])

    return {
        'overall_t1': overall_t1,
        'overall_t2': overall_t2,
        'form_t1':    form_t1_acc,
        'form_t2':    form_t2_acc,
        'cm':         cm,
        'y_true':     np.array(y_true_list),
        'y_pred':     np.array(y_pred_list),
        'num_patients': len(unique_subjs),
    }

