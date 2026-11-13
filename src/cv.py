"""K-Fold разбиение train_val. Фолды создаются один раз - работа всегда с одинаковыми разбиениями."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

ART_DIR = Path("artifacts/folds")

def make_folds(y, n_folds=5, random_state=42, out_dir=ART_DIR):
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_id = np.zeros(len(y), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[val_idx] = fold
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"folds_{n_folds}.npz", fold_id=fold_id)
    return fold_id

def load_folds(n_folds=5, out_dir=ART_DIR):
    return np.load(Path(out_dir) / f"folds_{n_folds}.npz")["fold_id"]

def iter_folds(X, y, fold_id):
    """Генератор: (номер_фолда, X_tr, X_val, y_tr, y_val)."""
    for fold in range(int(fold_id.max()) + 1):
        val = fold_id == fold
        yield fold, X[~val], X[val], y[~val], y[val]