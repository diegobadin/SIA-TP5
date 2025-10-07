from typing import Callable, Optional, Any, Dict

import os
import numpy as np

from src.mlp.mlp import MLP


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        return float(np.mean((y_pred.ravel() > 0.5).astype(int) == y_true.ravel().astype(int)))
    yhat = np.argmax(y_pred, axis=1)
    ytru = np.argmax(y_true, axis=1) if (y_true.ndim==2 and y_true.shape[1]>1) else y_true.ravel().astype(int)
    return float(np.mean(yhat == ytru))

def cross_validate(model_factory, X, Y, splitter, fit_kwargs=None, scoring=accuracy_score):
    """
       Realiza cross-validation con cualquier splitter (Holdout, KFold, StratifiedKFold)
       usando accuracy_scores para evaluar cada fold.

       Args:
           model_factory: función que devuelve una instancia nueva del modelo (MLP, etc.)
           X: datos de entrada
           Y: etiquetas
           splitter: objeto con método .split(X, Y) que genera (train_idx, test_idx)
           fit_kwargs: kwargs que se pasan al método fit()
           scoring: permite seleccionar la métrica de evaluación (accuracy, MSE, etc)

       Returns:
           dict con accuracy promedio, std y lista de accuracies por fold
    """
    fit_kwargs = fit_kwargs or {}
    accs = []
    for tr_idx, te_idx in splitter.split(X, Y):
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xte, Yte = X[te_idx], Y[te_idx]

        model = model_factory()
        model.fit(Xtr, Ytr, **fit_kwargs)

        preds = model.predict(Xte)

        # Si no se pasó nada, usa accuracy_scores por defecto
        score_fn = scoring
        acc = score_fn(Yte, preds)
        accs.append(acc)

    return {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "folds": accs
        }

def cross_validate_regression(model_factory, X, Y, splitter, scoring):
    results = []
    for train_idx, test_idx in splitter.split(X, Y):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        model = model_factory()
        model.fit(Xtr, Ytr)

        y_pred = np.array([model.predict(x) for x in Xte])  # predice fila por fila
        results.append(scoring(Yte, y_pred))
    return results


def sse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Sum of Squared Errors for regression.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return float(np.sum((y_true - y_pred) ** 2))


def save_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str,
    positive_label: int = 1,
    threshold: float = 0.5,
    comment: Optional[str] = None
) -> Dict[str, int]:
    """Compute TP/TN/FP/FN counts and store them in ``filepath``.

    Args:
        y_true: True labels or one-hot encoded targets.
        y_pred: Model predictions (probabilities or logits).
        filepath: Destination file for the counts.
        positive_label: Label treated as the positive class.
        threshold: Cutoff used when predictions are 1D probabilities.
        comment: Optional string to append at the end of the file.

    Returns:
        Dict with keys ``tp``, ``tn``, ``fp`` and ``fn``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true.ravel().astype(int)

    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_scores = y_pred.ravel()
        if np.issubdtype(y_pred_scores.dtype, np.integer):
            y_pred_labels = y_pred_scores.astype(int)
        else:
            y_pred_labels = (y_pred_scores > threshold).astype(int)

    y_true_pos = (y_true_labels == positive_label)
    y_pred_pos = (y_pred_labels == positive_label)

    tp = int(np.sum(y_true_pos & y_pred_pos))
    tn = int(np.sum(~y_true_pos & ~y_pred_pos))
    fp = int(np.sum(~y_true_pos & y_pred_pos))
    fn = int(np.sum(y_true_pos & ~y_pred_pos))

    counts = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    lines = [
        f"TP: {tp}",
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}"
    ]

    if comment:
        lines.append("")  # línea en blanco separadora
        lines.append(f"# {comment}")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return counts