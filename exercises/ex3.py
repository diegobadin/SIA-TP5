# exercises/ex3.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.mlp import MLP
from utils.activations import logistic_function_factory
from utils.parse_csv_data import parse_digits_7x5_txt



def run(item: str):
    item = item.lower()
    if item == "xor":
        return run_xor()
    if item == "parity":
        return run_parity()
    if item in ("digit", "digits"):
        return run_digits()
    raise ValueError(f"item desconocido: {item} (usa: xor | parity | digits)")


## item 1

def run_xor(
        alpha=0.1,
        epochs=5000,
        hidden=3,
        seed=0
):
    np.random.seed(seed)
    layer_sizes = [2, hidden, 1]
    activations = [
        logistic_function_factory(),  # capa oculta
        logistic_function_factory()   # capa salida
    ]
    mlp = MLP(layer_sizes, activations, alpha=alpha, max_iter=epochs)

    # Dataset XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    # Historial de error por época
    mlp.fit(X.tolist(), Y.tolist())

    # Resultados finales
    print("\n=== XOR Results ===")
    correct = 0
    print(f"{'Input':<10} {'Expected':<10} {'Pred':<10} {'Class':<6}")
    print("-" * 40)
    for x, y in zip(X, Y):
        pred = mlp.predict(x)
        pred_class = int(pred[0] > 0.5)
        print(f"{x} {y[0]:<10} {pred[0]:<10.3f} {pred_class:<6}")
        if pred_class == y[0]:
            correct += 1

    accuracy = correct / len(Y)
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

    # Graficar curva de error
    # plt.plot(mlp.error_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.title("Training Error on XOR (MLP)")
    # plt.show()


## item 2

def run_parity(
    data_path: str = "./data/TP3-ej3-digitos.txt",
    split_mode: str = "kfold",    # "holdout" | "stratified" | "kfold" | "stratified_kfold"
    val_ratio: float = 0.2,            # usado en holdout
    k: int = 5,                        # usado en kfold
    hidden: int = 16,
    alpha: float = 0.05,
    epochs: int = 300,
    optimizer: str = "adam",
    optimizer_params: dict | None = None,
    seed: int = 0,
    show_all_preds: bool = True        # imprime todas las preds del split de validación (holdout)
):
    # parseo data
    X, labels = parse_digits_7x5_txt(data_path)  # X: (N,35), labels: 0..9
    y_par = (labels % 2 == 0).astype(float).reshape(-1, 1)  # 1=par, 0=impar
    N, D = X.shape

    # separado según modo de partición con folds o holdout
    if split_mode in ("holdout", "stratified"):
        if split_mode == "holdout":
            tr_idx, va_idx = split_holdout(N, val_ratio=val_ratio, seed=seed)
        else:
            tr_idx, va_idx = split_stratified(y_par, val_ratio=val_ratio, seed=seed)

        Xtr, Ytr = X[tr_idx], y_par[tr_idx]
        Xva, Yva = X[va_idx], y_par[va_idx]

        mlp = _build_mlp(D, hidden, alpha, epochs, optimizer, optimizer_params)
        mlp.fit(Xtr.tolist(), Ytr.tolist())

        ytr_cls, _ = _predict_classes(mlp, Xtr)
        yva_cls, yva_prob = _predict_classes(mlp, Xva)

        acc_tr = _accuracy(Ytr, ytr_cls)
        acc_va = _accuracy(Yva, yva_cls)
        tp, fp, fn, tn = _confusion(Yva, yva_cls)

        print("\n=== Paridad (MLP) ===")
        print(f"Arquitectura: [{D}, {hidden}, 1] / opt: {optimizer}")
        print(f"alpha={alpha}  epochs={epochs}  hidden={hidden}")
        print(f"Accuracy Train: {acc_tr*100:.2f}%")
        print(f"Accuracy Val:   {acc_va*100:.2f}%")
        print("\nConfusión (Val) [clase=1: par]")
        print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        # tabla de validación
        df_val = pd.DataFrame({
            "idx": va_idx,
            "digit": labels[va_idx],
            "true_parity": Yva.flatten().astype(int),
            "predicted": yva_cls.flatten().astype(int),
            "prob": yva_prob.flatten()
        }).sort_values("idx").reset_index(drop=True)

        if show_all_preds:
            print("\nTodas las predicciones (Validación):")
            for i in range(len(df_val)):
                row = df_val.iloc[i]
                print(f"idx={int(row.idx):02d}  dígito={int(row.digit)}  "
                      f"y*={int(row.true_parity)}   ŷ={int(row.predicted)}   p={row.prob:.3f}")

        return df_val


    elif split_mode in ("kfold", "stratified_kfold"):

        folds = split_kfold(N, k=k, seed=seed) if split_mode == "kfold" else split_stratified_kfold(y_par, k=k,seed=seed)

        accs, rows = [], []

        all_val_rows = []  # acumulador de todas las predicciones de validación

        print(f"\n=== Paridad (MLP) K-Fold (modo={split_mode}, k={k}) ===")

        for i in range(k):
            va_idx = np.array(folds[i])
            tr_idx = np.setdiff1d(np.arange(N), va_idx, assume_unique=False)
            Xtr, Ytr = X[tr_idx], y_par[tr_idx]
            Xva, Yva = X[va_idx], y_par[va_idx]

            mlp = _build_mlp(D, hidden, alpha, epochs, optimizer, optimizer_params)

            mlp.fit(Xtr.tolist(), Ytr.tolist())

            yva_cls, yva_prob = _predict_classes(mlp, Xva)
            acc = _accuracy(Yva, yva_cls)
            accs.append(acc)

            print(f"\nFold {i + 1}/{k}  Acc={acc * 100:.1f}%  (val n={len(va_idx)})")
            print("Predicciones (Validación del fold):")

            df_fold = pd.DataFrame({
                "fold": i + 1,
                "idx": va_idx,
                "digit": labels[va_idx],
                "true_parity": Yva.flatten().astype(int),
                "predicted": yva_cls.flatten().astype(int),
                "prob": yva_prob.flatten()
            }).sort_values("idx").reset_index(drop=True)

            for j in range(len(df_fold)):
                r = df_fold.iloc[j]

                print(
                    f"idx={int(r.idx):02d}  dígito={int(r.digit)}  y*={int(r.true_parity)}   ŷ={int(r.predicted)}   p={r.prob:.3f}")

            rows.append({"fold": i + 1, "val_n": len(va_idx), "acc": acc})
            all_val_rows.append(df_fold)

        print(f"\nK-Fold   Acc media={np.mean(accs) * 100:.2f}%   ± {np.std(accs) * 100:.2f}%")

        df_k = pd.DataFrame(rows)
        df_k_all_preds = pd.concat(all_val_rows, ignore_index=True).sort_values(["fold", "idx"]).reset_index(drop=True)
        return {"summary": df_k, "val_predictions": df_k_all_preds}


    else:
        raise ValueError(f"split_mode desconocido: {split_mode}")


## item 3  // TODO

def run_digits(
    data_path="data/TP3-ej3-digitos.txt", split_mode="stratified_kfold", val_ratio=0.2, k=5,
    hidden=24, alpha=0.05, epochs=600, optimizer="adam", optimizer_params=None, seed=0, show_all_preds=True
):
    return -1

# === modos de partición ===

def split_holdout(n, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - val_ratio))
    tr, va = idx[:cut], idx[cut:]
    return tr, va

def split_stratified(y_bin, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    y = y_bin.flatten().astype(int)
    idx_even = np.where(y == 1)[0]
    idx_odd  = np.where(y == 0)[0]
    rng.shuffle(idx_even); rng.shuffle(idx_odd)

    def _split(idx):
        cut = int(len(idx) * (1 - val_ratio))
        return idx[:cut], idx[cut:]

    tr_e, va_e = _split(idx_even)
    tr_o, va_o = _split(idx_odd)
    tr = np.concatenate([tr_e, tr_o])
    va = np.concatenate([va_e, va_o])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

def split_kfold(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def split_stratified_kfold(y_bin, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = y_bin.flatten().astype(int)
    idx_even = np.where(y == 1)[0]
    idx_odd  = np.where(y == 0)[0]
    rng.shuffle(idx_even); rng.shuffle(idx_odd)
    folds_e = np.array_split(idx_even, k)
    folds_o = np.array_split(idx_odd, k)
    folds = [np.concatenate([folds_e[i], folds_o[i]]) for i in range(k)]
    # opcional: reordenar cada fold
    for i in range(k):
        rng.shuffle(folds[i])
    return folds


# ---------------

def _build_mlp(input_dim, hidden, alpha, epochs, optimizer, optimizer_params):
    act_h, dact_h = logistic_function_factory()
    act_o, dact_o = logistic_function_factory()
    mlp = MLP(
        layer_sizes=[input_dim, hidden, 1],
        activations=[(act_h, dact_h), (act_o, dact_o)],
        alpha=alpha,
        max_iter=epochs,
        optimizer=optimizer,
        optimizer_params=(optimizer_params or {})
    )
    return mlp

def _predict_classes(mlp, X):
    # predict por patrón, salida escalar
    probs = np.array([mlp.predict(x.tolist() if not isinstance(x, list) else x)[0] for x in X]).reshape(-1, 1)
    cls = (probs >= 0.5).astype(int)
    return cls, probs

def _accuracy(y_true, y_hat):
    return float((y_true == y_hat).mean())

def _confusion(y_true, y_hat):
    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    return tp, fp, fn, tn


