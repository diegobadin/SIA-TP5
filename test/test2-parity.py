# test/test2_paridad.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # para importar src/ y utils/

import time
import numpy as np
import matplotlib.pyplot as plt

# === Tu stack MLP (ajustá imports si difieren en tu repo) ===
from src.mlp.activations import TANH, SIGMOID
from src.mlp.erorrs import CrossEntropyLoss  # <- si en tu repo es 'errors', cambiá este import
from src.mlp.mlp import MLP
from src.mlp.optimizers import Adam, Momentum, SGD

# Utils
from utils.parse_csv_data import parse_digits_7x5_txt
from utils.splits import HoldoutSplit


def _binarize(p, thr=0.5):
    return (np.asarray(p).reshape(-1) >= thr).astype(float)

def _acc(y_true, y_pred_bin):
    y_true = np.asarray(y_true).reshape(-1)
    return float(np.mean(y_true == y_pred_bin))


def run_test2_paridad(
    txt_path: str = "data/TP3-ej3-digitos_extra.txt",
    hidden: int = 5,               # arquitectura [35, 5, 1]
    epochs: int = 50,
    batch_size: int = 16,
    seed: int = 123,
    w_init_scale: float = 0.2,
    # tasas por optimizador
    lr_gd: float = 1e-2,
    lr_mom: float = 1e-2, beta_mom: float = 0.9,
    lr_adam: float = 1e-3,
    # evaluación
    acc_threshold: float = 0.95,   # época en que el train acc cruza este umbral
    threshold: float = 0.5,        # umbral para convertir probas→clase
    # gráficos
    save_plots: bool = True,
    show_plots: bool = True,
    out_dir: str = "test/plots"
):
    # ---------- 1) Dataset paridad ----------
    X, labels = parse_digits_7x5_txt(txt_path)  # X: (N,35), labels: dígitos 0..9
    X = X * 2.0 - 1.0                           # a [-1,1] para TANH
    Y = (labels % 2).astype(float).reshape(-1, 1)  # 0=par, 1=impar
    in_dim = X.shape[1]
    assert in_dim == 35, f"Se esperaban 35 features, llegaron {in_dim}"

    # Holdout 70/30 estratificado
    splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=seed, stratify=True)
    tr_idx, te_idx = next(splitter.split(X, labels))  # estratificar por dígito
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    # ---------- 2) Definir optimizadores ----------
    OPTS = {
        "GD":       lambda: SGD(lr=lr_gd),
        "Momentum": lambda: Momentum(lr=lr_mom, beta=beta_mom),
        "Adam":     lambda: Adam(lr=lr_adam),
    }

    os.makedirs(out_dir, exist_ok=True)

    curves = {}   # guardamos curvas por optimizador
    summary = []  # métricas resumidas

    # ---------- 3) Entrenar cada optimizador (misma semilla/init) ----------
    for name, opt_factory in OPTS.items():
        model = MLP(
            layer_sizes=[in_dim, hidden, 1],
            activations=[TANH, SIGMOID],
            loss=CrossEntropyLoss(),      # BCE con sigmoid
            optimizer=opt_factory(),
            seed=seed,
            w_init_scale=w_init_scale
        )

        losses, tr_accs, te_accs = [], [], []
        t0 = time.perf_counter()
        for ep in range(epochs):
            last_loss = model.fit(
                Xtr, Ytr, epochs=1, batch_size=batch_size, shuffle=True, verbose=False
            )[-1]
            losses.append(last_loss)

            # predicciones y métricas
            ytr_hat = model.predict(Xtr)  # probs en [0,1]
            yte_hat = model.predict(Xte)
            tr_accs.append(_acc(Ytr, _binarize(ytr_hat, threshold)))
            te_accs.append(_acc(Yte, _binarize(yte_hat, threshold)))

        dur = time.perf_counter() - t0

        # Época en que el train acc cruza el umbral
        epoch_at_thr = next((i for i, a in enumerate(tr_accs, start=1) if a >= acc_threshold), None)

        curves[name] = dict(losses=np.array(losses), tr=np.array(tr_accs), te=np.array(te_accs))
        summary.append({
            "opt": name,
            "lr": (lr_gd if name=="GD" else lr_mom if name=="Momentum" else lr_adam),
            "epoch@thr": epoch_at_thr if epoch_at_thr is not None else "-",
            "final_train_acc": float(tr_accs[-1]),
            "final_test_acc": float(te_accs[-1]),
            "best_test_acc": float(np.max(te_accs)),
            "time_s": dur,
            "osc_up_steps": int(np.sum(np.diff(losses) > 0.0)),  # subidas del loss (oscilación)
        })

    # ---------- 4) Plots comparativos ----------
    # Loss overlay
    plt.figure()
    for name in OPTS.keys():
        plt.plot(curves[name]["losses"], label=name)
    plt.xlabel("Época"); plt.ylabel("Loss (BCE)")
    plt.title("Paridad · Loss train por optimizador")
    plt.legend(); plt.tight_layout()
    if save_plots:
        f = os.path.join(out_dir, "parity_loss_overlay.png")
        plt.savefig(f, dpi=200, bbox_inches="tight"); print(f"Guardado: {f}")
    if show_plots: plt.show()
    plt.close()

    # Accuracy test overlay
    plt.figure()
    for name in OPTS.keys():
        plt.plot(curves[name]["te"], label=name)
    plt.xlabel("Época"); plt.ylabel("Accuracy (test)")
    plt.title("Paridad · Accuracy test por optimizador")
    plt.legend(); plt.tight_layout()
    if save_plots:
        f = os.path.join(out_dir, "parity_acc_test_overlay.png")
        plt.savefig(f, dpi=200, bbox_inches="tight"); print(f"Guardado: {f}")
    if show_plots: plt.show()
    plt.close()

    # Accuracy train overlay (opcional pero útil para ver velocidad de convergencia)
    plt.figure()
    for name in OPTS.keys():
        plt.plot(curves[name]["tr"], label=name)
    plt.xlabel("Época"); plt.ylabel("Accuracy (train)")
    plt.title("Paridad · Accuracy train por optimizador")
    plt.legend(); plt.tight_layout()
    if save_plots:
        f = os.path.join(out_dir, "parity_acc_train_overlay.png")
        plt.savefig(f, dpi=200, bbox_inches="tight"); print(f"Guardado: {f}")
    if show_plots: plt.show()
    plt.close()

    # ---------- 5) Resumen en tabla ----------
    print("\nResumen por optimizador (holdout 70/30)")
    print("| Opt      |   lr    | Época@{:.0%} | Acc train fin | Acc test fin | Acc test max | up(loss) | tiempo(s) |".format(acc_threshold))
    print("|----------|---------|-------------|---------------|--------------|--------------|----------|-----------|")
    for row in summary:
        print(f"| {row['opt']:<8} | {row['lr']:<7.5f} | {str(row['epoch@thr']):^11} | "
              f"{row['final_train_acc']:^13.3f} | {row['final_test_acc']:^12.3f} | "
              f"{row['best_test_acc']:^12.3f} | {row['osc_up_steps']:^8} | {row['time_s']:^9.3f} |")

    print("\nEsperado:")
    print("- GD → más lento, puede estancarse.")
    print("- Momentum → más estable y suele llegar antes.")
    print("- Adam → típicamente el más rápido y con menos serrucho.")
    return summary


if __name__ == "__main__":
    run_test2_paridad(
        txt_path="data/TP3-ej3-digitos_extra.txt",
        hidden=5,            # [35, 5, 1]
        epochs=50,
        batch_size=16,
        seed=123,
        w_init_scale=0.2,
        lr_gd=1e-2,
        lr_mom=1e-2, beta_mom=0.9,
        lr_adam=1e-3,
        acc_threshold=0.95,
        threshold=0.5,
        save_plots=True,
        show_plots=True,
        out_dir="test/plots"
    )
