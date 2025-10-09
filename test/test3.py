# test/test3.py — experimento básico con guardado de gráficos
import os
from pathlib import Path
import matplotlib.pyplot as plt

from src.perceptron import Perceptron
from src.training_algorithms import nonlinear_perceptron
from utils.activations import (
    scaled_logistic_function_factory,
    scaled_tanh_function_factory,
)
from utils.parse_csv_data import parse_csv_data


def run_eta_sweep_basic(
    csv_file_path: str,
    activation: str = "tanh",
    beta: float = 1.0,
    etas = (0.001, 0.01, 0.1, 0.5),
    max_iter: int = 50,
    mode: str = "online",
    batch_size: int = 8,
    shuffle: bool = True,
    rel_factor: float = 0.5,        # época en que MSE cae al 50% del MSE inicial
    # --- NUEVO: opciones de guardado/visualización ---
    save_plots: bool = True,
    show_plots: bool = True,
    out_dir: str = "test/plots",
    out_name: str | None = None,    # si None => se arma a partir del CSV y la activación
    ylog: bool = False              # escala log en eje Y del overlay (opcional)
):
    # 1) Cargar y estandarizar como en tu pipeline
    x_features, y_labels = parse_csv_data(csv_file_path)
    if not x_features:
        print("No hay datos.")
        return []

    from utils.stats import fmean, pstdev
    cols = list(zip(*x_features))
    means = [fmean(col) for col in cols]
    stdevs = [max(1e-8, pstdev(col)) for col in cols]
    x_std = [[(val - m)/s for val, m, s in zip(row, means, stdevs)] for row in x_features]
    x_with_bias = [[1] + row for row in x_std]
    n_inputs = len(x_std[0])

    # 2) Activación escalada a [y_min, y_max]
    y_min = min(y_labels); y_max = max(y_labels)
    if activation == "tanh":
        act, dact = scaled_tanh_function_factory(beta, y_min, y_max)
        act_name = "tanh[scaled]"
    else:
        act, dact = scaled_logistic_function_factory(beta, y_min, y_max)
        act_name = "logistic[scaled]"

    # preparar paths de salida
    csv_base = Path(csv_file_path).stem
    base_name = out_name or f"eta_sweep_{csv_base}_{act_name}_b{beta}".replace(" ", "").replace("[","").replace("]","")
    out_path = Path(out_dir); 
    if save_plots: out_path.mkdir(parents=True, exist_ok=True)

    title = os.path.basename(csv_file_path)
    results = []
    N = len(y_labels)

    # 3) Overlay de curvas de MSE por época
    plt.figure()
    if ylog: plt.yscale("log")

    etas = tuple(sorted(set(etas)))
    all_err_curves = {}  # para el overlay

    for eta in etas:
        alpha = eta
        p = Perceptron(
            n_inputs=n_inputs,
            activation=act,
            activation_derivative=dact,
            alpha=alpha,
            max_iter=max_iter,
            training_algorithm=nonlinear_perceptron,
            mode=mode,
            batch_size=batch_size,
            shuffle=shuffle
        )
        p.fit(x_with_bias, y_labels)

        # SSE → MSE
        err_sse = list(getattr(p, "error_history", [])) or [float("nan")]
        err_mse = [e / max(1, N) for e in err_sse]
        all_err_curves[eta] = err_mse

        mse0 = float(err_mse[0])
        min_mse = float(min(err_mse))
        final_mse = float(err_mse[-1])

        epoch_at_rel = next((i for i, e in enumerate(err_mse, start=1) if e <= rel_factor * mse0), None)

        results.append({
            "eta": eta,
            "alpha": alpha,
            "epochs": len(err_mse),
            "epoch_at_rel": epoch_at_rel if epoch_at_rel is not None else "-",
            "mse_init": mse0,
            "mse_min": min_mse,
            "mse_final": final_mse,
            "reduction_%": 100.0 * (mse0 - min_mse) / max(1e-12, mse0),
        })

        # agregar la curva al overlay
        plt.plot(err_mse, label=f"η={eta}")

    # cerrar overlay
    plt.xlabel("Época"); plt.ylabel("Error (MSE)")
    plt.title(f"Perceptrón no lineal — {title}")
    plt.legend(); plt.tight_layout()
    if save_plots:
        f_overlay = out_path / f"{base_name}_overlay.png"
        plt.savefig(f_overlay, dpi=200, bbox_inches="tight")
        print(f"Guardado: {f_overlay}")
    if show_plots: plt.show()
    plt.close()

    # 4) Grafiquito resumen: Reducción (%) vs η
    plt.figure()
    xs = [r["eta"] for r in results]
    ys = [r["reduction_%"] for r in results]
    plt.plot(xs, ys, marker="o")
    plt.xscale("log")  # η suele cubrir órdenes de magnitud
    plt.xlabel("η (tasa de aprendizaje, escala log)")
    plt.ylabel("Reducción del MSE (%)")
    plt.title("Reducción (%) vs η")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_plots:
        f_redux = out_path / f"{base_name}_reduction_vs_eta.png"
        plt.savefig(f_redux, dpi=200, bbox_inches="tight")
        print(f"Guardado: {f_redux}")
    if show_plots: plt.show()
    plt.close()

    # 5) Tabla simple (con mejora relativa)
    umbral_txt = f"{int(rel_factor*100)}%"
    print("\nResumen por η (MSE)")
    print(f"|   η    | Épocas | Época@{umbral_txt}    |  MSE0   |  MSEmin | MSEfinal | Reducción % |")
    print(  "|--------|--------|--------------|---------|---------|---------|-------------|")
    for r in results:
        print(f"| {r['eta']:<6} | {r['epochs']:^6} | {str(r['epoch_at_rel']):^12} | "
              f"{r['mse_init']:^7.3g} | {r['mse_min']:^7.3g} | {r['mse_final']:^7.3g} | {r['reduction_%']:^11.1f} |")

    return results


if __name__ == "__main__":
    run_eta_sweep_basic(
        csv_file_path="data/TP3-Ej2-Conjunto.csv",
        activation="tanh",
        beta=0.1,
        etas=(0.001, 0.01, 0.1, 0.5),
        max_iter=40,
        mode="online",
        batch_size=8,
        shuffle=True,
        rel_factor=0.5,        # umbral relativo = 50% del MSE inicial
        save_plots=True,
        show_plots=True,
        out_dir="test/plots",
        out_name=None,         # o "mi_experimento_eta"
        ylog=False
    )
