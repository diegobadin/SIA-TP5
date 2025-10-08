# exercises/ex3.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.mlp.activations import (
    TANH, SIGMOID, SOFTMAX
)

from src.mlp.erorrs import (
     CrossEntropyLoss, MSELoss
    
)
from src.mlp.mlp import MLP

from src.mlp.optimizers import (
    Adam, Momentum, SGD
)
from utils.graphs import plot_loss, plot_acc_curves, train_with_acc_curves, plot_kfold_accuracies, \
    plot_decision_boundary, evaluate_digits_with_noise, plot_two_loss_curves, plot_two_acc_curves, \
    train_with_error_and_acc_curves, plot_multiple_acc_curves, plot_multiple_loss
from utils.metrics import cross_validate, accuracy_score, save_confusion_with_heatmap, save_confusion_counts

from utils.parse_csv_data import parse_digits_7x5_txt
from utils.splits import KFold, StratifiedKFold, HoldoutSplit


def run(item: str):
    item = (item or "").lower().strip()

    if item == "xor":
        run_ex3_xor()
    elif item == "parity":
        run_ex3_paridad()
    elif item == "digits":
        run_ex3_digitos()
    elif item == "architecture_comparison":
        run_architecture_comparison_experiment()
    else:
        print("Uso: python main.py ex3 <xor | parity | digits | architecture_comparison>")


def run_ex3_xor():
    X = np.array([[-1, 1],
                  [1, -1],
                  [-1, -1],
                  [1, 1]], dtype=float)
    Y = np.array([[1], [1], [0], [0]], dtype=float)

    model = MLP(
        layer_sizes=[2, 6, 1],
        activations=[TANH, SIGMOID],
        loss=MSELoss(),
        optimizer=Adam(lr=0.1),
        seed=7,
        w_init_scale=0.5
    )

    # Split 50/50 para poder graficar acc de train/test
    splitter = HoldoutSplit(test_ratio=0.5, shuffle=True, seed=7, stratify=True)

    tr_idx, te_idx = next(splitter.split(X, Y))

    # Construcción de sets con los índices; únicamente necesario en xor
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    losses, tr_accs, te_accs = train_with_acc_curves(
        model, Xtr, Ytr, Xte, Yte,
        epochs=200, batch_size=1, shuffle=True, verbose=True
    )

    plot_loss(
        losses,
        loss_name=type(model.loss).__name__,
        title="XOR - Loss vs Época",
        fname="xor_loss.png",
    )
    plot_acc_curves(
        tr_accs, te_accs,
        title="XOR - Accuracy vs Época",
        fname="xor_acc.png",
    )
    plot_decision_boundary(model, X, Y, "XOR - Decision boundary")
    plt.show()

def run_ex3_paridad(
    txt_path: str = "data/TP3-ej3-digitos_extra.txt",
    lr: float = 0.05,
    epochs: int = 30,
    hidden: int = 16,
    batch_size: int = 1,
    seed: int = 42,
):
    X, labels = parse_digits_7x5_txt(txt_path)
    # Escalar a [-1,1] para TANH
    X = X * 2.0 - 1.0
    # 0 = par, 1 = impar
    Y = (labels % 2).astype(float).reshape(-1, 1)
    in_dim = X.shape[1]

    model = MLP(
        layer_sizes=[in_dim, hidden, 1],
        activations=[TANH, SIGMOID],
        loss=CrossEntropyLoss(),  # BCE con sigmoid
        optimizer=SGD(lr=lr),
        seed=seed,
        w_init_scale=0.2
    )

    # Split holdout estratificado 70/30 (rinde un único fold). Útil para graficar train/test.
    splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=123, stratify=True)

    # splitter = KFold(n_splits=5, shuffle=True, seed=123)

    # splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=123)

    # --- Curvas con UN fold (para graficar train/test) ---
    tr_idx, te_idx = next(splitter.split(X, Y))
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    losses_tr, losses_te, acc_tr, acc_te = train_with_error_and_acc_curves(
        model, Xtr, Ytr, Xte, Yte,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=True
    )

    # === Experiment 1 ===

    plot_two_loss_curves(
        losses_tr, losses_te,
        title=f"Paridad · MLP [35,{hidden},1] · MSE train/test (η={lr}, épocas={epochs})",
        fname="paridad_exp1_mlp_mse.png"
    )
    plot_two_acc_curves(
        acc_tr, acc_te,
        title=f"Paridad · MLP [35,{hidden},1] · Accuracy train/test",
        fname="paridad_exp1_mlp_acc.png"
    )

    # ==== desde antes de los exp

    plot_loss(losses_tr,
              loss_name=type(model.loss).__name__,
              title="Parity - Loss vs Época",
              fname="parity_loss.png")
    plot_acc_curves(acc_tr, acc_te, title="Parity - Accuracy vs Época", fname="parity_acc.png")

    # OJO: StratifiedKFold se puede usar nomás cuando tenemos más de una sola muestra de test por dígito
    # CV no tiene sentido si estamos corriendo con holdout
    res = cross_validate(
        model_factory=lambda: MLP([in_dim, 32, 1], [TANH, SIGMOID],
                                  CrossEntropyLoss(), Adam(lr=1e-3),
                                  seed=123, w_init_scale=0.2),
        X=X, Y=Y,
        splitter=splitter,
        fit_kwargs=dict(epochs=50, batch_size=16, verbose=False),
        scoring=accuracy_score,
    )
    title = f"Parity · {splitter}"
    plot_kfold_accuracies(res["folds"], title=title)

    run_paridad_complexity_experiment(X, Y)

    plt.show()


def run_ex3_digitos():
    X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos_extra.txt")
    # Escalar a [-1,1] para TANH
    X = X * 2.0 - 1.0
    Y = np.eye(10, dtype=float)[labels]
    in_dim, out_dim = X.shape[1], Y.shape[1]

    model = MLP(
        layer_sizes=[in_dim, 32, 16, out_dim],
        activations=[TANH, TANH, SOFTMAX],
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=1e-3),
        seed=42,
        w_init_scale=0.2
    )

    # Split holdout estratificado 70/30 (rinde un único fold). Útil para graficar train/test.
    # splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=123, stratify=True)

    # splitter = KFold(n_splits=5, shuffle=True, seed=123)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=123)  # Aca se le pase el numero que le pase va a dividir en dos, lo que se puede hacer el ir duplicando la data para que pueda ir agrupando mas

    # --- Curvas con UN fold (para graficar train/test)
    tr_idx, te_idx = next(splitter.split(X, Y))
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    losses, tr_accs, te_accs = train_with_acc_curves(
        model, Xtr, Ytr, Xte, Yte, epochs=50, batch_size=16, shuffle=True, verbose=True
    )
    plot_loss(losses, loss_name=type(model.loss).__name__, title="Digits - Loss vs Época", fname="digits_loss.png")
    plot_acc_curves(tr_accs, te_accs, title="Digits - Accuracy vs Época", fname="digits_acc.png")

    # OJO: StratifiedKFold se puede usar nomás cuando tenemos más de una sola muestra de test por dígito
    res = cross_validate(
        model_factory=lambda: MLP([in_dim, 32, 16, out_dim],
                                  [TANH, TANH, SOFTMAX],
                                  CrossEntropyLoss(), Adam(lr=1e-3),
                                  seed=42, w_init_scale=0.2),
        X=X, Y=Y,
        splitter=splitter,
        fit_kwargs=dict(epochs=50, batch_size=16, verbose=False),
        scoring=accuracy_score,
    )
    title = f"Digits · {splitter}"
    plot_kfold_accuracies(res["folds"], title=title)

    run_noise_experiment(X, Y, in_dim=in_dim, out_dim=out_dim)

    plt.show()


def run_architecture_comparison_experiment():
    """
    Experimento: Comparación de arquitecturas MLP para reconocimiento de dígitos
    Compara 3 arquitecturas diferentes para estudiar el trade-off entre complejidad y generalización.
    """
    print("=" * 70)
    print("  EXPERIMENTO: COMPARACIÓN DE ARQUITECTURAS MLP")
    print("=" * 70)
    
    # Cargar datos
    X, labels = parse_digits_7x5_txt("data/TP3-ej3-digitos_extra.txt")
    # Escalar a [-1,1] para TANH
    X = X * 2.0 - 1.0
    Y = np.eye(10, dtype=float)[labels]
    in_dim, out_dim = X.shape[1], Y.shape[1]
    
    print(f"Datos: {len(X)} dígitos, {in_dim} características, {out_dim} clases")
        
    # Definir arquitecturas
    architectures = {
        'A': [in_dim, 10, out_dim],
        'B': [in_dim, 20, out_dim], 
        'C': [in_dim, 30, 15, out_dim]
    }
    
    # Hiperparámetros (ajustados para dataset pequeño)
    lr = 0.05
    epochs = 50 
    n_runs = 3
    
    print(f"Hiperparámetros: η={lr}, Épocas={epochs}, Runs={n_runs}")
    print(f"Arquitecturas: A={architectures['A']}, B={architectures['B']}, C={architectures['C']}")
    
    # Splitter para train/test (sin estratificación por el tamaño pequeño del dataset)
    splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=42, stratify=False)
    tr_idx, te_idx = next(splitter.split(X, Y))
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]
    
    print(f"Split: {len(Xtr)} train, {len(Xte)} test")
    
    # Almacenar resultados
    results = {}
    
    for arch_name, layer_sizes in architectures.items():
        print(f"\n{'='*20} Arquitectura {arch_name} {layer_sizes} {'='*20}")
        
        # Definir activaciones según la arquitectura
        if len(layer_sizes) == 3:  # A y B: [input, hidden, output]
            activations = [TANH, SOFTMAX]
        else:  # C: [input, hidden1, hidden2, output]
            activations = [TANH, TANH, SOFTMAX]
        
        # Almacenar accuracy por época para cada run
        train_accs_runs = []
        test_accs_runs = []
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}")
            
            # Crear modelo con nueva semilla para cada run
            model = MLP(
                layer_sizes=layer_sizes,
                activations=activations,
                loss=CrossEntropyLoss(),
                optimizer=Adam(lr=lr),
                seed=42 + run,  # Diferente semilla por run
                w_init_scale=0.2
            )
            
            # Entrenar y obtener curvas de accuracy
            losses, tr_accs, te_accs = train_with_acc_curves(
                model, Xtr, Ytr, Xte, Yte,
                epochs=epochs, batch_size=16, shuffle=True, verbose=False
            )
            
            train_accs_runs.append(tr_accs)
            test_accs_runs.append(te_accs)
            
            print(f"  Train Acc Final: {tr_accs[-1]:.4f}")
            print(f"  Test Acc Final:  {te_accs[-1]:.4f}")
        
        # Calcular promedio por época
        train_accs_avg = np.mean(train_accs_runs, axis=0)
        test_accs_avg = np.mean(test_accs_runs, axis=0)
        
        # Calcular desviación estándar
        train_accs_std = np.std(train_accs_runs, axis=0)
        test_accs_std = np.std(test_accs_runs, axis=0)
        
        results[arch_name] = {
            'train_acc_avg': train_accs_avg,
            'test_acc_avg': test_accs_avg,
            'train_acc_std': train_accs_std,
            'test_acc_std': test_accs_std,
            'layer_sizes': layer_sizes
        }
        
        print(f"\nArquitectura {arch_name} - Promedio Final:")
        print(f"  Train Acc: {train_accs_avg[-1]:.4f} ± {train_accs_std[-1]:.4f}")
        print(f"  Test Acc:  {test_accs_avg[-1]:.4f} ± {test_accs_std[-1]:.4f}")
    
    # === GRÁFICO COMPARATIVO ===
    plt.figure(figsize=(14, 8))
    
    colors = {'A': '#ff7f0e', 'B': '#2ca02c', 'C': '#d62728'}
    epochs_range = range(1, epochs + 1)
    
    # Plotear accuracy de test (generalización)
    for arch_name in ['A', 'B', 'C']:
        test_acc_avg = results[arch_name]['test_acc_avg']
        test_acc_std = results[arch_name]['test_acc_std']
        layer_sizes = results[arch_name]['layer_sizes']
        
        plt.plot(epochs_range, test_acc_avg, 
                label=f'Arquitectura {arch_name} {layer_sizes}', 
                linewidth=3, color=colors[arch_name], marker='o', markersize=2)
        
        # Agregar banda de desviación estándar (limitada a máximo 1.0)
        upper_bound = np.minimum(test_acc_avg + test_acc_std, 1.0)
        lower_bound = np.maximum(test_acc_avg - test_acc_std, 0.0)
        plt.fill_between(epochs_range, 
                        lower_bound, 
                        upper_bound, 
                        alpha=0.2, color=colors[arch_name])
    
    plt.xlabel('Época')
    plt.ylabel('Accuracy de Test')
    plt.title('Comparación de Arquitecturas MLP - Accuracy de Test vs Épocas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajustar límites del eje Y con padding para mejor visualización
    y_min = min(min(results[arch]['test_acc_avg']) for arch in ['A', 'B', 'C'])
    y_max = max(max(results[arch]['test_acc_avg']) for arch in ['A', 'B', 'C'])
    
    # Agregar padding del 5% arriba y abajo
    y_padding = (y_max - y_min) * 0.05
    plt.ylim(max(0, y_min - y_padding), min(1.05, y_max + y_padding))
    
    # Agregar líneas de referencia solo si están dentro del rango visible
    if y_max >= 0.9:
        plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Accuracy')
    if y_max >= 0.8:
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, label='80% Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # === RESUMEN FINAL ===
    print(f"\n{'='*70}")
    print("RESUMEN FINAL - ACCURACY DE TEST (GENERALIZACIÓN)")
    print(f"{'='*70}")
    
    for arch_name in ['A', 'B', 'C']:
        layer_sizes = results[arch_name]['layer_sizes']
        test_acc_final = results[arch_name]['test_acc_avg'][-1]
        test_acc_std_final = results[arch_name]['test_acc_std'][-1]
        
        print(f"Arquitectura {arch_name} {layer_sizes}:")
        print(f"  Accuracy Final: {test_acc_final:.4f} ± {test_acc_std_final:.4f}")
        
        # Calcular parámetros aproximados
        total_params = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        print(f"  Parámetros Aprox: {total_params}")
        print()
    
    # Análisis del trade-off
    print("ANÁLISIS DEL TRADE-OFF:")
    arch_a_final = results['A']['test_acc_avg'][-1]
    arch_b_final = results['B']['test_acc_avg'][-1] 
    arch_c_final = results['C']['test_acc_avg'][-1]
    
    best_arch = max([('A', arch_a_final), ('B', arch_b_final), ('C', arch_c_final)], key=lambda x: x[1])
    print(f"Mejor arquitectura: {best_arch[0]} con accuracy {best_arch[1]:.4f}")
    
    if arch_c_final < arch_b_final:
        print("⚠️  Arquitectura C muestra posible sobreajuste (menor generalización)")
    elif arch_b_final > arch_a_final:
        print("✅ Arquitectura B muestra mejor equilibrio complejidad-generalización")
    
    return results


ARCHITECTURES = {
    "A": [35, 5, 1],
    "B": [35, 10, 1],
    "C": [35, 10, 5, 1]
}


def make_model(arch_key, seed, lr):  # Adaptamos make_model
    layer_sizes = ARCHITECTURES[arch_key]

    num_hidden_layers = len(layer_sizes) - 2
    activations = [TANH] * num_hidden_layers + [SIGMOID]

    return MLP(
        layer_sizes=layer_sizes,
        activations=activations,
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=lr),
        seed=seed,
        w_init_scale=0.2
    )

def run_noise_experiment(X, Y, in_dim, out_dim):
    # --- Experimento con ruido ---
    model = MLP(
        layer_sizes=[in_dim, 20, out_dim],
        activations=[TANH, SOFTMAX],
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=1e-3),
        seed=42,
        w_init_scale=0.2
    )

    splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=123)  # Aca se le pase el numero que le pase va a dividir en dos, lo que se puede hacer el ir duplicando la data para que pueda ir agrupando mas

    # --- Curvas con UN fold (para graficar train/test)
    tr_idx, te_idx = next(splitter.split(X, Y))
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    losses, tr_accs, te_accs = train_with_acc_curves(
        model, Xtr, Ytr, Xte, Yte, epochs=50, batch_size=16, shuffle=True, verbose=True
    )

    evaluate_digits_with_noise(model, X, Y, noise_levels=[0.0, 0.1, 0.25, 0.5], n_show=10)



def run_paridad_complexity_experiment(X, Y, epochs=30, lr=0.05, seed=42):
    ARCHITECTURES_TO_RUN = ["A", "B", "C"]

    # Estructuras para los gráficos comparativos (necesitamos Train Loss, Test Loss y Test Acc)
    comp_test_acc = {}
    comp_train_loss = {}
    comp_test_loss = {}

    # Usar un Holdout Split 70/30 (el mismo de tu código)
    splitter = HoldoutSplit(test_ratio=0.3, shuffle=True, seed=123, stratify=True)
    tr_idx, te_idx = next(splitter.split(X, Y))
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]  # Xte/Yte es el set de Validación/Test del Holdout

    print("--- Iniciando Experimento 3: Sensibilidad a la Complejidad ---")

    for arch_key in ARCHITECTURES_TO_RUN:
        print(f"\n[Modelo {arch_key}] {ARCHITECTURES[arch_key]}")

        # 1. Crear modelo con Adam, lr=0.05
        model = make_model(arch_key, seed, lr)

        # 2. Entrenar y obtener curvas
        losses_tr, losses_te, acc_tr, acc_te = train_with_error_and_acc_curves(
            model, Xtr, Ytr, Xte, Yte,
            epochs=epochs,
            batch_size=1,  # Usamos batch_size=1 (SGD) para este experimento
            shuffle=True,
            verbose=False
        )

        # 3. Guardar curvas para la comparación
        comp_test_acc[arch_key] = acc_te
        comp_train_loss[arch_key] = losses_tr
        comp_test_loss[arch_key] = losses_te

        # --- 4. GRAFICADO INDIVIDUAL (Crucial para el análisis de C) ---
        # Curvas ACC y Loss individuales (las que ya tienes, pero renombradas)
        plot_two_acc_curves(acc_tr, acc_te, title=f"Paridad - Acc. (Arch. {arch_key})",
                            fname=f"paridad_acc_{arch_key}.png")
        plot_two_loss_curves(losses_tr, losses_te, title=f"Paridad - Loss (Arch. {arch_key})",
                             fname=f"paridad_loss_{arch_key}.png")

        # Matriz de Confusión (En el set de Test Xte/Yte del Holdout)
        preds_te = model.predict(Xte)
        final_acc = acc_te[-1]

        save_confusion_counts(
            y_true=Yte,
            y_pred=preds_te,
            filepath=f"results/confusion_counts_paridad_arch_{arch_key}_test.txt",
            positive_label=1,  # 1=Impar (por defecto), 0=Par
            threshold=0.5,
            comment=f"Arquitectura {arch_key} en Test. Acc: {final_acc:.4f}"
        )

    # --- 5. GRAFICADO COMPARATIVO FINAL ---

    # a. Comparación de Accuracy (Test)
    plot_multiple_acc_curves(comp_test_acc,
                             title="Paridad - Comparación de Accuracy (Test) vs Complejidad",
                             fname="paridad_comp_acc.png")

    # b. Comparación de Pérdida (Train)
    plot_multiple_loss(comp_train_loss,
                       title="Paridad - Comparación de Loss (Train) vs Complejidad",
                       fname="paridad_comp_loss_train.png")

    # c. Comparación de Pérdida (Test)
    plot_multiple_loss(comp_test_loss,
                       title="Paridad - Comparación de Loss (Test) vs Complejidad",
                       fname="paridad_comp_loss_test.png")

    print("\n--- Experimento de Complejidad Finalizado ---")