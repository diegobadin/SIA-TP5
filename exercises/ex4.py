from typing import List

import numpy as np
import time
from src.mlp.mlp import MLP
from src.mlp.activations import TANH, SIGMOID, RELU, SOFTMAX
from src.mlp.erorrs import CrossEntropyLoss
from src.mlp.optimizers import Adam, Momentum, SGD
from utils.graphs import plot_acc_curves, plot_kfold_accuracies, plot_loss, train_with_acc_curves, \
    plot_multiple_acc_curves, plot_multiple_loss, plot_time_table
from utils.metrics import accuracy_score, cross_validate, save_confusion_with_heatmap
from utils.splits import StratifiedKFold

import os, urllib.request

def run(experiment):
    def load_mnist_npz(path="mnist.npz"):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        if not os.path.exists(path):
            print(f"Descargando MNIST a {path} ...")
            urllib.request.urlretrieve(url, path)
        with np.load(path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test,  y_test  = data["x_test"],  data["y_test"]
        return (x_train, y_train), (x_test, y_test)

    (x_train, y_train_labels), (x_test, y_test_labels) = load_mnist_npz()

    # verifición que se descargó correctamente la data
    with np.load("mnist.npz") as f:
        assert set(f.keys()) == {"x_train", "x_test", "y_train", "y_test"}, \
            f"Las claves del archivo no son las esperadas: {list(f.keys())}"
        assert f["x_train"].shape == (60000, 28, 28), \
            f"x_train tiene shape {f['x_train'].shape}, esperado (60000, 28, 28)"
        assert f["x_test"].shape == (10000, 28, 28), \
            f"x_test tiene shape {f['x_test'].shape}, esperado (10000, 28, 28)"
        
    # Reshape, normalizar y convertir a float32
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

    #Etiquetas one-hot. Lo que hace es convertir el vector de etiquetas en una matriz one-hot
    y_train = np.eye(10)[y_train_labels]
    y_test = np.eye(10)[y_test_labels]

    def make_model(seed):
        return MLP(
            layer_sizes=[784, 128, 10], # 784 = 28*28, 128 capa oculta, 10 = 10 clases (0-9)
            activations=[TANH, SIGMOID],
            loss=CrossEntropyLoss(),
            optimizer=Adam(lr=0.001),
            seed=seed,
            w_init_scale=0.05 # para no saturar la activación desde el principio
        )


    SEED = 42
    if(experiment):
        model = make_model(SEED)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=SEED)

        # Curvas train/test con UN fold
        print("Iniciando Curvas train/test con UN fold")
        tr_idx, te_idx = next(splitter.split(x_train, y_train_labels))
        Xtr, Ytr = x_train[tr_idx], y_train[tr_idx] #tr = train
        Xva, Yva = x_train[te_idx], y_train[te_idx] #va = validation

        losses, tr_accs, va_accs = train_with_acc_curves(
            model, Xtr, Ytr, Xva, Yva,
            epochs=20,
            batch_size=64,         # mini-batch
            shuffle=True,
            verbose=True
        )

        plot_loss(losses, loss_name=type(model.loss).__name__, title="MNIST - Loss vs Época", fname="mnist_loss.png")
        plot_acc_curves(tr_accs, va_accs, title="MNIST - Accuracy (Train vs Val) vs Época", fname="mnist_acc.png")

        print("Iniciando Cross-validate")

        cv_results = cross_validate(
            model_factory=lambda: make_model(SEED),
            X=x_train,
            Y=y_train,
            splitter=splitter,
            fit_kwargs=dict(epochs=20, batch_size=64, verbose=True),
            scoring=accuracy_score,
        )

        title = f"MNIST · {splitter}"
        plot_kfold_accuracies(cv_results["folds"], title=title)

        print(f"[KFold] mean_acc={cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

        # Evaluación final en el set de TEST oficial de Keras

        print("Iniciando Evaluación final en el set de TEST oficial de Keras [Reentrenado con todo el conjunto de entrenamiento]")
        # Reentrenar un modelo con TODO el conjunto de entrenamiento
        final_model = make_model(SEED)
        final_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=True)

        test_acc = accuracy_score(y_test, final_model.predict(x_test))
        print(f"[TEST] Accuracy={test_acc:.4f}")
    else:
        run_depth_experiment(x_train, y_train, y_train_labels, SEED, x_test, y_test)


ARCHITECTURES = {
    "A": [784, 128, 10],
    "B": [784, 256, 128, 10],
    "C": [784, 512, 256, 128, 10]
}


def make_experiment_model(arch_key, seed):
    layer_sizes = ARCHITECTURES[arch_key]

    num_hidden_layers = len(layer_sizes) - 2
    activations = [RELU] * num_hidden_layers + [SOFTMAX]

    return MLP(
        layer_sizes=layer_sizes,
        activations=activations,
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=0.001),
        seed=seed,
        w_init_scale=0.05
    )


def run_depth_experiment(x_train, y_train, y_train_labels, seed, x_test, y_test):
    ARCHITECTURES_TO_RUN = ["A", "B", "C"]
    NUM_RUNS = 3

    comparison_results_mean = {}
    comparison_losses_mean = {}
    mean_times = {}

    # Usar un Split ÚNICO (80% Train, 20% Validation)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, seed=seed)
    tr_idx, te_idx = next(splitter.split(x_train, y_train_labels))
    Xtr, Ytr = x_train[tr_idx], y_train[tr_idx]
    Xva, Yva = x_train[te_idx], y_train[te_idx]

    print("--- Beginning of depth experiment ---")

    for arch_key in ARCHITECTURES_TO_RUN:
        arch_sizes = ARCHITECTURES[arch_key]
        print(f"\n[Modelo {arch_key}] {arch_sizes}")

        all_va_accs: List[np.ndarray] = []
        all_losses: List[np.ndarray] = []
        all_times: List[float] = []

        for run in range(NUM_RUNS):
            print(f"   --> Run {run + 1}/{NUM_RUNS}...")

            model = make_experiment_model(arch_key, seed + run)
            start_time = time.time()

            losses, tr_accs, va_accs = train_with_acc_curves(
                model, Xtr, Ytr, Xva, Yva,
                epochs=20,
                batch_size=64,
                shuffle=True,
                verbose=True
            )

            end_time = time.time()

            all_va_accs.append(va_accs)
            all_losses.append(losses)
            all_times.append(end_time - start_time)

            if run == NUM_RUNS - 1:
                plot_acc_curves(tr_accs, va_accs,
                                title=f"MNIST - Acc. (Arch. {arch_key}) - Último Run",
                                fname=f"mnist_acc_{arch_key}_last_run.png")

                preds_va = model.predict(Xva)
                save_confusion_with_heatmap(
                    y_true=Yva,
                    y_pred=preds_va,
                    filepath_prefix=f"results/confusion_arch_{arch_key}_val_last_run",
                    comment=f"Arquitectura {arch_key} en Validación (Run {run + 1}). Final Acc: {va_accs[-1]:.4f}",
                    class_labels=list(range(10))
                )
                print(f"[{arch_key}] Matriz de confusión guardada (último run).")

        va_accs_mean = np.mean(all_va_accs, axis=0)
        losses_mean = np.mean(all_losses, axis=0)
        mean_time = np.mean(all_times)

        comparison_results_mean[arch_key] = va_accs_mean
        comparison_losses_mean[arch_key] = losses_mean
        mean_times[arch_key] = mean_time

        print(f"[{arch_key}] Tiempo promedio de corrida: {mean_time:.2f} s")
        print(f"[{arch_key}] Accuracy promedio final: {va_accs_mean[-1]:.4f}")


    plot_multiple_acc_curves(comparison_results_mean,
                             title=f"MNIST - Comparación de Accuracy PROMEDIO ({NUM_RUNS} runs) por Profundidad (Validation)",
                             fname="mnist_depth_comparison_acc_mean.png")

    plot_multiple_loss(comparison_losses_mean,
                       title=f"MNIST - Comparación de Pérdida PROMEDIO ({NUM_RUNS} runs) por Profundidad (Train)",
                       fname="mnist_depth_comparison_loss_mean.png")

    # c. Tabla de Tiempos Promedio
    plot_time_table(mean_times)

    print("\n--- End of depth experiment ---")