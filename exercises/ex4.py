import numpy as np
from src.mlp.mlp import MLP
from src.mlp.activations import TANH, SIGMOID
from src.mlp.erorrs import CrossEntropyLoss
from src.mlp.optimizers import Adam, Momentum, SGD
from utils.graphs import plot_acc_curves, plot_kfold_accuracies, plot_loss, train_with_acc_curves
from utils.metrics import accuracy_score, cross_validate
from utils.splits import StratifiedKFold

import os, urllib.request

def run():
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

    print("Iniciando Evaluación final en el set de TEST oficial de Keras")
    # Reentrenar un modelo con TODO el conjunto de entrenamiento
    final_model = make_model(SEED)
    final_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=True)

    test_acc = accuracy_score(y_test, final_model.predict(x_test))
    print(f"[TEST] Accuracy={test_acc:.4f}")

