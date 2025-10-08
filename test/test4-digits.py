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
    train_with_error_and_acc_curves
from utils.metrics import cross_validate, accuracy_score

from utils.parse_csv_data import parse_digits_7x5_txt
from utils.splits import KFold, StratifiedKFold, HoldoutSplit

def run_test4_digits_baseline(
    txt_path: str = "data/TP3-ej3-digitos_extra.txt",
    lr: float = 0.5,
    epochs: int = 30,
    hidden: int = 16,
    batch_size: int = 4,
    seed: int = 42):

    X, labels = parse_digits_7x5_txt(txt_path)
    # Escalar a [-1,1] para TANH
    X = X * 2.0 - 1.0
    Y = np.eye(10, dtype=float)[labels]
    in_dim, out_dim = X.shape[1], Y.shape[1]

    model = MLP(
        layer_sizes=[in_dim, hidden, out_dim],
        activations=[TANH, SOFTMAX],
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=lr),
        seed=seed,
        w_init_scale=0.2
    )
    print("X", X.shape, "labels", Y.shape)

    splitter = StratifiedKFold(n_splits=2, shuffle=True, seed=123)

    
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


    plot_two_loss_curves(
        losses_tr, losses_te,
        title=f"Digitos · MLP [{in_dim},{hidden},{out_dim}] · MSE train/test",
        fname="Digitos_exp4_baseline_mlp_mse.png"
    )
    plot_two_acc_curves(
        acc_tr, acc_te,
        title=f"Digitos · MLP [{in_dim},{hidden},{out_dim}] · Accuracy train/test",
        fname="Digitos_exp4_baseline_mlp_acc.png"
    )
    plt.show()





if __name__ == "__main__":
    run_test4_digits_baseline()
