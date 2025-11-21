import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.autoencoder import Autoencoder
from src.mlp.activations import TANH, SIGMOID, RELU
from src.mlp.erorrs import MSELoss
from src.mlp.optimizers import Adam, SGD, Momentum
from utils.graphs import plot_loss
from utils.noise import add_noise
from utils.parse_font import parse_font_h


def _plot_grid(X, title, fname=None, shape=(7, 5), n_cols=8):
    n = X.shape[0]
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.4))
    for i in range(n):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(X[i].reshape(shape), cmap="gray_r")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=140)


def _plot_latent(latent, labels, title, fname=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(latent[:, 0], latent[:, 1], c="C0", alpha=0.8)
    for i, (x, y) in enumerate(latent):
        plt.text(x, y, str(labels[i]), fontsize=8, ha="center", va="center")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=140)


def _make_optimizer(cfg):
    name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 0.01)
    if name == "adam":
        return Adam(lr=lr)
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return Momentum(lr=lr, beta=cfg.get("beta", 0.9))
    raise ValueError(f"Optimizer '{name}' no reconocido.")


def _act_map(name: str):
    name = name.lower()
    if name == "tanh":
        return TANH
    if name == "sigmoid":
        return SIGMOID
    if name == "relu":
        return RELU
    raise ValueError(f"Activación '{name}' no soportada.")


def run_experiments():
    """
    Parrilla para estudiar arquitecturas (latente=2) con múltiples capas ocultas
    y distintas activaciones/optimizadores sobre font.h.
    """
    CONFIGS = [
        # TANH/TANH con múltiples capas (datos en [-1,1])
        dict(name="tanh_tanh_h16_adam_lr005_w002", latent_dim=2, hidden=[16], epochs=500, optimizer="adam", lr=0.005, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.02),
        dict(name="tanh_tanh_h16_adam_lr003_w002", latent_dim=2, hidden=[16], epochs=600, optimizer="adam", lr=0.003, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.02),
        dict(name="tanh_tanh_h16_adam_lr005_bs8_w002", latent_dim=2, hidden=[16], epochs=550, optimizer="adam", lr=0.005, batch_size=8, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.02),
        dict(name="tanh_tanh_h24_12_adam_lr003", latent_dim=2, hidden=[24, 12], epochs=1000, optimizer="adam", lr=0.003, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.04),
        dict(name="tanh_tanh_h32_16_adam_lr003", latent_dim=2, hidden=[32, 16], epochs=600, optimizer="adam", lr=0.003, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.03),
        dict(name="tanh_tanh_h32_16_8_adam_lr002", latent_dim=2, hidden=[32, 16, 8], epochs=700, optimizer="adam", lr=0.002, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.02),
        dict(name="tanh_tanh_h20_10_adam_lr005", latent_dim=2, hidden=[20, 10], epochs=500, optimizer="adam", lr=0.005, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.03),
        dict(name="tanh_tanh_h35_20_adam_lr003", latent_dim=2, hidden=[35, 20], epochs=600, optimizer="adam", lr=0.003, batch_size=4, noise=0.0,
             scale="-11", hidden_act="tanh", out_act="tanh", w_init_scale=0.025),
        # ReLU/TANH con 2 capas (escala [-1,1])
        dict(name="relu_tanh_h24_12_sgd_lr02",  latent_dim=2, hidden=[24, 12], epochs=400, optimizer="sgd",  lr=0.02, batch_size=4, noise=0.0,
             scale="-11", hidden_act="relu", out_act="tanh", w_init_scale=0.05),
        dict(name="relu_tanh_h32_16_momentum_lr02",  latent_dim=2, hidden=[32, 16], epochs=500, optimizer="momentum",  lr=0.02, beta=0.9, batch_size=4, noise=0.0,
             scale="-11", hidden_act="relu", out_act="tanh", w_init_scale=0.04),
        # ReLU/SIGMOID con 2 capas (escala [0,1])
        dict(name="relu_sigmoid_h24_12_adam_lr02", latent_dim=2, hidden=[24, 12], epochs=400, optimizer="adam", lr=0.02, batch_size=4, noise=0.0,
             scale="01", hidden_act="relu", out_act="sigmoid", w_init_scale=0.04),
        dict(name="relu_sigmoid_h32_16_momentum_lr015", latent_dim=2, hidden=[32, 16], epochs=500, optimizer="momentum", lr=0.015, beta=0.9, batch_size=4, noise=0.0,
             scale="01", hidden_act="relu", out_act="sigmoid", w_init_scale=0.03),
    ]

    results = []
    os.makedirs("outputs", exist_ok=True)

    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        hidden = cfg.get("hidden", [])
        noise = cfg.get("noise", 0.0)
        scale = cfg.get("scale", "-11")

        X, labels = parse_font_h(scale=scale)
        X_in = add_noise(X, noise_level=noise, seed=42) if noise > 0 else X

        hidden_act = _act_map(cfg.get("hidden_act", "tanh"))
        out_act = _act_map(cfg.get("out_act", "tanh"))

        encoder_acts = [hidden_act] * (len(hidden) + 1)
        decoder_acts = ([hidden_act] * len(hidden)) + [out_act]

        ae = Autoencoder(
            input_dim=X.shape[1],
            latent_dim=cfg["latent_dim"],
            encoder_hidden=hidden,
            decoder_hidden=list(reversed(hidden)),
            encoder_activations=encoder_acts,
            decoder_activations=decoder_acts,
            loss=MSELoss(),
            optimizer=_make_optimizer(cfg),
            w_init_scale=cfg.get("w_init_scale", 0.1),
            seed=42,
        )

        losses = ae.fit(
            X_in, X,
            epochs=cfg["epochs"],
            batch_size=cfg.get("batch_size", 4),
            shuffle=True,
            verbose=False,
        )

        recon = ae.predict(X)
        final_loss = losses[-1] if losses else None
        recon_mse = ae.reconstruct_error(X)
        pix_err = ae.pixel_error(X, threshold=0.0 if scale == "-11" else 0.5).mean()
        latent = ae.get_latent_representation(X)

        print(f"latent={cfg['latent_dim']} hidden={hidden} opt={cfg['optimizer']} lr={cfg.get('lr')} noise={noise} scale={scale}")
        print(f"loss_final={final_loss:.6f}  mse_recon={recon_mse:.6f}  pixel_error_prom={pix_err:.3f}")
        # Sensibilidad a distintos umbrales de binarización para el pixel_error
        if scale == "-11":
            for thr in (-0.25, 0.0, 0.25):
                pe_thr = ae.pixel_error(X, threshold=thr).mean()
                print(f"  pixel_error_prom (thr={thr:+.2f}) = {pe_thr:.3f}")
        else:
            for thr in (0.3, 0.5, 0.7):
                pe_thr = ae.pixel_error(X, threshold=thr).mean()
                print(f"  pixel_error_prom (thr={thr:+.2f}) = {pe_thr:.3f}")
        # Errores por carácter (muestra a muestra)
        per_sample_pix = ae.pixel_error(X, threshold=0.0 if scale == "-11" else 0.5)
        print("  Errores por carácter:")
        for i, err in enumerate(per_sample_pix):
            print(f"    idx {i:2d} -> {err:2d} pix")

        loss_fname = f"loss_{cfg['name']}.png"
        plot_loss(losses, loss_name="MSE", title=f"Loss - {cfg['name']}", fname=loss_fname)
        _plot_grid(recon, f"Reconstruidos - {cfg['name']}", fname=os.path.join("outputs", f"recon_{cfg['name']}.png"))
        _plot_latent(latent, labels, f"Latente (z1,z2) - {cfg['name']}", fname=os.path.join("outputs", f"latent_{cfg['name']}.png"))

        results.append(dict(cfg=cfg, final_loss=final_loss, recon_mse=recon_mse, pixel_err_mean=pix_err))

    print("\n=== Resumen ===")
    for r in results:
        cfg = r["cfg"]
        print(f"{cfg['name']:28s} | latent={cfg['latent_dim']:2d} hidden={cfg.get('hidden', [])} "
              f"opt={cfg['optimizer']:8s} noise={cfg.get('noise',0.0):.1f} "
              f"loss={r['final_loss']:.5f} mse={r['recon_mse']:.5f} pix_err={r['pixel_err_mean']:.3f}")


def run_until_target_error(target_pix_err: float = 1.0, report_every_sec: int = 300):
    """
    Entrena la config base (tanh_tanh_h16_adam_lr005_w002) hasta que el pixel_error medio < target_pix_err.
    Reporta cada report_every_sec segundos.
    """
    cfg = dict(
        name="tanh_tanh_h16_adam_lr005_w002",
        latent_dim=2,
        hidden=[16],
        epochs=50,  # se repite en ciclos
        optimizer="adam",
        lr=0.005,
        batch_size=4,
        noise=0.0,
        scale="-11",
        hidden_act="tanh",
        out_act="tanh",
        w_init_scale=0.02,
    )

    X, labels = parse_font_h(scale=cfg["scale"])
    X_in = add_noise(X, noise_level=cfg["noise"], seed=42) if cfg["noise"] > 0 else X

    hidden_act = _act_map(cfg.get("hidden_act", "tanh"))
    out_act = _act_map(cfg.get("out_act", "tanh"))
    encoder_acts = [hidden_act] * (len(cfg["hidden"]) + 1)
    decoder_acts = ([hidden_act] * len(cfg["hidden"])) + [out_act]

    ae = Autoencoder(
        input_dim=X.shape[1],
        latent_dim=cfg["latent_dim"],
        encoder_hidden=cfg["hidden"],
        decoder_hidden=list(reversed(cfg["hidden"])),
        encoder_activations=encoder_acts,
        decoder_activations=decoder_acts,
        loss=MSELoss(),
        optimizer=_make_optimizer(cfg),
        w_init_scale=cfg.get("w_init_scale", 0.1),
        seed=42,
    )

    start = time.time()
    last_report = start
    total_epochs = 0
    while True:
        losses = ae.fit(
            X_in, X,
            epochs=cfg["epochs"],
            batch_size=cfg.get("batch_size", 4),
            shuffle=True,
            verbose=False,
        )
        total_epochs += cfg["epochs"]
        pix_err = ae.pixel_error(X, threshold=0.0 if cfg["scale"] == "-11" else 0.5).mean()
        now = time.time()
        if now - last_report >= report_every_sec:
            print(f"[{cfg['name']}] epochs={total_epochs} pix_err={pix_err:.3f} time_elapsed={((now-start)/60):.1f} min")
            last_report = now
        if pix_err <= target_pix_err:
            print(f"[{cfg['name']}] Objetivo alcanzado: pix_err={pix_err:.3f} en {total_epochs} epochs ({((now-start)/60):.1f} min)")
            break


if __name__ == "__main__":
    run_experiments()
    print("\n=== Experimento hasta pixel_err < 1 ===")
    run_until_target_error(target_pix_err=1.0, report_every_sec=300)
