import sys
from exercises import ej1, ej1b_denoising

EXPERIMENTS = {
    "ex1": ej1.run,
    "denoising": ej1b_denoising.run
}


"""
Usage:
  python main.py ex1 <latent_dim?> <epochs?> <noise_level?> <deep?> <batch_size?> <lr?> <scale?>
  python main.py denoising
"""

def run_experiment(name: str, *args):
    if name not in EXPERIMENTS:
        print(f"Experiment '{name}' not recognized.")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")
        return

    # Pasar los argumentos a la función de run si los necesita
    EXPERIMENTS[name](*args)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ex1 <latent_dim?> <epochs?> <noise_level?> <deep?> <batch_size?> <lr?> <scale?>")
        print("  python main.py denoising")
        return

    experiment_name = sys.argv[1]

    if experiment_name == "ex1":
        latent = int(sys.argv[2]) if len(sys.argv) >= 3 else 2
        epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
        noise = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0
        deep = sys.argv[5].lower() == "true" if len(sys.argv) >= 6 else False
        batch_size = int(sys.argv[6]) if len(sys.argv) >= 7 else 4
        lr = float(sys.argv[7]) if len(sys.argv) >= 8 else 0.01
        scale = sys.argv[8] if len(sys.argv) >= 9 else "-11"
        run_experiment("ex1", latent, epochs, noise, deep, batch_size, lr, scale)
    elif experiment_name == "denoising":
        run_experiment("denoising")
    else:
        print(f"Experiment '{experiment_name}' not recognized.")
        print(f"Available experiments: {list(EXPERIMENTS.keys())}")


if __name__ == "__main__":
    main()
