import sys
from exercises import ex5_autoencoder

EXPERIMENTS = {"ex5": ex5_autoencoder.run}


"""
python main.py ex5 <latent_dim?> <epochs?> <noise?>
"""

def run_experiment(name: str, *args):
    if name not in EXPERIMENTS:
        print(f"Experiment '{name}' not recognized.")
        return

    # Pasar los argumentos a la función de run si los necesita
    EXPERIMENTS[name](*args)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py ex5 <latent_dim?> <epochs?> <noise?>")
        return

    experiment_name = sys.argv[1]

    if experiment_name == "ex5":
        latent = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
        epochs = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
        noise = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0
        run_experiment("ex5", latent, epochs, noise)
    else:
        print(f"Experiment '{experiment_name}' not recognized.")


if __name__ == "__main__":
    main()
