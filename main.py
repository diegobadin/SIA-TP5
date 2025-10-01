import sys
from exercises import ex1, ex2, ex3

EXPERIMENTS = {
    "exercise1": ex1.run,
    "exercise2": ex2.run,
    "exercise3": ex3.run,
}

def run_experiment(name: str, *args):
    if name not in EXPERIMENTS:
        print(f"Experiment '{name}' not recognized.")
        return

    # Pasar los argumentos a la función de run si los necesita
    EXPERIMENTS[name](*args)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_name> [optional_args...]")
        return

    experiment_name = sys.argv[1]

    if experiment_name == "exercise1":
        run_experiment("exercise1")

    elif experiment_name == "exercise2":
        if len(sys.argv) < 4:
            print("Usage: python main.py exercise2 <path_to_csv_file> <model_type>")
            print("       <model_type> options: 'lineal' (Regression) or 'no_lineal' (Classification)")
            return
        csv_file_path = sys.argv[2]
        model_type = sys.argv[3]
        run_experiment("exercise2", csv_file_path, model_type)

    elif experiment_name == "exercise3":
        # Por ahora no necesita argumentos
        run_experiment("exercise3")

    else:
        print(f"Experiment '{experiment_name}' not recognized.")


if __name__ == "__main__":
    main()
