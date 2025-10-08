import sys
from exercises import ex1, ex2, ex3, ex4

EXPERIMENTS = {
    "ex1": ex1.run,
    "ex2": ex2.run,
    "ex3": ex3.run,
    "ex4": ex4.run,
}


"""
python main.py ex1 

python main.py ex2 <path_to_csv_file> <model_type>

python main.py ex3 xor

python main.py ex3 parity

python main.py ex3 digits

python main.py ex4 MNIST

"""

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

    if experiment_name == "ex1":
        run_experiment("ex1")

    elif experiment_name == "ex2":
        if len(sys.argv) < 4:
            print("Usage: python main.py ex2 <path_to_csv_file> <model_type>")
            print("       <model_type> options: 'lineal' (Regression) or 'no_lineal' (Classification)")
            return
        csv_file_path = sys.argv[2]
        model_type = sys.argv[3]
        run_experiment("ex2", csv_file_path, model_type)

    elif experiment_name == "ex3":
        if len(sys.argv) < 3:
            print("Usage: python main.py ex3 <item>")
            print("       <item> options: xor | parity | digits")
            return
        item = sys.argv[2].lower()
        if item not in ("xor", "parity", "digits", "architecture_comparison"):
            print("Invalid item. Use: xor | parity | digits | architecture_comparison")
            return
        run_experiment("ex3",item)

    elif experiment_name == "ex4":
        run_experiment("ex4")

    else:
        print(f"Experiment '{experiment_name}' not recognized.")


if __name__ == "__main__":
    main()
