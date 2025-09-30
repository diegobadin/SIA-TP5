import sys
from exercises import ex1, ex2

EXPERIMENTS = {
    "exercise1": ex1.run,
    "exercise2": ex2.run,
}

def run_experiment(name: str):
    if name in EXPERIMENTS:
        EXPERIMENTS[name]()
    else:
        print(f"Experiment '{name}' not recognized.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_name> [optional_args...]")
        return

    experiment_name = sys.argv[1]

    if experiment_name == "exercise1":
        EXPERIMENTS["exercise1"]()

    elif experiment_name == "exercise2":
        if len(sys.argv) < 4:
            print("Usage: python main.py ej2 <path_to_csv_file> <model_type>")
            print("       <model_type> options: 'lineal' (Regression) or 'no_lineal' (Classification)")
            return

        csv_file_path = sys.argv[2]
        model_type = sys.argv[3]

        EXPERIMENTS["exercise2"](csv_file_path, model_type)

    else:
        print(f"Experiment '{experiment_name}' not recognized.")

if __name__ == "__main__":
    main()
