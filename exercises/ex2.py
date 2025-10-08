import os

import numpy as np

from src.perceptron import Perceptron
from src.training_algorithms import linear_perceptron, nonlinear_perceptron
from utils.activations import (
    linear_function,
    scaled_logistic_function_factory,
    scaled_tanh_function_factory, tanh_function_factory,
)
from utils.experiments import run_experimento1_baseline_lineal, run_ex2_experimento4_cv_boxplot
from utils.graphs import plot_accuracy_folds, plot_predictions_vs_real, plot_absolute_error
from utils.metrics import cross_validate_regression, sse_score
from utils.parse_csv_data import parse_csv_data
import matplotlib.pyplot as plt

from utils.splits import StratifiedKFold, KFold, HoldoutSplit


def run_linear_perceptron_regression(csv_file_path: str):
    """Runs the Delta Rule for Linear Regression."""

    # 1. DATA LOADING AND PREPARATION
    # Load feature (X) and target (Y) data from the specified CSV file.
    x_features, y_labels = parse_csv_data(csv_file_path)
    if not x_features: return

    # Prepend the bias term (1) to each feature vector (x0 = 1).
    x_with_bias = [[1] + row for row in x_features]
    # Determine the number of input features (excluding the bias).
    n_inputs = len(x_features[0])

    # 2. PERCEPTRON INITIALIZATION
    # Instantiate the Perceptron model with hyperparameters.
    p = Perceptron(
        n_inputs=n_inputs,
        activation=linear_function,
        alpha=0.005,
        max_iter=100,
        training_algorithm=linear_perceptron,
        mode="online", # If mode is online, batch_size is ignored
        batch_size=8,
        shuffle=True
    )

    # 3. CONSOLE OUTPUT (HEADER)
    print("==================================================")
    print("  LINEAR PERCEPTRON (REGRESSION - DELTA RULE)")
    print(f"  {len(x_features)} SAMPLES LOADED FROM: {os.path.basename(csv_file_path)}")
    print("==================================================")

    # 4. TRAINING
    # Execute the training algorithm (Delta Rule) to find optimal weights.
    p.fit(x_with_bias, y_labels)

    # 5. RESULTS SUMMARY
    # Display the final weights found by the training algorithm.
    formatted_weights = [f"{w:.3f}" for w in p.weights]
    print(f"Final Weights (B, w1, w2, w3): {formatted_weights}")

    # Display the final error metric (Sum of Squared Errors) achieved on the training set.
    print(f"Final SSE (Sum of Squared Errors): {p.min_error:.4f}")

    # 6. DETAILED PREDICTION TABLE
    print("\n| Input (x1, x2, x3) | Expected (y) | Predicted (ŷ) | Absolute Error |")
    print("|--------------------|--------------|---------------|----------------|")

    total_abs_error = 0
    # Iterate over the training samples to show individual predictions.
    for x_full, y_expected in zip(x_with_bias, y_labels):
        y_pred = p.predict(x_full)
        abs_error_val = abs(y_expected - y_pred)
        total_abs_error += abs_error_val
        input_features = x_full[1:]
        input_str = f"[{input_features[0]:.1f}, {input_features[1]:.1f}, {input_features[2]:.1f}]"
        print(f"| {input_str:<18} | {y_expected:^12.3f} | {y_pred:^13.3f} | {abs_error_val:^14.3f} |")

    # Graph the training error over epochs
    plt.plot(p.error_history)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(f"Training Error ({p.mode})")
    plt.show()

def run_nonlinear_perceptron(csv_file_path: str, activation: str = "tanh", beta: float = 1.0):
    """
    Runs the non-linear perceptron (Delta Rule with sigmoid/tanh activation).
    activation: "tanh" maps to θ(h)=tanh(βh), "logistic" maps to θ(h)=1/(1+exp(-2βh)).
    """
    x_features, y_labels = parse_csv_data(csv_file_path)
    if not x_features: return

    # Standardize features to zero-mean / unit-variance per column (using utils.stats)
    from utils.stats import fmean, pstdev
    cols = list(zip(*x_features))
    means = [fmean(col) for col in cols]
    stdevs = [max(1e-8, pstdev(col)) for col in cols]
    x_std = [[(val - m)/s for val, m, s in zip(row, means, stdevs)] for row in x_features]

    x_with_bias = [[1] + row for row in x_std]
    n_inputs = len(x_std[0])

    # Build a scaled activation directly to map to [y_min, y_max]
    y_min = min(y_labels)
    y_max = max(y_labels)
    if activation == "tanh":
        act, dact = scaled_tanh_function_factory(beta, y_min, y_max)
        act_name = "tanh[scaled]"
    else:
        act, dact = scaled_logistic_function_factory(beta, y_min, y_max)
        act_name = "logistic[scaled]"

    # Adjust learning rate to compensate for scaled activation range
    base_alpha = 0.005
    range_span = max(1e-8, y_max - y_min)
    effective_alpha = base_alpha / range_span

    p = Perceptron(
        n_inputs=n_inputs,
        activation=act,
        activation_derivative=dact,
        alpha=effective_alpha,
        max_iter=500,
        training_algorithm=nonlinear_perceptron,
        mode="online",  # If mode is online, batch_size is ignored
        batch_size=8,
        shuffle=True
    )

    print("==================================================")
    print(f"  NON-LINEAR PERCEPTRON ({act_name.upper()} - β={beta})")
    print(f"  {len(x_features)} SAMPLES LOADED FROM: {os.path.basename(csv_file_path)}")
    print("==================================================")

    # Train using original y (activation already outputs in [y_min,y_max])
    p.fit(x_with_bias, y_labels)

    formatted_weights = [f"{w:.3f}" for w in p.weights]
    print(f"Final Weights (B, w1, w2, w3): {formatted_weights}")
    print(f"Final SSE (Sum of Squared Errors): {p.min_error:.4f}")

    print("\n| Input (x1, x2, x3) | Expected (y) | Predicted (ŷ) | Absolute Error |")
    print("|--------------------|--------------|---------------|----------------|")
    for x_full, y_expected in zip(x_with_bias, y_labels):
        y_pred = p.predict(x_full)
        abs_error_val = abs(y_expected - y_pred)
        input_features = x_full[1:]
        input_str = f"[{input_features[0]:.1f}, {input_features[1]:.1f}, {input_features[2]:.1f}]"
        print(f"| {input_str:<18} | {y_expected:^12.3f} | {y_pred:^13.3f} | {abs_error_val:^14.3f} |")

    # Graph the training error over epochs
    plt.plot(p.error_history)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(f"Training Error ({p.mode})")
    plt.show()


def run_perceptron_experiment(csv_file_path: str, lr: float = 0.01, epochs: int = 100, beta: float = 1.0):

    # --- Cargar datos ---
    X, Y = parse_csv_data(csv_file_path)
    if not X:
        return
    X = np.array(X)
    Y = np.array(Y)

    # --- Agregar bias ---
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    n_inputs = X_bias.shape[1]

    # --- Definir splitter (solo KFold) ---
    splitter = KFold(n_splits=5, shuffle=True, seed=42)

    # --- Definir factories de perceptrones ---
    linear_factory = lambda: Perceptron(
        n_inputs=n_inputs,
        activation=linear_function,
        alpha=lr,
        max_iter=epochs,
        training_algorithm=linear_perceptron,
    )

    non_linear_factory = lambda: Perceptron(
        n_inputs=n_inputs,
        activation=lambda h: np.tanh(beta * h),
        activation_derivative=lambda h: 1 - np.tanh(beta * h) ** 2,
        alpha=lr,
        max_iter=epochs,
        training_algorithm=nonlinear_perceptron,
    )

    results = {}

    # --- Cross-validate (MSE por fold) ---
    mse_lin_folds = cross_validate_regression(linear_factory, X_bias, Y, splitter, scoring=sse_score)
    mse_nonlin_folds = cross_validate_regression(non_linear_factory, X_bias, Y, splitter, scoring=sse_score)

    results['Lineal'] = mse_lin_folds
    results['No Lineal'] = mse_nonlin_folds

    # --- Graficar MSE por fold ---
    plot_accuracy_folds(results, title="MSE por fold (KFold)")

    # --- Entrenar ambos modelos con todo el dataset para ver predicciones ---
    model_lin_full = linear_factory()
    model_lin_full.fit(X_bias, Y)
    y_pred_lin_full = np.array([model_lin_full.predict(x) for x in X_bias])

    model_nonlin_full = non_linear_factory()
    model_nonlin_full.fit(X_bias, Y)
    y_pred_nonlin_full = np.array([model_nonlin_full.predict(x) for x in X_bias])

    y_pred_dict_full = {'Lineal': y_pred_lin_full, 'No Lineal': y_pred_nonlin_full}

    # --- Graficar predicciones vs reales y errores absolutos ---
    plot_predictions_vs_real(Y, y_pred_dict_full, title="Predicciones vs Reales (Todo el dataset)")
    plot_absolute_error(Y, y_pred_dict_full, title="Error absoluto por muestra (Todo el dataset)")

    # --- Mostrar resumen de MSE ---
    for model_name, vals in results.items():
        print(f"{model_name}: mean MSE={np.mean(vals):.3f}, std MSE={np.std(vals):.3f}")

    return results


def run_activation_comparison_experiment(csv_file_path: str, lr: float = 0.01, epochs: int = 20, beta: float = 1.0):
    """
    Experimento: Comparación de funciones de activación escaladas
    Compara sigmoid escalada vs tanh escalada para evaluar impacto en convergencia y error.
    """
    print("=" * 60)
    print("  COMPARACIÓN: SIGMOID ESCALADA vs TANH ESCALADA")
    print("=" * 60)
    print(f"Hiperparámetros: η={lr}, Épocas={epochs}, β={beta}")

    # --- Cargar datos ---
    if csv_file_path.endswith('.txt'):
        from utils.parse_csv_data import parse_digits_7x5_txt
        X, Y = parse_digits_7x5_txt(csv_file_path)
        print(f"Archivo de dígitos detectado: {len(X)} dígitos de {X.shape[1]} píxeles")
    else:
        X, Y = parse_csv_data(csv_file_path)
        if not X:
            return
        X = np.array(X)
        Y = np.array(Y)

    print(f"Datos: {len(X)} muestras, {X.shape[1]} características")

    # --- Normalizar características ---
    from utils.stats import fmean, pstdev
    cols = list(zip(*X))
    means = [fmean(col) for col in cols]
    stdevs = [max(1e-8, pstdev(col)) for col in cols]
    X_normalized = np.array([[(val - m)/s for val, m, s in zip(row, means, stdevs)] for row in X])

    # --- Agregar bias ---
    X_bias = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])
    n_inputs = X_bias.shape[1]

    # --- Definir splitter ---
    splitter = KFold(n_splits=5, shuffle=True, seed=42)

    # --- Factories de perceptrones ---
    def sigmoid_factory():
        y_min, y_max = Y.min(), Y.max()
        sigmoid_act, sigmoid_deriv = scaled_logistic_function_factory(beta, y_min, y_max)
        range_span = max(1e-8, y_max - y_min)
        effective_alpha = lr / range_span
        
        return Perceptron(
            n_inputs=n_inputs,
            activation=sigmoid_act,
            activation_derivative=sigmoid_deriv,
            alpha=effective_alpha,
            max_iter=epochs,
            training_algorithm=nonlinear_perceptron,
            mode="online"
        )

    def tanh_factory():
        y_min, y_max = Y.min(), Y.max()
        tanh_act, tanh_deriv = scaled_tanh_function_factory(beta, y_min, y_max)
        range_span = max(1e-8, y_max - y_min)
        effective_alpha = lr / range_span
        
        return Perceptron(
            n_inputs=n_inputs,
            activation=tanh_act,
            activation_derivative=tanh_deriv,
            alpha=effective_alpha,
            max_iter=epochs,
            training_algorithm=nonlinear_perceptron,
            mode="online"
        )

    # --- Validación cruzada ---
    print(f"\nEjecutando validación cruzada...")
    mse_sigmoid_folds = cross_validate_regression(sigmoid_factory, X_bias, Y, splitter, scoring=sse_score)
    mse_tanh_folds = cross_validate_regression(tanh_factory, X_bias, Y, splitter, scoring=sse_score)

    # Convertir SSE a MSE
    mse_sigmoid_folds = [sse / len(Y) for sse in mse_sigmoid_folds]
    mse_tanh_folds = [sse / len(Y) for sse in mse_tanh_folds]

    # --- Entrenar modelos completos para obtener curvas de entrenamiento ---
    print(f"\nEntrenando modelos completos...")
    model_sigmoid_full = sigmoid_factory()
    model_sigmoid_full.fit(X_bias, Y)
    
    model_tanh_full = tanh_factory()
    model_tanh_full.fit(X_bias, Y)

    # === GRÁFICO: MSE vs Épocas ===
    plt.figure(figsize=(12, 8))
    
    # Verificar si tienen historial de MSE
    if hasattr(model_sigmoid_full, 'mse_history') and model_sigmoid_full.mse_history:
        plt.plot(range(1, len(model_sigmoid_full.mse_history) + 1), model_sigmoid_full.mse_history, 
                label='Sigmoid Escalada', linewidth=3, color='#ff7f0e', marker='o', markersize=4)
    else:
        # Si no hay mse_history, usar error_history
        plt.plot(range(1, len(model_sigmoid_full.error_history) + 1), model_sigmoid_full.error_history, 
                label='Sigmoid Escalada', linewidth=3, color='#ff7f0e', marker='o', markersize=4)
    
    if hasattr(model_tanh_full, 'mse_history') and model_tanh_full.mse_history:
        plt.plot(range(1, len(model_tanh_full.mse_history) + 1), model_tanh_full.mse_history, 
                label='Tanh Escalada', linewidth=3, color='#2ca02c', marker='s', markersize=4)
    else:
        # Si no hay mse_history, usar error_history
        plt.plot(range(1, len(model_tanh_full.error_history) + 1), model_tanh_full.error_history, 
                label='Tanh Escalada', linewidth=3, color='#2ca02c', marker='s', markersize=4)
    
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.title('MSE vs Épocas - Comparación de Funciones de Activación Escaladas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Configurar eje X para mostrar solo valores enteros
    max_epochs = max(len(model_sigmoid_full.error_history), len(model_tanh_full.error_history))
    plt.xticks(range(1, max_epochs + 1, max(1, max_epochs // 10)))
    
    plt.tight_layout()
    plt.show(block=True)

    # --- Métricas finales ---
    y_pred_sigmoid = np.array([model_sigmoid_full.predict(x) for x in X_bias])
    y_pred_tanh = np.array([model_tanh_full.predict(x) for x in X_bias])
    
    mse_sigmoid_final = np.mean((Y - y_pred_sigmoid) ** 2)
    mse_tanh_final = np.mean((Y - y_pred_tanh) ** 2)

    print(f"\n" + "="*50)
    print("RESULTADOS:")
    print("="*50)
    print(f"MSE Final:")
    print(f"  Sigmoid Escalada:     {mse_sigmoid_final:.6f}")
    print(f"  Tanh Escalada:       {mse_tanh_final:.6f}")
    
    print(f"\nValidación Cruzada:")
    print(f"  Sigmoid:     {np.mean(mse_sigmoid_folds):.6f} ± {np.std(mse_sigmoid_folds):.6f}")
    print(f"  Tanh:        {np.mean(mse_tanh_folds):.6f} ± {np.std(mse_tanh_folds):.6f}")

    return {
        'sigmoid_mse': mse_sigmoid_final,
        'tanh_mse': mse_tanh_final,
        'cv_results': {
            'Sigmoid Escalada': mse_sigmoid_folds,
            'Tanh Escalada': mse_tanh_folds
        }
    }


def run_advanced_comparison_analysis(csv_file_path: str, lr: float = 0.01, epochs: int = 20, beta: float = 1.0):
    """
    Análisis comparativo: perceptrón lineal vs no lineal con sigmoid.
    Genera solo 3 gráficos específicos usando las utilidades de utils/.
    """
    print("=" * 60)
    print("  COMPARACIÓN: PERCEPTRÓN LINEAL vs NO LINEAL (SIGMOID)")
    print("=" * 60)
    print(f"Hiperparámetros: η={lr}, Épocas={epochs}, β={beta}")

    # --- Cargar datos ---
    if csv_file_path.endswith('.txt'):
        from utils.parse_csv_data import parse_digits_7x5_txt
        X, Y = parse_digits_7x5_txt(csv_file_path)
        print(f"Archivo de dígitos detectado: {len(X)} dígitos de {X.shape[1]} píxeles")
    else:
        X, Y = parse_csv_data(csv_file_path)
        if not X:
            return
        X = np.array(X)
        Y = np.array(Y)

    print(f"Datos: {len(X)} muestras, {X.shape[1]} características")

    # --- Normalizar características ---
    from utils.stats import fmean, pstdev
    cols = list(zip(*X))
    means = [fmean(col) for col in cols]
    stdevs = [max(1e-8, pstdev(col)) for col in cols]
    X_normalized = np.array([[(val - m)/s for val, m, s in zip(row, means, stdevs)] for row in X])

    # --- Agregar bias ---
    X_bias = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])
    n_inputs = X_bias.shape[1]

    # --- Definir splitter ---
    splitter = KFold(n_splits=5, shuffle=True, seed=42)

    # --- Factories de perceptrones ---
    def linear_factory():
        return Perceptron(
            n_inputs=n_inputs,
            activation=linear_function,
            alpha=lr,
            max_iter=epochs,
            training_algorithm=linear_perceptron,
            mode="online"
        )

    def nonlinear_factory():
        y_min, y_max = Y.min(), Y.max()
        sigmoid_act, sigmoid_deriv = scaled_logistic_function_factory(beta, y_min, y_max)
        range_span = max(1e-8, y_max - y_min)
        effective_alpha = lr / range_span
        
        return Perceptron(
            n_inputs=n_inputs,
            activation=sigmoid_act,
            activation_derivative=sigmoid_deriv,
            alpha=effective_alpha,
            max_iter=epochs,
            training_algorithm=nonlinear_perceptron,
            mode="online"
        )

    # --- Validación cruzada ---
    print(f"\nEjecutando validación cruzada...")
    mse_lin_folds = cross_validate_regression(linear_factory, X_bias, Y, splitter, scoring=sse_score)
    mse_nonlin_folds = cross_validate_regression(nonlinear_factory, X_bias, Y, splitter, scoring=sse_score)

    # Convertir SSE a MSE
    mse_lin_folds = [sse / len(Y) for sse in mse_lin_folds]
    mse_nonlin_folds = [sse / len(Y) for sse in mse_nonlin_folds]

    # --- Entrenar modelos completos ---
    print(f"\nEntrenando modelos completos...")
    model_lin_full = linear_factory()
    model_lin_full.fit(X_bias, Y)
    y_pred_lin_full = np.array([model_lin_full.predict(x) for x in X_bias])

    model_nonlin_full = nonlinear_factory()
    model_nonlin_full.fit(X_bias, Y)
    y_pred_nonlin_full = np.array([model_nonlin_full.predict(x) for x in X_bias])

    # === 1️⃣ GRÁFICO 1: Mejora Porcentual ===
    plt.figure(figsize=(10, 6))
    
    # Calcular mejora porcentual
    mse_linear_mean = np.mean(mse_lin_folds)
    mse_nonlinear_mean = np.mean(mse_nonlin_folds)
    mejora_porcentual = ((mse_linear_mean - mse_nonlinear_mean) / mse_linear_mean) * 100
    
    # Crear gráfico de barras con mejora
    labels = ['Perceptrón Lineal', 'Perceptrón No Lineal']
    means = [mse_linear_mean, mse_nonlinear_mean]
    stds = [np.std(mse_lin_folds), np.std(mse_nonlin_folds)]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('MSE (Mean Squared Error)')
    plt.title('Comparación MSE: Lineal vs No Lineal')
    plt.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Agregar texto de mejora destacado
    plt.text(0.5, 0.95, f'El perceptrón no lineal reduce el error en un {mejora_porcentual:.1f}% respecto al lineal',
             transform=plt.gca().transAxes, ha='center', va='top', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=True)

    # === 2️⃣ GRÁFICO 2: Curvas de Entrenamiento (MSE vs Épocas) ===
    if hasattr(model_lin_full, 'mse_history') and model_lin_full.mse_history:
        plt.figure(figsize=(10, 6))
        
        # Solo gráfico completo sin zoom
        plt.plot(range(1, len(model_lin_full.mse_history) + 1), model_lin_full.mse_history, 
                label='Perceptrón Lineal', linewidth=3, color='#ff7f0e', marker='o', markersize=4)
        plt.plot(range(1, len(model_nonlin_full.mse_history) + 1), model_nonlin_full.mse_history, 
                label='Perceptrón No Lineal', linewidth=3, color='#2ca02c', marker='s', markersize=4)
        
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.title('MSE vs Épocas - Comparación de Convergencia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Configurar eje X para mostrar solo valores enteros
        plt.xticks(range(1, len(model_lin_full.mse_history) + 1, max(1, len(model_lin_full.mse_history) // 10)))
        
        plt.tight_layout()
        plt.show(block=True)
        
        # Información adicional en consola
        print(f"\n📊 INFORMACIÓN DE CONVERGENCIA:")
        print(f"Perceptrón Lineal:")
        print(f"  MSE inicial: {model_lin_full.mse_history[0]:.6f}")
        print(f"  MSE final:   {model_lin_full.mse_history[-1]:.6f}")
        print(f"  Reducción:  {((model_lin_full.mse_history[0] - model_lin_full.mse_history[-1]) / model_lin_full.mse_history[0] * 100):.2f}%")
        
        print(f"\nPerceptrón No Lineal:")
        print(f"  MSE inicial: {model_nonlin_full.mse_history[0]:.6f}")
        print(f"  MSE final:   {model_nonlin_full.mse_history[-1]:.6f}")
        print(f"  Reducción:  {((model_nonlin_full.mse_history[0] - model_nonlin_full.mse_history[-1]) / model_nonlin_full.mse_history[0] * 100):.2f}%")
        
        # Verificar si están solapadas
        diff_final = abs(model_lin_full.mse_history[-1] - model_nonlin_full.mse_history[-1])
        if diff_final < 1e-6:
            print(f"\n⚠️  ADVERTENCIA: Las curvas están prácticamente solapadas (diferencia final: {diff_final:.2e})")
        else:
            print(f"\n✅ Las curvas son distinguibles (diferencia final: {diff_final:.6f})")

    # === 3️⃣ GRÁFICO 3: Predicciones vs Valores Reales ===
    plt.figure(figsize=(8, 8))
    plt.scatter(Y, y_pred_lin_full, alpha=0.7, color='#ff7f0e', label='Perceptrón Lineal', s=60, edgecolors='black', linewidth=0.5)
    plt.scatter(Y, y_pred_nonlin_full, alpha=0.7, color='#2ca02c', label='Perceptrón No Lineal', s=60, edgecolors='black', linewidth=0.5)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', label='y=x (Predicción Perfecta)', linewidth=3)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

    # --- Métricas finales ---
    mse_linear_final = np.mean((Y - y_pred_lin_full) ** 2)
    mse_nonlinear_final = np.mean((Y - y_pred_nonlin_full) ** 2)
    improvement = ((mse_linear_final - mse_nonlinear_final) / mse_linear_final) * 100

    print(f"\n" + "="*50)
    print("RESULTADOS:")
    print("="*50)
    print(f"MSE Final:")
    print(f"  Perceptrón Lineal:     {mse_linear_final:.6f}")
    print(f"  Perceptrón No Lineal: {mse_nonlinear_final:.6f}")
    print(f"  Mejora:               {improvement:+.2f}%")
    
    print(f"\nValidación Cruzada:")
    print(f"  Lineal:     {np.mean(mse_lin_folds):.6f} ± {np.std(mse_lin_folds):.6f}")
    print(f"  No Lineal:  {np.mean(mse_nonlin_folds):.6f} ± {np.std(mse_nonlin_folds):.6f}")

    return {
        'linear_mse': mse_linear_final,
        'nonlinear_mse': mse_nonlinear_final,
        'improvement': improvement,
        'cv_results': {
            'Perceptrón Lineal': mse_lin_folds,
            'Perceptrón No Lineal': mse_nonlin_folds
        }
    }


def run(csv_file_path: str, model_type: str):
    """Maps the model type to the corresponding runner function."""
    model_type = model_type.lower()
    if model_type == "lineal":
        run_linear_perceptron_regression(csv_file_path)
    elif model_type == "no_lineal":
        run_nonlinear_perceptron(csv_file_path)
    elif model_type == "experiment":
        run_perceptron_experiment(csv_file_path)
    elif model_type == "advanced":
        run_advanced_comparison_analysis(csv_file_path)
    elif model_type == "activation_comparison":
        run_activation_comparison_experiment(csv_file_path)
    elif model_type in ("baseline_lineal", "exp1"):
        run_experimento1_baseline_lineal(csv_file_path, lr=0.01, epochs=50)
    elif model_type in ("boxplot_cv", "exp4"):
        run_ex2_experimento4_cv_boxplot(csv_file_path,
                                        k=5, eta=0.01, epochs=200,
                                        beta=1.0,seed= 42)

    else:
        print(f"Error: Model type '{model_type}' not recognized. Use 'lineal', 'no_lineal', 'experiment', 'advanced', or 'activation_comparison'.")

