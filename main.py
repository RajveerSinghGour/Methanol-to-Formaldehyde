import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from colorama import init, Fore, Style

from objective_function import objective_function, obj_wrapper
from models import hougen_watson, mars_van_krevelen, jiru_model
from dataset import load_data 

# Initialize colorama
init(autoreset=True)

# --- MODEL SELECTION ---
# 'HW'   → Hougen–Watson
# 'MVK'  → Mars–van Krevelen (Arrhenius)
# 'JIRU' → Jiru et al. with Arrhenius k1, k2
MODEL_TO_RUN = 'MVK'

# --- DATASET SELECTION ---
DATASET_TO_USE = 'MATLAB'


def get_model_params(model_name):
    """Defines parameters, bounds, and the function based on the model name."""
    
    if model_name == 'HW':
        print(Fore.CYAN + f"** Running Hougen-Watson Model (3 parameters: b1, b2, b3) **")
        par0 = np.array([0.034616542, 4.354892129, 14.80747964])
        parlb = np.array([0.005, 5.0, 14.80747964])
        parub = np.array([0.05, 7.0, 20.0])
        kinetic_model = hougen_watson
    
    elif model_name == 'MVK':
        print(Fore.CYAN + "** Running Mars-van Krevelen Model with Arrhenius k_r & k_o "
              "(4 parameters: ln(kr0), Ea_r, ln(ko0), Ea_o) **")
        
        par0 = np.array([
            np.log(1e3),   # ln_kr0  ~ 1000
            60000.0,       # Ea_r    ~ 60 kJ/mol
            np.log(1e2),   # ln_ko0  ~ 100
            40000.0        # Ea_o    ~ 40 kJ/mol
        ])

        parlb = np.array([
            np.log(1e-5),  # ln_kr0 lower
            1e4,           # Ea_r lower (10 kJ/mol)
            np.log(1e-5),  # ln_ko0 lower
            1e4            # Ea_o lower
        ])
        parub = np.array([
            np.log(1e10),  # ln_kr0 upper
            2e5,           # Ea_r upper (200 kJ/mol)
            np.log(1e10),  # ln_ko0 upper
            2e5            # Ea_o upper
        ])

        kinetic_model = mars_van_krevelen

    elif model_name == 'JIRU':
        print(Fore.CYAN + "** Running Jiru et al. Model with Arrhenius k1 & k2 "
              "(6 parameters: ln(k10), Ea1, ln(k20), Ea2, m, n) **")

        # Initial guesses – tune if needed
        par0 = np.array([
            np.log(1e3),  # ln_k10
            60000.0,      # Ea1  (~60 kJ/mol)
            np.log(1e2),  # ln_k20
            40000.0,      # Ea2  (~40 kJ/mol)
            1.0,          # m
            1.0           # n
        ])

        parlb = np.array([
            np.log(1e-5),  # ln_k10 lower
            1e4,           # Ea1 lower
            np.log(1e-5),  # ln_k20 lower
            1e4,           # Ea2 lower
            0.0,           # m lower
            0.0            # n lower
        ])
        parub = np.array([
            np.log(1e10),  # ln_k10 upper
            2e5,           # Ea1 upper
            np.log(1e10),  # ln_k20 upper
            2e5,           # Ea2 upper
            3.0,           # m upper
            3.0            # n upper
        ])

        kinetic_model = jiru_model
        
    else:
        raise ValueError("Invalid MODEL_TO_RUN. Choose 'HW', 'MVK', or 'JIRU'.")

    bounds = list(zip(parlb, parub))
    return par0, bounds, kinetic_model


def pretty_header(title):
    line = "=" * 70
    print(Fore.MAGENTA + line)
    print(Fore.MAGENTA + title.center(70))
    print(Fore.MAGENTA + line)


def main():
    pretty_header("PARAMETER ESTIMATION TOOL")
    print(Fore.YELLOW + f"Selected Model  : {MODEL_TO_RUN}")
    print(Fore.YELLOW + f"Data Source     : {DATASET_TO_USE}\n")
    
    # --- 1. Load Data ---
    print(Fore.CYAN + "📥 Loading experimental data ...")
    xexp, rate_exp, temp_exp, matlab_mask = load_data(DATASET_TO_USE)
    
    if len(rate_exp) == 0:
        print(Fore.RED + "❌ No data points were loaded. Check your Excel/CSV file and path.")
        return

    if DATASET_TO_USE == 'ALL' and matlab_mask is not None:
        num_points_matlab = np.sum(matlab_mask)
        num_points_table = np.sum(~matlab_mask)
        print(Fore.GREEN + f"✅ Successfully loaded {len(rate_exp)} total data points: "
              f"{num_points_matlab} (MATLAB) + {num_points_table} (TABLE).")
    else:
        print(Fore.GREEN + f"✅ Successfully loaded {len(rate_exp)} data points from dataset: '{DATASET_TO_USE}'.")
    
    # --- 2. Initial Setup ---
    print()
    print(Fore.CYAN + "🧪 Initializing model parameters and bounds ...")
    par0, bounds, kinetic_model = get_model_params(MODEL_TO_RUN)

    # --- 3. Optimization ---
    print()
    print(Fore.CYAN + "⚙️  Starting optimization (SLSQP, fmincon-like) ...\n")
    
    result = minimize(
        obj_wrapper,
        par0,
        args=(xexp, rate_exp, kinetic_model, temp_exp),
        method='SLSQP',
        bounds=bounds,
        options={'disp': True, 'ftol': 1e-9, 'maxiter': 9000}
    )

    pari = result.x

    # Get fval, predicted rates, and R² from the objective function
    fval, ratec, r2 = objective_function(pari, xexp, rate_exp, kinetic_model, temp_exp)

    # --- 4. Beautiful Results Section ---
    pretty_header("OPTIMIZATION RESULTS")

    print(Fore.YELLOW + "📌 Best-Fit Parameters:\n")
    if MODEL_TO_RUN == 'MVK':
        ln_kr0, Ea_r, ln_ko0, Ea_o = pari
        kr0 = np.exp(ln_kr0)
        ko0 = np.exp(ln_ko0)

        print(Fore.WHITE + f"  ln(kr0)  = {ln_kr0:12.6f}   " + Fore.BLUE + f"→  kr0  = {kr0:12.6e}")
        print(Fore.WHITE + f"  Ea_r     = {Ea_r:12.3f} J/mol")
        print(Fore.WHITE + f"  ln(ko0)  = {ln_ko0:12.6f}   " + Fore.BLUE + f"→  ko0  = {ko0:12.6e}")
        print(Fore.WHITE + f"  Ea_o     = {Ea_o:12.3f} J/mol")

    elif MODEL_TO_RUN == 'HW':
        b1, b2, b3 = pari
        print(Fore.WHITE + f"  b1       = {b1:12.6f}")
        print(Fore.WHITE + f"  b2       = {b2:12.6f}")
        print(Fore.WHITE + f"  b3       = {b3:12.6f}")

    elif MODEL_TO_RUN == 'JIRU':
        ln_k10, Ea1, ln_k20, Ea2, m, n = pari
        k10 = np.exp(ln_k10)
        k20 = np.exp(ln_k20)
        print(Fore.WHITE + f"  ln(k10)  = {ln_k10:12.6f}   " + Fore.BLUE + f"→  k10 = {k10:12.6e}")
        print(Fore.WHITE + f"  Ea1      = {Ea1:12.3f} J/mol")
        print(Fore.WHITE + f"  ln(k20)  = {ln_k20:12.6f}   " + Fore.BLUE + f"→  k20 = {k20:12.6e}")
        print(Fore.WHITE + f"  Ea2      = {Ea2:12.3f} J/mol")
        print(Fore.WHITE + f"  m (MeOH) = {m:12.6f}")
        print(Fore.WHITE + f"  n (O2)   = {n:12.6f}")

    print("\n" + Fore.YELLOW + "📉 Objective Function:")
    print(Fore.WHITE + f"  RMSD Error (fval)  = {fval:10.6f}")
    print(Fore.WHITE + f"  R² (Goodness fit)  = {r2:10.6f}")

    print("\n" + Fore.YELLOW + "📊 Performance Summary:\n")
    print(Fore.WHITE + f"  Total Data Points        : {len(rate_exp)}")
    print(Fore.WHITE + f"  Minimum Rate (calc/exp)  : {min(ratec):.6f} / {min(rate_exp):.6f}")
    print(Fore.WHITE + f"  Maximum Rate (calc/exp)  : {max(ratec):.6f} / {max(rate_exp):.6f}")

    print("\n" + Fore.CYAN + "-" * 70)
    print(Fore.CYAN + "📍 Next Step: Inspect the parity plot to visually judge model quality")
    print(Fore.CYAN + "-" * 70 + "\n")

    # --- 5. Parity Plot ---
    try:
        plt.figure(figsize=(7, 7))
        
        min_val = min(min(ratec), min(rate_exp)) * 0.9
        max_val = max(max(ratec), max(rate_exp)) * 1.1

        if DATASET_TO_USE == 'ALL' and matlab_mask is not None:
            ratec_matlab = ratec[matlab_mask]
            rate_exp_matlab = rate_exp[matlab_mask]
            
            ratec_table = ratec[~matlab_mask]
            rate_exp_table = rate_exp[~matlab_mask]
            
            plt.plot(ratec_matlab, rate_exp_matlab, 'bo', 
                     label=f'MATLAB Data ({len(ratec_matlab)} points)')
            plt.plot(ratec_table, rate_exp_table, 'gs', 
                     label=f'TABLE Data ({len(ratec_table)} points)')
        else:
            plt.plot(ratec, rate_exp, 'bo', label=f'Data Points ({len(rate_exp)})')

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Fit Line')
        
        plt.xlabel('Calculated Rate (ratec)')
        plt.ylabel('Experimental Rate (rate)')
        plt.title(f'Parity Plot: {MODEL_TO_RUN} Model on {DATASET_TO_USE} Data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    except ImportError:
        print(Fore.RED + "\nMatplotlib is not installed. Skipping plot.")


if __name__ == "__main__":
    main()
