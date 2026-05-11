import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from colorama import init, Fore, Style

from objective_function import objective_function, obj_wrapper
from models import hougen_watson, mars_van_krevelen, jiru_model, langmuir_hinshelwood, lhhw_formaldehyde
from dataset import load_data

# Initialize colorama
init(autoreset=True)

# --- MODEL SELECTION ---
# 'HW'   → Hougen–Watson
# 'MVK'  → Mars–van Krevelen (Arrhenius)
# 'JIRU' → Jiru et al.
# 'LH'   → Langmuir–Hinshelwood
# 'LHHW' → Advanced formaldehyde model
MODEL_TO_RUN = 'LHHW'

# --- DATASET SELECTION ---
DATASET_TO_USE = 'MATLAB'


def get_model_params(model_name):

    if model_name == 'HW':
        print(Fore.CYAN + f"** Running Hougen-Watson Model (3 parameters: b1, b2, b3) **")

        par0 = np.array([0.034616542, 4.354892129, 14.80747964])
        parlb = np.array([0.005, 5.0, 14.80747964])
        parub = np.array([0.05, 7.0, 20.0])

        kinetic_model = hougen_watson

    elif model_name == 'MVK':
        print(Fore.CYAN + "** Running Mars-van Krevelen Model with Arrhenius k_r & k_o **")

        par0 = np.array([
            np.log(1e3),
            60000.0,
            np.log(1e2),
            40000.0
        ])

        parlb = np.array([
            np.log(1e-5),
            1e4,
            np.log(1e-5),
            1e4
        ])

        parub = np.array([
            np.log(1e10),
            2e5,
            np.log(1e10),
            2e5
        ])

        kinetic_model = mars_van_krevelen

    elif model_name == 'JIRU':
        print(Fore.CYAN + "** Running Jiru Model with Arrhenius k1 & k2 **")

        par0 = np.array([
            np.log(1e3),
            60000.0,
            np.log(1e2),
            40000.0,
            1.0,
            1.0
        ])

        parlb = np.array([
            np.log(1e-5),
            1e4,
            np.log(1e-5),
            1e4,
            0.0,
            0.0
        ])

        parub = np.array([
            np.log(1e10),
            2e5,
            np.log(1e10),
            2e5,
            3.0,
            3.0
        ])

        kinetic_model = jiru_model

    elif model_name == 'LH':
        print(Fore.CYAN + "** Running Langmuir–Hinshelwood Model **")

        par0 = np.array([
            np.log(1e3),
            60000.0,
            1.0,
            1.0
        ])

        parlb = np.array([
            np.log(1e-5),
            1e4,
            0.0,
            0.0
        ])

        parub = np.array([
            np.log(1e10),
            2e5,
            50.0,
            50.0
        ])

        kinetic_model = langmuir_hinshelwood

    # ⭐ NEW MODEL
    elif model_name == 'LHHW':
        print(Fore.CYAN + "** Running LHHW Formaldehyde Model (6 parameters) **")

        par0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        parlb = np.array([1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
        parub = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        kinetic_model = lhhw_formaldehyde

    else:
        raise ValueError("Invalid MODEL_TO_RUN. Choose 'HW', 'MVK', 'JIRU', 'LH', or 'LHHW'.")

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

    # --- Load Data ---
    print(Fore.CYAN + "📥 Loading experimental data ...")
    xexp, rate_exp, temp_exp, matlab_mask = load_data(DATASET_TO_USE)

    if len(rate_exp) == 0:
        print(Fore.RED + "❌ No data points were loaded.")
        return

    print(Fore.GREEN + f"✅ Successfully loaded {len(rate_exp)} data points.")

    # --- Setup ---
    print()
    print(Fore.CYAN + "🧪 Initializing model parameters and bounds ...")
    par0, bounds, kinetic_model = get_model_params(MODEL_TO_RUN)

    # --- Optimization ---
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

    fval, ratec, r2 = objective_function(pari, xexp, rate_exp, kinetic_model, temp_exp)

    # --- Results ---
    pretty_header("OPTIMIZATION RESULTS")

    print(Fore.YELLOW + "📌 Best-Fit Parameters:\n")

    if MODEL_TO_RUN == 'MVK':
        ln_kr0, Ea_r, ln_ko0, Ea_o = pari
        kr0 = np.exp(ln_kr0)
        ko0 = np.exp(ln_ko0)

        print(Fore.WHITE + f"  ln(kr0)  = {ln_kr0:12.6f} → kr0 = {kr0:12.6e}")
        print(Fore.WHITE + f"  Ea_r     = {Ea_r:12.3f} J/mol")
        print(Fore.WHITE + f"  ln(ko0)  = {ln_ko0:12.6f} → ko0 = {ko0:12.6e}")
        print(Fore.WHITE + f"  Ea_o     = {Ea_o:12.3f} J/mol")

    elif MODEL_TO_RUN == 'HW':
        b1, b2, b3 = pari
        print(Fore.WHITE + f"  b1 = {b1:12.6f}")
        print(Fore.WHITE + f"  b2 = {b2:12.6f}")
        print(Fore.WHITE + f"  b3 = {b3:12.6f}")

    elif MODEL_TO_RUN == 'JIRU':
        ln_k10, Ea1, ln_k20, Ea2, m, n = pari
        k10 = np.exp(ln_k10)
        k20 = np.exp(ln_k20)

        print(Fore.WHITE + f"  ln(k10) = {ln_k10:12.6f} → k10 = {k10:12.6e}")
        print(Fore.WHITE + f"  Ea1     = {Ea1:12.3f} J/mol")
        print(Fore.WHITE + f"  ln(k20) = {ln_k20:12.6f} → k20 = {k20:12.6e}")
        print(Fore.WHITE + f"  Ea2     = {Ea2:12.3f} J/mol")
        print(Fore.WHITE + f"  m       = {m:12.6f}")
        print(Fore.WHITE + f"  n       = {n:12.6f}")

    elif MODEL_TO_RUN == 'LH':
        ln_k0, Ea, KMeOH, KO2 = pari
        k0 = np.exp(ln_k0)

        print(Fore.WHITE + f"  ln(k0) = {ln_k0:12.6f} → k0 = {k0:12.6e}")
        print(Fore.WHITE + f"  Ea     = {Ea:12.3f} J/mol")
        print(Fore.WHITE + f"  KMeOH  = {KMeOH:12.6f}")
        print(Fore.WHITE + f"  KO2    = {KO2:12.6f}")

    elif MODEL_TO_RUN == 'LHHW':
        kf, K1, K2, K5, K6, K7 = pari

        print(Fore.WHITE + f"  kf  = {kf:12.6f}")
        print(Fore.WHITE + f"  K1  = {K1:12.6f}")
        print(Fore.WHITE + f"  K2  = {K2:12.6f}")
        print(Fore.WHITE + f"  K5  = {K5:12.6f}")
        print(Fore.WHITE + f"  K6  = {K6:12.6f}")
        print(Fore.WHITE + f"  K7  = {K7:12.6f}")

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

    # --- Parity Plot ---
    plt.figure(figsize=(7, 7))

    min_val = min(min(ratec), min(rate_exp)) * 0.9
    max_val = max(max(ratec), max(rate_exp)) * 1.1

    plt.plot(ratec, rate_exp, 'bo', label=f'Data Points ({len(rate_exp)})')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

    plt.xlabel('Calculated Rate (ratec)')
    plt.ylabel('Experimental Rate (rate)')
    plt.title(f'Parity Plot: {MODEL_TO_RUN} Model')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.show()


if __name__ == "__main__":
    main()