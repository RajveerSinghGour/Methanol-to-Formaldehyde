import numpy as np

# Universal gas constant (J/mol/K)
R = 8.314


def hougen_watson(parm, x, temp=None):
    b1, b2, b3 = parm

    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    numerator = (b1 * b2 * b3) * P_MeOH * np.sqrt(P_O2)
    denominator = (1 + b2 * P_MeOH + b3 * np.sqrt(P_O2)) ** 2

    yhat = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator != 0)

    return yhat[0] if x.shape[0] == 1 else yhat


def mars_van_krevelen(parm, x, temp):
    if temp is None:
        raise ValueError("Temperature required for MVK model")

    temp = np.asarray(temp, dtype=float)
    T = temp + 273.15

    ln_kr0, Ea_r, ln_ko0, Ea_o = parm

    kr0 = np.exp(ln_kr0)
    ko0 = np.exp(ln_ko0)

    kr = kr0 * np.exp(-Ea_r / (R * T))
    ko = ko0 * np.exp(-Ea_o / (R * T))

    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    numerator = kr * ko * P_MeOH * np.sqrt(P_O2)
    denominator = kr * P_MeOH + ko * np.sqrt(P_O2)

    yhat = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator != 0)

    return yhat[0] if x.shape[0] == 1 else yhat


def jiru_model(parm, x, temp):
    if temp is None:
        raise ValueError("Temperature required for Jiru model")

    temp = np.asarray(temp, dtype=float)
    T = temp + 273.15

    ln_k10, Ea1, ln_k20, Ea2, m, n = parm

    k10 = np.exp(ln_k10)
    k20 = np.exp(ln_k20)

    k1 = k10 * np.exp(-Ea1 / (R * T))
    k2 = k20 * np.exp(-Ea2 / (R * T))

    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    numerator = k1 * (P_MeOH ** m)
    denominator = 1 + 0.5 * (k1 * (P_MeOH ** m)) / (k2 * (P_O2 ** n))

    yhat = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator != 0)

    return yhat[0] if x.shape[0] == 1 else yhat


def langmuir_hinshelwood(parm, x, temp):
    if temp is None:
        raise ValueError("Temperature required for L-H model")

    temp = np.asarray(temp, dtype=float)
    T = temp + 273.15

    ln_k0, Ea, K_MeOH, K_O2 = parm

    k0 = np.exp(ln_k0)
    k = k0 * np.exp(-Ea / (R * T))

    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    numerator = k * P_MeOH * P_O2
    denominator = (1 + K_MeOH * P_MeOH + K_O2 * P_O2) ** 2

    yhat = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator != 0)

    return yhat[0] if x.shape[0] == 1 else yhat


# ⭐ NEW MODEL ADDED
def lhhw_formaldehyde(parm, x, temp=None):
    """
    Langmuir–Hinshelwood–Hougen–Watson (LHHW) model for formaldehyde

    x[:,0] = PMeOH
    x[:,1] = PO2
    x[:,2] = PH2
    x[:,3] = PH2O
    """

    if x.ndim != 2 or x.shape[1] < 4:
        raise ValueError(
            f"LHHW model requires 4 input columns: PMeOH, PO2, PH2, PH2O. Received shape {x.shape}. "
            "Please add 'P_H2_in' and 'P_H2O_in' to your dataset or choose a compatible model."
        )

    kf, K1, K2, K5, K6, K7 = parm

    PMeOH = x[:, 0]
    PO2 = x[:, 1]
    PH2 = x[:, 2]
    PH2O = x[:, 3]

    numerator = (
        kf
        * K1
        * K2
        * PMeOH
        * np.sqrt(K5)
        / np.sqrt(PH2)
    )

    denominator = (
        1
        + K1 * PMeOH
        + np.sqrt(PH2 / K5)
        + np.sqrt(K6 * PO2)
        + np.sqrt(K7 * PH2O)
    ) ** 2

    yhat = np.divide(numerator, denominator,
                     out=np.zeros_like(numerator),
                     where=denominator != 0)

    return yhat[0] if x.shape[0] == 1 else yhat