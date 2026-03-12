import numpy as np

# Universal gas constant (J/mol/K)
R = 8.314


def hougen_watson(parm, x, temp=None):
    """
    Hougen–Watson model (FT3 RDS-10): 
        r = ((b1*b2*b3) * x1 * x2^0.5) / (1 + b2*x1 + b3*x2^0.5)^2

    :param parm: Model parameters [b1, b2, b3].
    :param x: Independent variables (N x 2 array): x[:, 0] is x1, x[:, 1] is x2.
    :param temp: Not used (only for interface compatibility).
    :return: Calculated rate (yhat).
    """
    b1, b2, b3 = parm[0], parm[1], parm[2]
    x1 = x[:, 0]
    x2 = x[:, 1]

    numerator = (b1 * b2 * b3) * x1 * np.sqrt(x2)
    denominator = (1 + b2 * x1 + b3 * np.sqrt(x2))**2

    yhat = numerator / denominator

    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat


def mars_van_krevelen(parm, x, temp):
    """
    Mars–van Krevelen Redox model with Arrhenius temperature dependence:

        k_r(T) = k_r0 * exp(-Ea_r / (R*T))
        k_o(T) = k_o0 * exp(-Ea_o / (R*T))

    Rate expression (example form):
        r = (k_r * k_o * P_MeOH * P_O2^0.5) / (k_r * P_MeOH + k_o * P_O2^0.5)

    Parameters:
        parm = [ln_kr0, Ea_r, ln_ko0, Ea_o]
            ln_kr0 : ln(pre-exponential factor for k_r)
            Ea_r   : activation energy for k_r (J/mol)
            ln_ko0 : ln(pre-exponential factor for k_o)
            Ea_o   : activation energy for k_o (J/mol)

    :param parm: Model parameters [ln_kr0, Ea_r, ln_ko0, Ea_o].
    :param x:    Independent variables (N x 2 array): P_MeOH, P_O2.
    :param temp: Temperature array (N,) in °C.
    :return: Calculated rate (yhat).
    """
    if temp is None:
        raise ValueError("Temperature array 'temp' must be provided for the MVK Arrhenius model.")

    temp = np.asarray(temp, dtype=float)
    ln_kr0, Ea_r, ln_ko0, Ea_o = parm

    # Convert °C -> K (assumes Excel temp is in °C)
    T = temp + 273.15

    # Arrhenius rate constants
    kr0 = np.exp(ln_kr0)
    ko0 = np.exp(ln_ko0)

    kr = kr0 * np.exp(-Ea_r / (R * T))
    ko = ko0 * np.exp(-Ea_o / (R * T))

    P_MeOH = x[:, 0]
    P_O2 = x[:, 1]

    numerator = kr * ko * P_MeOH * np.sqrt(P_O2)
    denominator = kr * P_MeOH + ko * np.sqrt(P_O2)

    # Robust division
    yhat = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat


def jiru_model(parm, x, temp):
    """
    Jiru et al. model with Arrhenius temperature dependence for k1 and k2:

        k1(T) = k10 * exp(-Ea1 / (R*T))
        k2(T) = k20 * exp(-Ea2 / (R*T))

        r = (k1(T) * P_MeOH^m) /
            ( 1 + 0.5 * (k1(T) * P_MeOH^m) / (k2(T) * P_O2^n) )

    Parameters
    ----------
    parm : array-like
        [ln_k10, Ea1, ln_k20, Ea2, m, n]
        ln_k10, ln_k20 : logs of pre-exponential factors
        Ea1, Ea2       : activation energies (J/mol)
        m, n           : reaction orders in MeOH and O2

    x : ndarray (N x 2)
        x[:, 0] = P_MeOH
        x[:, 1] = P_O2

    temp : ndarray (N,) or (1,)
        Temperatures in °C (will be converted to K).

    Returns
    -------
    yhat : ndarray
        Predicted rates for each (P_MeOH, P_O2).
    """
    if temp is None:
        raise ValueError("Temperature array 'temp' must be provided for the Jiru Arrhenius model.")

    temp = np.asarray(temp, dtype=float)

    ln_k10, Ea1, ln_k20, Ea2, m, n = parm

    # Temperature in K
    T = temp + 273.15

    # Arrhenius rate constants
    k10 = np.exp(ln_k10)
    k20 = np.exp(ln_k20)

    k1 = k10 * np.exp(-Ea1 / (R * T))
    k2 = k20 * np.exp(-Ea2 / (R * T))

    P_MeOH = x[:, 0]
    P_O2   = x[:, 1]

    num = k1 * (P_MeOH ** m)
    den = 1.0 + 0.5 * (k1 * (P_MeOH ** m)) / (k2 * (P_O2 ** n))

    # Robust division
    yhat = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat
