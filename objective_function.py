import numpy as np

def objective_function(pari, xexp, rate_exp, kinetic_model, temp_exp=None):
    """
    Calculates the objective value (RMSD), predicted rates, and R^2 value.
    """
    
    number = xexp.shape[0]
    ratec = np.zeros(number)
    
    for ii in range(number):
        x0 = xexp[ii:ii+1, :]

        if temp_exp is not None:
            temp0 = np.array([temp_exp[ii]])
        else:
            temp0 = None

        ratec[ii] = kinetic_model(pari, x0, temp0)

    # RMSD objective function (same as before)
    vs = len(rate_exp)
    relative_deviation = 1 - (ratec / rate_exp)
    obj = np.sum(relative_deviation**2)
    fval = np.sqrt(obj / vs)

    # ---- R² calculation ----
    ss_res = np.sum((rate_exp - ratec)**2)
    ss_tot = np.sum((rate_exp - np.mean(rate_exp))**2)

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0

    return fval, ratec, r2


def obj_wrapper(pari, xexp, rate_exp, kinetic_model, temp_exp=None):
    fval, _, _ = objective_function(pari, xexp, rate_exp, kinetic_model, temp_exp)
    return fval
