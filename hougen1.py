import numpy as np

def hougen1(parm, x):
    """
    Calculates the reaction rate (yhat) based on the Hougen-Watson model.
    The model is: yhat = ((b1*b2*b3) * x1 * x2^0.5) / (1 + b2*x1 + b3*x2^0.5)^2

    :param parm: Model parameters [b1, b2, b3].
    :param x: Experimental conditions (N x 2 array), where x[:, 0] is x1 and x[:, 1] is x2.
    :return: Calculated rate (yhat).
    """
    # Unpack parameters: b(1), b(2), b(3) in MATLAB
    b1, b2, b3 = parm[0], parm[1], parm[2]

    # Variables: x(:,1) and x(:,2) in MATLAB
    x1 = x[:, 0]
    x2 = x[:, 1]

    # Rate function calculation
    numerator = (b1 * b2 * b3) * x1 * np.sqrt(x2)
    denominator = (1 + b2 * x1 + b3 * np.sqrt(x2))**2
    yhat = numerator / denominator

    # If only a single data point was passed (x.shape[0] == 1), return the scalar rate.
    if x.shape[0] == 1:
        return yhat[0]
    else:
        return yhat