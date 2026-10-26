"""Functions for the F1 25 aero map regression project.

Units: ride heights in mm, loads in N.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Static axle loads measured with the car stationary
# (sheet "Sweeps RH Changes MEX", cells G3 and I3).
STATIC_FRONT_N = 3737.14892
STATIC_REAR_N = 4300.501314


def load_sweep(path):
    """Read the ride-height sweep CSV into a pandas DataFrame."""
    return pd.read_csv(path)


def add_aero_loads(df, static_front=STATIC_FRONT_N, static_rear=STATIC_REAR_N):
    """Add aero load columns: load at speed minus static load."""
    out = df.copy()
    out["front_aero_n"] = out["ffz_n"] - static_front
    out["rear_aero_n"] = out["rfz_n"] - static_rear
    return out

def plane(X, a, b, c):
    """Linear model: load = a + b * front ride height + c * rear ride height."""
    frh, rrh = X
    return a + b * frh + c * rrh


def fit_plane(df, target):
    """Fit the linear model to one target column.

    Returns the coefficients (a, b, c) and their standard errors.
    """
    X = (df["dyn_frh_mm"].to_numpy(), df["dyn_rrh_mm"].to_numpy())
    y = df[target].to_numpy()
    coefs, cov = curve_fit(plane, X, y)
    std_errors = np.sqrt(np.diag(cov))
    return coefs, std_errors


def r_squared(y, y_pred):
    """Share of the variation in y explained by the model (1.0 = perfect)."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot