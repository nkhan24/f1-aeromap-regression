"""Step 3 - scikit-learn: compare a straight-line model with a curved one
using leave-one-out cross-validation (each point is predicted by a model
that never saw it)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from aeromap import load_sweep, add_aero_loads

df = add_aero_loads(load_sweep("data/ride_height_sweep.csv"))
X = df[["dyn_frh_mm", "dyn_rrh_mm"]]

models = {
    "Linear": LinearRegression(),
    "Quadratic": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
}

for target in ["front_aero_n", "rear_aero_n"]:
    y = df[target]
    print(target)
    for name, model in models.items():
        # Error on the data the model was trained on
        fit_pred = model.fit(X, y).predict(X)
        fit_rmse = np.sqrt(np.mean((y - fit_pred) ** 2))
        # Error on points the model never saw
        cv_pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
        cv_rmse = np.sqrt(np.mean((y - cv_pred) ** 2))
        print(f"  {name:9s} training error {fit_rmse:5.2f} N | unseen-point error {cv_rmse:5.2f} N")
    print()