"""Step 2 - SciPy: fit the two-variable model and check it against Excel LINEST."""
import matplotlib.pyplot as plt
from aeromap import load_sweep, add_aero_loads, fit_plane, plane, r_squared

df = add_aero_loads(load_sweep("data/ride_height_sweep.csv"))
X = (df["dyn_frh_mm"], df["dyn_rrh_mm"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, target, name in zip(axes, ["front_aero_n", "rear_aero_n"], ["Front", "Rear"]):
    coefs, se = fit_plane(df, target)
    pred = plane(X, *coefs)
    resid = df[target] - pred
    print(f"{name} axle")
    print(f"  per mm front ride height: {coefs[1]:8.2f} N  (standard error {se[1]:.3f})")
    print(f"  per mm rear ride height:  {coefs[2]:8.2f} N  (standard error {se[2]:.3f})")
    print(f"  R squared: {r_squared(df[target], pred):.5f}")
    print(f"  largest residual: {resid.abs().max():.1f} N\n")
    ax.scatter(df["dyn_frh_mm"], resid)
    ax.axhline(0, color="grey")
    ax.set_xlabel("Dynamic front ride height (mm)")
    ax.set_ylabel("Residual (N)")
    ax.set_title(f"{name} axle residuals")

plt.tight_layout()
plt.savefig("results/residuals.png", dpi=150)
print("Saved results/residuals.png")