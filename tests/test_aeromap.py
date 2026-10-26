"""Tests: run with  python -m pytest"""
import numpy as np
import pandas as pd
from aeromap import add_aero_loads, fit_plane


def test_aero_load_is_zero_at_static_load():
    df = pd.DataFrame({"ffz_n": [3737.14892], "rfz_n": [4300.501314]})
    out = add_aero_loads(df)
    assert out["front_aero_n"][0] == 0
    assert out["rear_aero_n"][0] == 0


def test_fit_recovers_known_slopes():
    # Made-up data with known answers: load = 100 - 40*frh + 6*rrh
    frh = np.array([10, 12, 14, 16, 18, 20.0])
    rrh = np.array([22, 30, 23, 35, 24, 40.0])
    df = pd.DataFrame({"dyn_frh_mm": frh, "dyn_rrh_mm": rrh,
                       "load": 100 - 40 * frh + 6 * rrh})
    coefs, _ = fit_plane(df, "load")
    assert np.allclose(coefs, [100, -40, 6])