import pandas as pd

from lox.regimes.fci import build_fci_feature_matrix


def test_build_fci_feature_matrix_smoke():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    X = pd.DataFrame(
        {
            "rates_z_ust_10y_chg_20d": [0, 1, 2, 3, 4],
            "usd_strength_score": [0, 0, 0, 0, 0],
            "funding_tightness_score": [1, 1, 1, 1, 1],
            "vol_pressure_score": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=idx,
    )
    fci = build_fci_feature_matrix(X)
    assert "fci_score" in fci.columns
    assert fci.shape[0] == X.shape[0]
    assert fci["fci_score"].isna().sum() == 0


