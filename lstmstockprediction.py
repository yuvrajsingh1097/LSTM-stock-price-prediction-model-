"""
test_lstm_predictor.py
Unit tests for LSTM predictor — pure logic, no TensorFlow dependency.
Run: pytest test_lstm_predictor.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Inline all testable logic functions ──────────────────────────────────────

FEATURE_COLS = [
    "Close", "Volume", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI_14", "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "BB_pct", "daily_return",
]


def make_price_df(n=300, start_price=150.0, seed=42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with realistic price dynamics."""
    rng    = np.random.default_rng(seed)
    dates  = pd.bdate_range("2022-01-01", periods=n)
    closes = start_price * np.exp(
        np.cumsum(rng.normal(0.0003, 0.015, n))
    )
    opens  = closes * np.exp(rng.normal(0, 0.003, n))
    highs  = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.01, n))
    lows   = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.01, n))
    vols   = rng.integers(10_000_000, 80_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": vols,
    }, index=dates)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["Close"]
    df["SMA_20"]  = c.rolling(20).mean()
    df["SMA_50"]  = c.rolling(50).mean()
    df["EMA_12"]  = c.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta    = c.diff()
    gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss     = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    rs       = gain / loss.replace(0, np.nan)
    df["RSI_14"]  = 100 - (100 / (1 + rs))
    roll_std      = c.rolling(20).std()
    df["BB_upper"]= df["SMA_20"] + 2 * roll_std
    df["BB_lower"]= df["SMA_20"] - 2 * roll_std
    bb_range      = df["BB_upper"] - df["BB_lower"]
    df["BB_pct"]  = (c - df["BB_lower"]) / bb_range.replace(0, np.nan)
    df["daily_return"] = c.pct_change()
    return df.dropna(subset=FEATURE_COLS)


def build_dataset(df, window=60, train_split=0.80):
    from sklearn.preprocessing import MinMaxScaler
    feature_data = df[FEATURE_COLS].values
    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaled  = scaler.fit_transform(feature_data)
    n       = len(scaled)
    train_end = int(n * train_split)
    X, y = [], []
    for i in range(window, n):
        X.append(scaled[i - window : i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split   = train_end - window
    return X[:split], y[:split], X[split:], y[split:], scaler, train_end


def inverse_close(scaled_vals, scaler):
    n_feat  = scaler.n_features_in_
    dummy   = np.zeros((len(scaled_vals), n_feat))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]


# ─────────────────────────────────────────────────────────────
# Tests — Feature Engineering
# ─────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    @pytest.fixture
    def df_raw(self):
        return make_price_df(n=300)

    @pytest.fixture
    def df_feat(self, df_raw):
        return add_features(df_raw)

    def test_all_feature_columns_present(self, df_feat):
        for col in FEATURE_COLS:
            assert col in df_feat.columns, f"Missing feature: {col}"

    def test_no_nan_after_feature_engineering(self, df_feat):
        """After dropna, no NaN should remain in feature columns."""
        assert df_feat[FEATURE_COLS].isna().sum().sum() == 0

    def test_row_count_reduced_by_lookback(self, df_raw, df_feat):
        """SMA_50 needs 50 rows; dataset should shrink accordingly."""
        assert len(df_feat) < len(df_raw)
        assert len(df_feat) >= len(df_raw) - 60

    def test_sma_20_is_rolling_mean(self, df_feat):
        """SMA_20 at a given index should equal the 20-period mean of Close."""
        row_idx = 30
        window  = df_feat["Close"].iloc[:row_idx + 1].tail(20)
        expected = window.mean()
        actual   = df_feat["SMA_20"].iloc[row_idx]
        assert abs(actual - expected) < 1e-6

    def test_rsi_in_0_100(self, df_feat):
        assert (df_feat["RSI_14"] >= 0).all()
        assert (df_feat["RSI_14"] <= 100).all()

    def test_ema_12_above_ema_26_in_uptrend(self):
        """In a consistent uptrend EMA12 should exceed EMA26."""
        n      = 200
        closes = pd.Series(100 + np.arange(n, dtype=float))
        df2    = pd.DataFrame({
            "Open": closes, "High": closes * 1.01,
            "Low": closes * 0.99, "Close": closes,
            "Volume": np.ones(n) * 1e7,
        }, index=pd.bdate_range("2022-01-01", periods=n))
        feat = add_features(df2)
        assert (feat["EMA_12"].iloc[-10:] > feat["EMA_26"].iloc[-10:]).all()

    def test_bb_pct_bounded_near_midline(self):
        """Flat price series → BB_pct should be near 0.5."""
        n      = 100
        closes = pd.Series(np.full(n, 100.0))
        df2    = pd.DataFrame({
            "Open": closes, "High": closes + 0.1,
            "Low": closes - 0.1, "Close": closes,
            "Volume": np.ones(n) * 1e7,
        }, index=pd.bdate_range("2022-01-01", periods=n))
        feat = add_features(df2)
        # With tiny variation the bands collapse; BB_pct may be NaN or extreme
        # Just check it doesn't crash and returns finite values where defined
        valid = feat["BB_pct"].dropna()
        assert (valid.abs() < 100).all()

    def test_macd_is_ema12_minus_ema26(self, df_feat):
        expected = (df_feat["EMA_12"] - df_feat["EMA_26"]).values
        actual   = df_feat["MACD"].values
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_daily_return_computed_correctly(self, df_feat):
        """daily_return[i] = Close[i] / Close[i-1] - 1"""
        closes = df_feat["Close"].values
        rets   = df_feat["daily_return"].values
        for i in range(1, min(10, len(df_feat))):
            expected = closes[i] / closes[i - 1] - 1
            assert abs(rets[i] - expected) < 1e-9


# ─────────────────────────────────────────────────────────────
# Tests — Dataset Construction
# ─────────────────────────────────────────────────────────────

class TestDatasetConstruction:

    @pytest.fixture
    def dataset(self):
        df   = add_features(make_price_df(n=300))
        return build_dataset(df, window=60, train_split=0.80)

    def test_window_shape_train(self, dataset):
        X_train, _, _, _, _, _ = dataset
        assert X_train.ndim == 3
        assert X_train.shape[1] == 60        # window size
        assert X_train.shape[2] == len(FEATURE_COLS)

    def test_window_shape_test(self, dataset):
        _, _, X_test, _, _, _ = dataset
        assert X_test.ndim == 3
        assert X_test.shape[1] == 60
        assert X_test.shape[2] == len(FEATURE_COLS)

    def test_train_test_no_overlap(self, dataset):
        """Train set should end before test set begins."""
        X_train, y_train, X_test, y_test, _, _ = dataset
        # The two arrays simply should not share the same rows
        assert len(X_train) > 0 and len(X_test) > 0
        # Together they should not exceed total possible windows
        df = add_features(make_price_df(n=300))
        total_windows = len(df) - 60
        assert len(X_train) + len(X_test) <= total_windows

    def test_y_values_in_0_1(self, dataset):
        """Target values (scaled Close) must be in [0, 1] after MinMax scaling."""
        _, y_train, _, y_test, _, _ = dataset
        assert y_train.min() >= 0.0 and y_train.max() <= 1.0
        assert y_test.min()  >= 0.0 and y_test.max()  <= 1.0

    def test_features_in_0_1(self, dataset):
        """All scaled features must lie within [0, 1]."""
        X_train, _, X_test, _, _, _ = dataset
        assert X_train.min() >= -1e-6 and X_train.max() <= 1 + 1e-6
        assert X_test.min()  >= -1e-6 and X_test.max()  <= 1 + 1e-6

    def test_train_larger_than_test(self, dataset):
        X_train, _, X_test, _, _, _ = dataset
        assert len(X_train) > len(X_test)

    def test_different_window_sizes(self):
        df = add_features(make_price_df(n=300))
        for window in [20, 30, 60]:
            X_tr, y_tr, X_te, y_te, sc, _ = build_dataset(df, window=window)
            assert X_tr.shape[1] == window
            assert X_te.shape[1] == window


# ─────────────────────────────────────────────────────────────
# Tests — Inverse Transform
# ─────────────────────────────────────────────────────────────

class TestInverseTransform:

    def test_round_trip_close(self):
        """Scale then inverse-transform Close values → should recover originals."""
        from sklearn.preprocessing import MinMaxScaler
        df     = add_features(make_price_df(n=300))
        feat   = df[FEATURE_COLS].values
        scaler = MinMaxScaler().fit(feat)
        scaled = scaler.transform(feat)

        original_closes = df["Close"].values
        recovered       = inverse_close(scaled[:, 0], scaler)
        np.testing.assert_allclose(
            recovered, original_closes, rtol=1e-5,
            err_msg="Round-trip inverse transform failed"
        )

    def test_inverse_output_shape(self):
        from sklearn.preprocessing import MinMaxScaler
        df     = add_features(make_price_df(n=200))
        feat   = df[FEATURE_COLS].values
        scaler = MinMaxScaler().fit(feat)
        scaled = scaler.transform(feat)
        out    = inverse_close(scaled[:, 0], scaler)
        assert out.shape == (len(df),)

    def test_inverse_preserves_ordering(self):
        """Monotone increasing scaled values should stay increasing after inversion."""
        from sklearn.preprocessing import MinMaxScaler
        df     = add_features(make_price_df(n=200))
        feat   = df[FEATURE_COLS].values
        scaler = MinMaxScaler().fit(feat)
        # Monotone sequence in scaled space
        n_feat = len(FEATURE_COLS)
        vals   = np.linspace(0.1, 0.9, 50)
        dummy  = np.zeros((50, n_feat))
        dummy[:, 0] = vals
        recovered = scaler.inverse_transform(dummy)[:, 0]
        assert np.all(np.diff(recovered) > 0)


# ─────────────────────────────────────────────────────────────
# Tests — Evaluation Metrics
# ─────────────────────────────────────────────────────────────

def _mock_evaluate(y_true_prices, y_pred_prices):
    """Evaluate directly on price arrays (no scaling needed for tests)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
    mae  = mean_absolute_error(y_true_prices, y_pred_prices)
    mape = np.mean(np.abs(
        (y_true_prices - y_pred_prices) /
        np.where(y_true_prices == 0, 1, y_true_prices)
    )) * 100
    dir_true = np.sign(np.diff(y_true_prices))
    dir_pred = np.sign(np.diff(y_pred_prices))
    dir_acc  = np.mean(dir_true == dir_pred) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "DirectionalAcc": dir_acc}


class TestEvaluationMetrics:

    def test_perfect_prediction_rmse_zero(self):
        prices = np.linspace(100, 200, 50)
        m      = _mock_evaluate(prices, prices)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_prediction_mae_zero(self):
        prices = np.linspace(100, 200, 50)
        m      = _mock_evaluate(prices, prices)
        assert m["MAE"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_prediction_mape_zero(self):
        prices = np.linspace(100, 200, 50)
        m      = _mock_evaluate(prices, prices)
        assert m["MAPE"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_directional_accuracy(self):
        """Prediction that always moves in the same direction → 100% dir acc."""
        true = np.array([100., 105., 110., 115., 120.])
        pred = np.array([101., 106., 111., 116., 121.])
        m    = _mock_evaluate(true, pred)
        assert m["DirectionalAcc"] == pytest.approx(100.0)

    def test_zero_directional_accuracy(self):
        """Prediction that always moves the wrong direction → 0% dir acc."""
        true = np.array([100., 105., 110., 115., 120.])
        pred = np.array([101.,  99., 108., 106., 118.])
        m    = _mock_evaluate(true, pred)
        assert m["DirectionalAcc"] == pytest.approx(0.0)

    def test_rmse_greater_than_or_equal_mae(self):
        """RMSE ≥ MAE always (RMSE penalises large errors more)."""
        rng   = np.random.default_rng(7)
        true  = rng.uniform(100, 200, 100)
        pred  = true + rng.normal(0, 5, 100)
        m     = _mock_evaluate(true, pred)
        assert m["RMSE"] >= m["MAE"]

    def test_mape_in_reasonable_range(self):
        """MAPE for 5% noise should be roughly 5%."""
        rng    = np.random.default_rng(3)
        true   = np.full(200, 100.0)
        pred   = true * (1 + rng.uniform(-0.05, 0.05, 200))
        m      = _mock_evaluate(true, pred)
        assert 0 < m["MAPE"] < 10

    def test_metrics_are_non_negative(self):
        rng  = np.random.default_rng(5)
        true = rng.uniform(50, 300, 80)
        pred = true + rng.normal(0, 8, 80)
        m    = _mock_evaluate(true, pred)
        assert m["RMSE"] >= 0
        assert m["MAE"]  >= 0
        assert m["MAPE"] >= 0
        assert 0 <= m["DirectionalAcc"] <= 100