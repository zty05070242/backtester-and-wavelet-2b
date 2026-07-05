"""
Wavelet denoising utilities.

Two entry points:
  - wavelet_denoise():         global (full-series) denoising, USES FUTURE DATA.
                               Offline / research only. Never inside a backtest signal.
  - rolling_wavelet_denoise(): causal rolling-window version. Slow but honest.
                               Live-tradeable: only past data is ever used at time t.

The core recipe is the classic Donoho-Johnstone universal threshold:
  1. Discrete wavelet transform (DWT) decomposes the signal into coarse
     approximation + a pyramid of detail coefficient bands.
  2. Estimate noise level sigma from the finest detail band via MAD
     (median absolute deviation), which is robust to outliers.
  3. Apply the universal threshold sigma * sqrt(2 * log(n)) to every
     detail band (soft or hard thresholding).
  4. Inverse transform to reconstruct the cleaned signal.

DSP note: the same pattern as denoising an audio signal — keep the coarse
envelope, squash the noisy high-frequency detail. Soft thresholding
(shrink-toward-zero) is the mean-square-error-optimal choice under Gaussian
noise; hard thresholding (keep-or-kill) preserves edges better but is
discontinuous.
"""

import numpy as np
import pandas as pd
import pywt


def _estimate_sigma(detail_coeffs: np.ndarray) -> float:
    """MAD-based sigma estimate from the finest detail band.

    0.6745 is the 75th percentile of N(0,1), so MAD/0.6745 is consistent
    with the std of Gaussian noise.
    """
    return np.median(np.abs(detail_coeffs)) / 0.6745


def _universal_threshold(n: int, sigma: float) -> float:
    """Donoho-Johnstone universal threshold: sigma * sqrt(2 * log(n)).

    Scales with series length — more samples, higher chance of a noise spike
    exceeding any fixed threshold.
    """
    return sigma * np.sqrt(2.0 * np.log(n))


def _denoise_array(x: np.ndarray, wavelet: str, level, mode: str,
                   threshold_scale: float = 1.0) -> np.ndarray:
    """
    Core denoising on a raw numpy array. Used by both the global and the
    rolling wrappers so the thresholding logic lives in one place.

    threshold_scale multiplies the universal threshold before applying it.
    1.0 = original Donoho-Johnstone behaviour (aggressive — designed for
    worst-case noise recovery in stationary signals like returns).
    0.5 is a good starting point when denoising price directly, preserving
    more medium-frequency structure (weekly/monthly swings) while still
    suppressing the finest-scale noise.
    """
    # pywt.dwt_max_level tells us the deepest decomposition this length supports
    # for this wavelet. If the caller didn't pick a level, use the maximum.
    if level is None:
        level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        # Guard: pywt.dwt_max_level can return 0 for tiny inputs.
        level = max(level, 1)

    # pywt rejects read-only numpy buffers (pandas' .values is often read-only),
    # so force a writable copy up front. np.array(..., copy=True) guarantees
    # a freshly allocated, writable array.
    x = np.array(x, dtype=np.float64, copy=True)

    # wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    # cA_n = coarse approximation at the deepest level
    # cD_k = detail coefficients (high-frequency) at level k
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # Noise sigma is estimated from the FINEST detail band (cD_1, last in list).
    # Finest detail is where noise dominates signal.
    sigma = _estimate_sigma(coeffs[-1])
    # threshold_scale lets callers dial down the aggressiveness; see docstring.
    threshold = _universal_threshold(len(x), sigma) * threshold_scale

    # Threshold every detail band; leave the coarse approximation untouched
    # (that's the low-frequency trend we want to keep).
    new_coeffs = [coeffs[0]]
    for cD in coeffs[1:]:
        new_coeffs.append(pywt.threshold(cD, threshold, mode=mode))

    # Inverse DWT to reconstruct. waverec output length can be +-1 vs input
    # (depends on wavelet filter length + boundary handling), so we trim.
    cleaned = pywt.waverec(new_coeffs, wavelet)
    return cleaned[: len(x)]


def wavelet_denoise(
    series: pd.Series,
    wavelet: str = "db6",
    level=None,
    mode: str = "soft",
    threshold_scale: float = 1.0,
) -> pd.Series:
    """Global wavelet denoising — one-shot over the full series.

    WARNING: uses future data. Reconstruction at index t depends on samples
    before and after t, so don't use this inside a backtest signal. It's for
    offline analysis and as an upper-bound baseline against the causal version.

    series: pandas Series (close prices or returns).
    wavelet: pywt wavelet name. db6 is a good default — smooth, compact support.
    level: decomposition depth; None uses the max the data length allows.
    mode: 'soft' (shrink toward zero) or 'hard' (keep-or-kill).
    threshold_scale: multiplier on the universal threshold. 1.0 = standard DJ.
        0.5 = less aggressive, better for denoising price directly.
    """
    if mode not in ("soft", "hard"):
        raise ValueError("mode must be 'soft' or 'hard'")
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    # Drop NaNs first — DWT can't handle them. Caller keeps the original index,
    # so we re-align at the end.
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return series.copy()

    cleaned_vals = _denoise_array(clean_series.values, wavelet, level, mode,
                                  threshold_scale)
    return pd.Series(cleaned_vals, index=clean_series.index, name=series.name)


def rolling_wavelet_denoise(
    series: pd.Series,
    window: int = 252,
    wavelet: str = "db6",
    mode: str = "soft",
    threshold_scale: float = 1.0,
) -> pd.Series:
    """Causal rolling-window wavelet denoising — live-tradeable version.

    At each bar t, denoise the prior window samples and keep only the last
    reconstructed value as the estimate at t. Nothing after t is touched,
    so no look-ahead bias — same guarantee as Kalman .filter() calls.

    O(window * N) work — slower than the global version. If that's a problem,
    lower window (e.g. 128) or cache results between runs.

    series: pandas Series with a DatetimeIndex.
    window: lookback in bars. 252 ~ 1 year of daily data.
    wavelet: pywt wavelet name.
    mode: 'soft' or 'hard' thresholding.
    threshold_scale: multiplier on the universal threshold. 1.0 = standard DJ.
        0.5 = gentler, better for denoising price directly.

    first window-1 values in the output are NaN (window not yet full).
    """
    if mode not in ("soft", "hard"):
        raise ValueError("mode must be 'soft' or 'hard'")
    if window < 8:
        # Wavelet decomposition needs a reasonable minimum length to be meaningful.
        raise ValueError("window too small for wavelet decomposition (min 8)")

    clean_series = series.dropna()
    values = clean_series.values
    n = len(values)

    out = np.full(n, np.nan)
    # Pick a fixed decomposition level for the window size so every slice is
    # treated the same way — reproducibility matters when comparing to global.
    level = pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len)
    level = max(level, 1)

    # Slide through the series. For each full-sized window, denoise and
    # keep only the last reconstructed sample.
    for t in range(window - 1, n):
        window_slice = values[t - window + 1 : t + 1]
        denoised = _denoise_array(window_slice, wavelet, level, mode, threshold_scale)
        out[t] = denoised[-1]

    return pd.Series(out, index=clean_series.index, name=series.name)


if __name__ == "__main__":
    import plotly.graph_objects as go
    from data_loader import load_historical_data

    # Demo: compare raw returns to global-denoised and rolling-denoised returns
    # on the S&P 500. We denoise RETURNS, not price, because returns are
    # (approximately) stationary — which is the assumption baked into the
    # universal-threshold noise model.
    df = load_historical_data("^GSPC", "2010-01-01", "2026-04-15")

    returns = df["close"].pct_change().dropna()

    # Global: uses future data. Sharper, but cheating for live use.
    returns_global = wavelet_denoise(returns, wavelet="db6", mode="soft")

    # Rolling: causal. This is what you could actually trade on.
    returns_rolling = rolling_wavelet_denoise(
        returns, window=252, wavelet="db6", mode="soft"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns,
                             name="Raw returns", line=dict(width=1, color="lightgray")))
    fig.add_trace(go.Scatter(x=returns_global.index, y=returns_global,
                             name="Global denoised (LOOK-AHEAD)",
                             line=dict(width=1.2, color="blue")))
    fig.add_trace(go.Scatter(x=returns_rolling.index, y=returns_rolling,
                             name="Rolling denoised (causal)",
                             line=dict(width=1.2, color="orange")))

    fig.update_layout(
        title="Wavelet denoising — S&P 500 daily returns (db6, soft threshold)",
        xaxis_title="Date", yaxis_title="Return",
        hovermode="x unified",
    )

    save_path = "wavelet_denoise_demo.html"
    fig.write_html(save_path)
    print(f"Saved {save_path}")
