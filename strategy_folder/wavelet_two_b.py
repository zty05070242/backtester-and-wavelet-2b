from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
from wavelet_denoiser import rolling_wavelet_denoise
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class WaveletTwoB(Strategy):
    """
    Sperandeo's 2B failed-breakout rule with swing highs/lows identified
    by a DSP pivot-detection pipeline instead of a fixed rolling-max lookback.

    Plain TwoB uses high.rolling(N).max() as the prior swing high — the
    highest bar in the last N bars, even if it's a single noisy spike. This
    class replaces that with a pipeline that approximates how a human reads
    a chart: denoise, find structural peaks, gate by prominence and lag.

        rolling causal wavelet denoise of close
            -> scipy.signal.find_peaks on the denoised series
            -> filter peaks by prominence in ATR units
            -> confirm only after pivot_confirm_bars of follow-through

    The 2B failed-breakout logic itself is identical to TwoB — only the
    source of swing levels changes.

    Causality note: rolling_wavelet_denoise is causal (denoised[t] uses only
    the prior denoise_window bars). A peak at bar k is only "known" once
    t >= k + pivot_confirm_bars. One asterisk: scipy.find_peaks computes
    prominence over the full array, so pivot selection could shift slightly
    as later data arrives. The trade decision itself doesn't look ahead.
    """

    def __init__(
        self,
        denoise_window: int = 128,
        wavelet: str = "db6",
        threshold_scale: float = 0.5,
        min_prominence_atr: float = 1.0,
        min_pivot_distance: int = 5,
        pivot_confirm_bars: int = 3,
        confirmation_days: int = 3,
        atr_period: int = 14,
    ):
        # denoise_window: rolling lookback for the causal wavelet denoiser
        # wavelet: pywt wavelet name — db6 is smooth with compact support
        # threshold_scale: multiplier on the DJ universal threshold; 0.5 keeps
        #   medium-frequency swings while squashing tick noise
        # min_prominence_atr: peak counts only if prominence >= this * ATR at
        #   the peak bar — dimensionless, self-adapts across assets
        # min_pivot_distance: minimum bar gap between detected pivots
        # pivot_confirm_bars: pivot at bar k is "known" only at k + this value
        # confirmation_days: bars after breakout to close back through (1-5)
        # atr_period: ATR window for prominence-gating
        if denoise_window < 16:
            raise ValueError("denoise_window too small for wavelet decomposition.")
        if confirmation_days < 1 or confirmation_days > 5:
            raise ValueError("confirmation_days must be between 1 and 5.")
        if pivot_confirm_bars < 1:
            raise ValueError("pivot_confirm_bars must be >= 1.")
        if min_pivot_distance < 1:
            raise ValueError("min_pivot_distance must be >= 1.")

        super().__init__(
            name=f"Wavelet-2B (win={denoise_window}, prom={min_prominence_atr}ATR)"
        )
        self.denoise_window = denoise_window
        self.wavelet = wavelet
        self.threshold_scale = threshold_scale
        self.min_prominence_atr = min_prominence_atr
        self.min_pivot_distance = min_pivot_distance
        self.pivot_confirm_bars = pivot_confirm_bars
        self.confirmation_days = confirmation_days
        self.atr_period = atr_period

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")

        df = self.data.copy()

        # --- Step 1: causal wavelet denoise of close ---
        df['wavelet_close'] = rolling_wavelet_denoise(
            df['close'],
            window=self.denoise_window,
            wavelet=self.wavelet,
            mode="soft",
            threshold_scale=self.threshold_scale,
        )

        # --- Step 2: ATR for prominence-gating and breakout-distance filter ---
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()

        # Drop warm-up rows where wavelet or ATR aren't yet defined.
        df = df.dropna(subset=['wavelet_close', 'atr']).copy()

        denoised = df['wavelet_close'].values
        atrs = df['atr'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)

        # --- Step 3: find peaks (swing highs) and troughs (swing lows) ---
        # prominence=0 returns every local extremum with its prominence value;
        # we filter per-pivot below so the threshold can scale with ATR at
        # each pivot bar (find_peaks `prominence=` only accepts a scalar).
        peak_idx, peak_props = find_peaks(
            denoised, prominence=0.0, distance=self.min_pivot_distance
        )
        trough_idx, trough_props = find_peaks(
            -denoised, prominence=0.0, distance=self.min_pivot_distance
        )

        # Prominence is in the same units as the input (price), so dividing
        # by ATR gives a dimensionless multiple — keep pivots whose prominence
        # is at least min_prominence_atr * ATR at the pivot bar.
        peak_keep = peak_props['prominences'] >= self.min_prominence_atr * atrs[peak_idx]
        trough_keep = trough_props['prominences'] >= self.min_prominence_atr * atrs[trough_idx]
        peak_idx = peak_idx[peak_keep]
        trough_idx = trough_idx[trough_keep]

        # The breakout has to clear the actual bar extreme, not the denoised
        # value — use the raw high at the peak bar (raw low at the trough bar)
        # as the reference price for the 2B test.
        peak_levels = highs[peak_idx]
        trough_levels = lows[trough_idx]

        # --- Step 4: build per-bar "active swing high / swing low" ---
        # At bar t, the active swing high is the level of the MOST RECENT peak
        # that has reached its confirmation lag (k + pivot_confirm_bars <= t).
        # Sperandeo's 2B reference is the most recent swing, not the all-time
        # high — what matters is the most recent failed breakout level.
        active_swing_high = np.full(n, np.nan)
        active_swing_low = np.full(n, np.nan)

        p_ptr, t_ptr = 0, 0
        current_high = np.nan
        current_low = np.nan
        for t in range(n):
            # Promote any peak whose confirmation lag has elapsed.
            while p_ptr < len(peak_idx) and peak_idx[p_ptr] + self.pivot_confirm_bars <= t:
                current_high = peak_levels[p_ptr]
                p_ptr += 1
            while t_ptr < len(trough_idx) and trough_idx[t_ptr] + self.pivot_confirm_bars <= t:
                current_low = trough_levels[t_ptr]
                t_ptr += 1
            active_swing_high[t] = current_high
            active_swing_low[t] = current_low

        df['swing_high'] = active_swing_high
        df['swing_low'] = active_swing_low

        # --- Step 5: 2B failed-breakout logic against the wavelet pivots ---
        # Identical to TwoB.generate_signals() — only the source of
        # swing_high / swing_low has changed.
        signals = np.zeros(n)
        for i in range(n):
            sh = active_swing_high[i]
            sl = active_swing_low[i]

            # Bearish 2B (short): break above prior swing high...
            if not np.isnan(sh) and highs[i] > sh:
                # ...then close back below within confirmation_days.
                end = min(i + self.confirmation_days + 1, n)
                for j in range(i, end):
                    if closes[j] < sh:
                        signals[j] = -1.0
                        break

            # Bullish 2B (long): break below prior swing low...
            if not np.isnan(sl) and lows[i] < sl:
                end = min(i + self.confirmation_days + 1, n)
                for j in range(i, end):
                    if closes[j] > sl:
                        # Don't overwrite an existing short signal on the
                        # same bar — same precedence rule as TwoB.
                        if signals[j] == 0.0:
                            signals[j] = 1.0
                        break

        df['signal'] = signals
        self.data = df
        self._signals_generated = True

        return df


if __name__ == "__main__":
    # Standalone smoke test: generate signals, run a backtest, save the chart.
    # Mirrors the pattern in backtester.py's __main__ so you can run this file
    # directly and see the equity curve + signals.
    from backtester import Backtester
    from chart import plot_signals

    strategy = WaveletTwoB(
        denoise_window=128,
        wavelet="db6",
        threshold_scale=0.5,
        min_prominence_atr=1.0,
        min_pivot_distance=5,
        pivot_confirm_bars=3,
        confirmation_days=3,
        atr_period=14,
    )
    print(f"Testing strategy: {strategy.name}")
    df = load_historical_data("SI=F", "2000-01-01", "2026-04-15")

    backtester = Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001)
    results = backtester.run(df, strategy, verbose=True)

    plot_signals(strategy)
