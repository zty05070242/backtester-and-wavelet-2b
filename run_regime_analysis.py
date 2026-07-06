"""
run_regime_analysis.py — HMM regime overlay on the 2B vs Wavelet-2B experiment.

For each of the 10 commodity futures, fits a 3-state Gaussian HMM
(ranging / trending / volatile) on rolling realized vol + 20-day return, then:

  1. Splits the trade log from each backtest by regime at entry date — showing
     which regime each strategy earns (or loses) money in.
  2. Runs a crisis-filtered version of each backtest: signals in the volatile
     regime are zeroed out using a causal rolling HMM (no look-ahead).

The regime-split table answers the NG=F question: if a market spends most of
its time in the volatile regime and that regime has a profit factor below 1,
the strategy's underperformance is structurally explained, not a mystery.

Saves:
  results/regime_analysis_YYYYMMDD.csv — per-(ticker, strategy, regime) trade metrics
  results/regime_gated_YYYYMMDD.csv    — ungated vs crisis-filtered comparison

Usage: python run_regime_analysis.py
"""

from __future__ import annotations
import os
from datetime import date
from typing import List, Tuple

import pandas as pd

from data_loader import load_historical_data
from backtester import Backtester
from backtester_scaled import BacktesterScaled
from regime_hmm import decode_regimes_full, rolling_causal_regimes
from strategy_folder._strategy_base_class import Strategy
from strategy_folder.two_b import TwoB
from strategy_folder.wavelet_two_b import WaveletTwoB


# --- Config (mirrors run_comparison.py) ----------------------------------------

TICKERS = [
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F',
    'ZW=F', 'ZC=F', 'ZS=F', 'KC=F', 'LE=F',
]
TICKER_NAMES = {
    'GC=F': 'Gold',        'SI=F': 'Silver',    'CL=F': 'WTI Crude',
    'NG=F': 'Natural Gas', 'HG=F': 'Copper',    'ZW=F': 'Wheat',
    'ZC=F': 'Corn',        'ZS=F': 'Soybeans',  'KC=F': 'Coffee',
    'LE=F': 'Live Cattle',
}
START_DATE      = '2000-01-01'
END_DATE        = '2026-04-15'
INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.02
SLIPPAGE_PCT    = 0.0001
MAX_TRANCHES    = 3
RESULTS_DIR     = 'results'

HMM_N_STATES    = 3
HMM_TRAIN_WIN   = 1260   # ~5 years for causal rolling refit
HMM_REFIT_EVERY = 63     # ~1 quarter

VOLATILE_REGIME = 'volatile'
ALL_REGIMES     = ['ranging', 'trending', 'volatile']

BACKTESTER_VARIANTS: List[Tuple[str, object]] = [
    ('',          lambda: Backtester(INITIAL_BALANCE, RISK_PCT, SLIPPAGE_PCT)),
    (' (scaled)', lambda: BacktesterScaled(
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        slippage_pct=SLIPPAGE_PCT,
        max_tranches=MAX_TRANCHES,
    )),
]


# -------------------------------------------------------------------------------

class _ReplayStrategy(Strategy):
    """Wraps a pre-computed signals DataFrame so generate_signals() is free."""
    def __init__(self, name: str, signals_df: pd.DataFrame):
        super().__init__(name=name)
        self._cached = signals_df.copy()

    def generate_signals(self) -> pd.DataFrame:
        self.data = self._cached.copy()
        self._signals_generated = True
        return self.data


def _make_strategies() -> List[Tuple[str, object]]:
    return [
        ('2B Rule', TwoB(lookback=20, confirmation_days=3)),
        ('Wavelet-2B', WaveletTwoB(
            denoise_window=128, min_prominence_atr=1.0,
            pivot_confirm_bars=3, confirmation_days=3,
        )),
    ]


def _trade_metrics(trades: list) -> dict:
    """Trade-level metrics for a subset of the trade log."""
    n = len(trades)
    if n == 0:
        return {'n_trades': 0, 'win_rate_pct': None, 'profit_factor': None, 'avg_pnl': None}
    wins      = [t for t in trades if t['pnl'] > 0]
    losses    = [t for t in trades if t['pnl'] < 0]
    gp        = sum(t['pnl'] for t in wins)
    gl        = abs(sum(t['pnl'] for t in losses))
    pf        = round(gp / gl, 2) if gl > 0 else float('inf')
    return {
        'n_trades':      n,
        'win_rate_pct':  round(len(wins) / n * 100, 1),
        'profit_factor': pf,
        'avg_pnl':       round(sum(t['pnl'] for t in trades) / n, 2),
    }


def run_regime_analysis(tickers: List[str] = None) -> None:
    tickers = tickers or TICKERS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    analysis_rows: list[dict] = []   # per (ticker, strategy, regime) trade metrics
    gated_rows:    list[dict] = []   # ungated vs crisis-filtered full metrics

    for ticker in tickers:
        name = TICKER_NAMES.get(ticker, ticker)
        print(f"\n=== {ticker} ({name}) ===")
        df = load_historical_data(ticker, START_DATE, END_DATE)

        # --- Regime classification ---
        print("  fitting HMM (full-series, retrospective)...")
        regimes_full = decode_regimes_full(df['close'], n_states=HMM_N_STATES)

        print("  fitting HMM (causal rolling — this takes a minute)...")
        regimes_causal = rolling_causal_regimes(
            df['close'],
            train_window=HMM_TRAIN_WIN,
            refit_every=HMM_REFIT_EVERY,
            n_states=HMM_N_STATES,
        )

        # Regime time distribution — the key descriptive stat
        valid_full = regimes_full.dropna()
        regime_pct = valid_full.value_counts(normalize=True).mul(100)
        print("  regime time distribution (full-series):")
        for reg in ALL_REGIMES:
            pct = regime_pct.get(reg, 0.0)
            print(f"    {reg:12s}: {pct:.1f}%")

        # Dates where causal regime is volatile (used for signal gating)
        volatile_dates = set(regimes_causal[regimes_causal == VOLATILE_REGIME].index)

        # --- Per-strategy loop ---
        for s_label, strat in _make_strategies():
            print(f"  generating signals: {s_label}")
            strat.set_data(df)
            signals_df = strat.generate_signals()

            for bt_suffix, make_bt in BACKTESTER_VARIANTS:
                full_label = s_label + bt_suffix
                print(f"    backtest: {full_label}")

                # ---- Ungated ----
                replay  = _ReplayStrategy(strat.name, signals_df)
                metrics = make_bt().run(df, replay, verbose=False)
                trades  = metrics['trades']

                # Split trade log by regime at entry date
                for regime in ALL_REGIMES:
                    subset = [t for t in trades
                              if regimes_full.get(t['entry_date']) == regime]
                    m = _trade_metrics(subset)
                    analysis_rows.append({
                        'ticker':            ticker,
                        'strategy':          full_label,
                        'regime':            regime,
                        'regime_pct_time':   round(regime_pct.get(regime, 0.0), 1),
                        **m,
                    })

                # ---- Crisis-filtered (causal gating) ----
                gated_sigs = signals_df.copy()
                gated_sigs.loc[gated_sigs.index.isin(volatile_dates), 'signal'] = 0

                replay_gated  = _ReplayStrategy(strat.name + ' [gated]', gated_sigs)
                metrics_gated = make_bt().run(df, replay_gated, verbose=False)

                gated_rows.append({
                    'ticker':           ticker,
                    'strategy':         full_label,
                    'sharpe_ungated':   metrics['sharpe_ratio'],
                    'sharpe_gated':     metrics_gated['sharpe_ratio'],
                    'dd_ungated':       metrics['max_drawdown_pct'],
                    'dd_gated':         metrics_gated['max_drawdown_pct'],
                    'trades_ungated':   metrics['num_trades'],
                    'trades_gated':     metrics_gated['num_trades'],
                    'return_ungated':   round(metrics['total_return_pct'], 2),
                    'return_gated':     round(metrics_gated['total_return_pct'], 2),
                })

    # --- Save CSVs ---
    stamp = date.today().strftime('%Y%m%d')

    analysis_df = pd.DataFrame(analysis_rows)
    csv1 = os.path.join(RESULTS_DIR, f'regime_analysis_{stamp}.csv')
    analysis_df.to_csv(csv1, index=False)
    print(f"\nPer-regime trade breakdown -> {csv1}")

    gated_df = pd.DataFrame(gated_rows)
    csv2 = os.path.join(RESULTS_DIR, f'regime_gated_{stamp}.csv')
    gated_df.to_csv(csv2, index=False)
    print(f"Ungated vs gated comparison -> {csv2}")

    # --- Summary print ---
    print("\n" + "=" * 80)
    print("REGIME TIME DISTRIBUTION — what fraction of days each ticker spends per regime")
    print("=" * 80)
    dist_rows = []
    for ticker in tickers:
        df_t = load_historical_data(ticker, START_DATE, END_DATE)
        reg  = decode_regimes_full(df_t['close'], n_states=HMM_N_STATES).dropna()
        pct  = reg.value_counts(normalize=True).mul(100)
        dist_rows.append({
            'ticker':   ticker,
            'name':     TICKER_NAMES.get(ticker, ticker),
            'ranging':  round(pct.get('ranging',  0.0), 1),
            'trending': round(pct.get('trending', 0.0), 1),
            'volatile': round(pct.get('volatile', 0.0), 1),
        })
    print(pd.DataFrame(dist_rows).to_string(index=False))

    print("\n" + "=" * 80)
    print("CRISIS-FILTERED vs UNGATED — Sharpe / Max Drawdown")
    print("=" * 80)
    print(gated_df.to_string(index=False))


if __name__ == '__main__':
    run_regime_analysis()
