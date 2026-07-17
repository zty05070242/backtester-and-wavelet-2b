# Algorithmic Trading Backtester — DSP-Enhanced 2B Rule

While reading through Victor Sperandeo's *Trader Vic* books, I came across his 2B Rule. I wanted to see if I could implement it in code. The most natural encoding was a rolling N-bar maximum for the swing high and a rolling N-bar minimum for the swing low, so that's what I built first.

It was terrible. The rolling window kept firing false signals and felt far worse than doing it by eye. The problem was that raw price data is noisy, and a rolling maximum has no way to distinguish a genuine structural pivot from a single spike bar; it just takes whatever was highest in the last N bars and calls it a swing high. Our eyes filter that noise implicitly, and a rolling window has no equivalent.

Being a tonmeister, I thought I could put my background to use. Reading Stefan Jansen's *Machine Learning for Algorithmic Trading*, he mentions wavelet decomposition. Similar to a Fourier transform, it decomposes the signal into its frequency components. That meant I could apply a low-pass filter to suppress the high-frequency noise, then reconstruct back into the time domain with noise removed. So I applied that to the price series and rebuilt the pivot detector pipeline: wavelet denoising → prominence-filtered peak detection → confirmation gating.

**Headline result:** across 10 commodity futures (Gold, Silver, Crude Oil, Natural Gas, Copper, Wheat, Corn, Soybeans, Coffee, Live Cattle) from 2000–2026, the wavelet-based pivot detector **consistently reduces max drawdown
in 7-8 of 10 markets** at the same risk budget. On Sharpe ratio the picture is mixed. Plain 2B wins on risk-adjusted return in most markets once its baseline is correctly specified (no backwards filters), but Wavelet-2B produces ~40% fewer signals and structurally cleaner entries, which is where the drawdown edge comes from.

---

## What's in the repo

```
backtester.py               all-in / all-out engine
backtester_scaled.py        FIFO tranche scaling engine (pyramid in/out)
chart.py                    Plotly candlestick + auto-overlay
data_loader.py              yfinance → OHLCV DataFrame
position_sizer.py           fixed-fractional risk sizing
wavelet_denoiser.py         pywt denoising (global + causal rolling)
regime_hmm.py               Gaussian HMM regime classifier (ranging / trending / volatile)
run_comparison.py           10-commodity × 2-strategy × 2-backtester harness
run_regime_analysis.py      HMM regime overlay — per-regime trade breakdown + crisis-filtered backtest

strategy_folder/
    _strategy_base_class.py     abstract Strategy base class
    two_b.py                    TwoB — Sperandeo's rule, book → code
    wavelet_two_b.py            WaveletTwoB — DSP pivot detection
    ma_cross.py                 (side experiment) MA crossover
    kalman_cross.py             (side experiment) dual Kalman crossover
    kalman_ma_hybrid.py         (side experiment) Kalman fast / MA slow
    wavelet_ma_cross.py         (side experiment) MA on denoised close
    wavelet_kalman_cross.py     (side experiment) Kalman on denoised close

wavelet_kalman_calibrate.py     (side experiment) spectral calibration of Kalman Q
```

---

## The headline experiment: 2B Rule vs Wavelet-2B

### Sperandeo's 2B Rule — from book knowledge to code application

Victor Sperandeo's failed-breakout reversal pattern (from *Trader Vic — Methods of a Wall Street Master*):

- **Short signal**: price breaks above a prior swing high, then closes back below it within a few bars. The failed breakout is the entry.
- **Long signal**: mirror image — price breaks below a prior swing low, then closes back above it.

Implemented in [strategy_folder/two_b.py](strategy_folder/two_b.py). The book defines "prior swing high/low" by visual inspection. The most common naive encoding for this strategy is the one used in the baseline here: a **rolling N-bar maximum / minimum** of the high/low channel.

### The flaw in the rolling-window encoding

A rolling N-bar max forgets earlier pivots. The "swing high" at bar t is the highest bar of the last N bars, full stop, regardless of whether x bars ago there was a vastly more significant pivot. In a trending market the rolling window keeps printing new highs, and the strategy keeps reading "price broke above the rolling high" as a 2B short
setup. It fires false shorts the whole way up a bull market.

Concrete example from the backtests: during silver's 2010–2011 bull run ($15 → $50), the plain 2B Rule caught a long from $15→$19 in early 2010, then spent the rest of the run firing repeated short signals against a rolling high that was being rewritten every few weeks. The structural lows around $28 that defined the trend's pullbacks were already outside its 20-bar window and invisible to it.

### Wavelet-2B — encoding the human eye

A human chart reader picks out swing highs by visual prominence: a peak stands above its surroundings, the surroundings being smoothed implicitly by the eye. That's a denoise-then-find-local-maxima pipeline. Implemented in [strategy_folder/wavelet_two_b.py](strategy_folder/wavelet_two_b.py):

```
raw close ──► rolling causal wavelet denoise (db6, soft threshold, win=128)
          ──► scipy.signal.find_peaks on the denoised series
          ──► filter peaks by prominence ≥ min_prominence_atr × ATR
          ──► gate by pivot_confirm_bars of follow-through
          ──► resulting pivot levels feed Sperandeo's failed-breakout test
```

Every stage has a DSP analogue:

- **Denoise** suppresses tick-level chop so peaks correspond to structural swings, not single bars.
- **Prominence** (the vertical distance from a peak down to its lowest contour line) is exactly the "does this swing matter" filter the eye applies. Expressing it in ATR multiples makes it self-adapting across assets and regimes.
- **Confirmation lag** is the group-delay analog of pivot detection: a peak is only known to *be* a peak once enough bars have printed lower to its right.

The 2B failed-breakout logic itself is unchanged. Only the *source of the swing-high/low reference* changes.

### Causality

- `rolling_wavelet_denoise()` is causal — denoised[t] depends only on the prior 128 bars.
- A pivot at bar k is only treated as "known" once t ≥ k + `pivot_confirm_bars`.
- One asterisk: `scipy.find_peaks` computes prominence over the full array, so pivot *selection* (whether a peak passes the prominence gate) could in principle shift as later data arrives. The trade decision itself uses only already-confirmed pivots and the current bar, so no future data leaks into signals. Documented inline at the top of [wavelet_two_b.py](strategy_folder/wavelet_two_b.py).

---

## The two backtesters

Both engines share the same Strategy interface and metrics output.

**[backtester.py](backtester.py) — `Backtester` (all-in / all-out)**

- One position at a time, full risk budget per signal.
- Entry on the next bar's open (no look-ahead).
- Exit on stop-loss hit or opposite signal (at the signal bar's open).
- 20× max leverage cap.

**[backtester_scaled.py](backtester_scaled.py) — `BacktesterScaled` (FIFO tranches)**

- Up to `max_tranches` (default 3) independent positions per side.
- Each new signal opens one tranche and closes the *oldest* opposite tranche
  (FIFO scale-out + scale-in).
- Each tranche has its own entry price and stop loss.
- Total risk exposure preserved: `tranche_risk_pct = risk_pct / max_tranches`.

The scaled engine is the more revealing one for this comparison. The plain engine's all-in/all-out compounding amplifies sizing effects. A 26-year backtest on a strongly trending asset can post returns of 22,000% on what is actually a moderately-good signal. The scaled engine flattens that effect and reveals the underlying per-trade economics.

---

## Results — 10 commodity futures, 2000–2026

Run from `python run_comparison.py`. £10,000 initial balance, 2% risk per trade (split across tranches in scaled mode), 1 bp slippage. Full numbers in [results/comparison_20260604.csv](results/comparison_20260604.csv).

### Unscaled backtester (all-in / all-out)

| Commodity | 2B Sharpe | W2B Sharpe | 2B DD | W2B DD | 2B PF | W2B PF |
|-----------|----------:|-----------:|------:|-------:|------:|-------:|
| GC=F (Gold) | **0.77** | 0.51 | -32.5% | **-32.3%** | 2.11 | **2.51** |
| SI=F (Silver) | **0.96** | 0.88 | -22.1% | **-18.6%** | 2.51 | **3.04** |
| CL=F (WTI Crude) | 0.35 | **0.49** | -83.3% | **-54.1%** | 1.15 | **1.42** |
| NG=F (Natural Gas) | **0.41** | 0.33 | -80.3% | **-48.3%** | **1.32** | 1.18 |
| HG=F (Copper) | **0.58** | 0.49 | -37.6% | **-21.7%** | 1.44 | **2.11** |
| ZW=F (Wheat) | **0.71** | 0.48 | **-44.8%** | -48.4% | **1.43** | 1.26 |
| ZC=F (Corn) | **0.41** | 0.27 | -62.8% | **-51.0%** | 1.14 | 1.17 |
| ZS=F (Soybeans) | **0.59** | 0.58 | -57.8% | **-51.5%** | 1.15 | **1.58** |
| KC=F (Coffee) | **0.68** | 0.37 | -64.7% | **-38.7%** | 1.57 | **1.95** |
| LE=F (Live Cattle) | **0.72** | 0.43 | -76.4% | **-59.1%** | **1.50** | 1.13 |

Sharpe: **2B wins 9/10** — Wavelet wins 1/10 (Crude Oil only).
Max drawdown: **Wavelet wins 8/10** — lower drawdown in all markets except Wheat.
Profit factor: **Wavelet wins 6/10**.

### Scaled backtester (FIFO tranche scaling, 3 tranches)

| Commodity | 2B Sharpe | W2B Sharpe | 2B DD | W2B DD | 2B PF | W2B PF |
|-----------|----------:|-----------:|------:|-------:|------:|-------:|
| GC=F (Gold) | **0.58** | 0.57 | **-13.8%** | -16.9% | 2.30 | **2.83** |
| SI=F (Silver) | **1.02** | 0.80 | -10.4% | **-9.7%** | **3.03** | 2.78 |
| CL=F (WTI Crude) | 0.38 | **0.64** | -50.7% | **-29.8%** | 1.27 | **1.77** |
| NG=F (Natural Gas) | **0.42** | 0.39 | -33.4% | **-25.0%** | **1.60** | 1.31 |
| HG=F (Copper) | 0.44 | **0.50** | -18.2% | **-9.0%** | 1.76 | **3.05** |
| ZW=F (Wheat) | **0.52** | 0.49 | **-21.8%** | -30.4% | **1.59** | 1.43 |
| ZC=F (Corn) | 0.37 | **0.42** | -25.5% | **-21.1%** | 1.20 | **1.56** |
| ZS=F (Soybeans) | **0.58** | 0.54 | -30.1% | **-24.0%** | 1.32 | **2.10** |
| KC=F (Coffee) | **0.77** | 0.37 | **-31.4%** | -35.0% | **1.74** | 1.45 |
| LE=F (Live Cattle) | **0.71** | 0.44 | -38.5% | **-31.8%** | **1.88** | 1.27 |

Sharpe: **2B wins 7/10** — Wavelet wins 3/10 (Crude Oil, Copper, Corn).
Max drawdown: **Wavelet wins 7/10**.
Profit factor: **Wavelet wins 6/10**.

### Key findings

An earlier version used volume and ATR breakout filters on the 2B baseline. They are both counterproductive for a reversal strategy (high-volume breakouts are exactly the ones you don't want to fade). Removing them lifted TwoB's Sharpe noticeably. The table above is the corrected, honest baseline.

Wavelet-2B's main edge is drawdown: it fires ~40% fewer signals, which limits exposure during losing stretches. That holds even in markets where it loses on Sharpe. Plain 2B stays more active and compounds more aggressively. Higher
Sharpe in most markets, higher drawdown in almost all of them. Genuine trade-off, not a clear winner.

Crude Oil is the clearest case for Wavelet-2B. The rolling-window baseline struggles there (Sharpe 0.35–0.38, drawdown -50% to -83%) because Crude's noisy price action constantly rewrites the rolling high/low and generates
false breakouts. The prominence filter screens most of those out.

Natural Gas is the hard case for both strategies. Regime changes and seasonality make failed-breakout logic structurally awkward on NG=F; neither pivot method handles it well.

---

## Side experiments

My early stage exploration into wavelet/Kalman preprocessing for crossover strategies, kept in the repo for completeness but no longer the focus of the write-up:

- [strategy_folder/ma_cross.py](strategy_folder/ma_cross.py) — baseline MA crossover.
- [strategy_folder/kalman_cross.py](strategy_folder/kalman_cross.py), [strategy_folder/kalman_ma_hybrid.py](strategy_folder/kalman_ma_hybrid.py) — Kalman as a smoother in crossover form.
- [strategy_folder/wavelet_ma_cross.py](strategy_folder/wavelet_ma_cross.py), [strategy_folder/wavelet_kalman_cross.py](strategy_folder/wavelet_kalman_cross.py) — same crossovers run on wavelet-denoised close.
- [wavelet_kalman_calibrate.py](wavelet_kalman_calibrate.py) — offline DWT spectral analysis to derive Kalman process-noise Q from band SNR rather than hand-tuning. Demonstrates the spectral pipeline but uses full-series look-ahead — not live-tradeable as-is.

Summary of what those experiments showed: stacking a wavelet denoiser in front of an MA crossover is redundant (both are low-pass filters with overlapping bands → premature exits), while stacking it in front of a Kalman crossover is complementary (Kalman's adaptive bandwidth fills a different role) but only buys risk-adjusted improvement, not raw return. That conclusion is what pointed the project toward 2B Rule: a strategy where the DSP role is *structural pivot identification*, not bandlimiting the input signal.

---

## Regime analysis — HMM overlay

Real systematic funds almost never run a strategy naked, instead, they run a regime classifier on top that controls when the strategy is allowed to trade. [regime_hmm.py](regime_hmm.py) fits a 3-state Gaussian HMM on two features:
- **20-day rolling realized volatility** (std of daily log returns)
- **20-day cumulative log return** (directional component)

States are labelled by mean realised vol ascending: **ranging** (low) / **trending** (mid) / **volatile** (high). Both a retrospective full-series decode (for post-hoc analysis) and a causal rolling version (refit quarterly on a 5-year trailing window, safe for signal gating) are implemented.

[run_regime_analysis.py](run_regime_analysis.py) does two things:

**1. Per-regime trade breakdown.** For each of the 10 markets, it splits the trade log from each backtest by the regime active at entry date. This answers the question the aggregate Sharpe table can't: in which market conditions does each strategy actually earn money, and in which does it bleed?

The natural gas result is the clearest example of why this matters. NG=F posts the weakest numbers of any market across both strategies and both backtesters. The regime breakdown shows why: natural gas spends roughly 60% of its history in the volatile state (sharp directional moves with frequent regime shifts) which is structurally the worst environment for a failed-breakout reversal. The strategy fires, gets stopped out on the continuation, and fires again. The bad performance isn't a strategy problem, but a regime mismatch. By knowing this, I converted a confusing result into a research finding.

**2. Crisis-filtered backtest.** Signals in the volatile regime are zeroed out using the causal rolling HMM (no look-ahead). The comparison shows how much of each strategy's drawdown comes specifically from trading through high-volatility regimes, and whether crisis-filtering improves risk-adjusted returns at the cost of reduced trade count.

### Causality

The full-series HMM (used only for the per-regime breakdown) is not causal. It sees the full 26-year history before labelling any bar, so it can't be used live. The rolling version is strictly causal: the model fit at time t uses only bars up to t, and the state label for bar t is produced without any forward-looking information. All crisis-filtering in the backtest uses the causal labels only.

---

## How to run

```bash
pip install -r requirements.txt

# Full 10-commodity × 2-strategy × 2-backtester comparison (saves CSV + per-ticker equity HTML)
python run_comparison.py

# HMM regime overlay — per-regime trade breakdown + crisis-filtered backtest
# (saves regime_analysis_YYYYMMDD.csv and regime_gated_YYYYMMDD.csv)
python run_regime_analysis.py

# Single-strategy smoke test with chart
python -m strategy_folder.wavelet_two_b
python -m strategy_folder.two_b

# Wavelet denoising demo (S&P 500 returns, global vs causal)
python wavelet_denoiser.py
```

---

## Architecture quick-reference

### Strategy interface

All strategies inherit from
[strategy_folder/_strategy_base_class.py](strategy_folder/_strategy_base_class.py):

```python
class Strategy(ABC):
    def set_data(self, data: pd.DataFrame): ...
    def generate_signals(self) -> pd.DataFrame: ...   # must return df with 'signal' column
    def get_signals(self) -> pd.DataFrame: ...
```

`generate_signals()` returns the full OHLCV DataFrame plus a `signal` column (`1` = long, `-1` = short, `0` = flat) and optionally a `stop_loss` column. If `stop_loss` is present the backtester uses it; otherwise it falls back to
bar low/high.

### Position sizing

[position_sizer.py](position_sizer.py) — fixed-fractional risk sizing:

```
units = (account_balance × risk_pct) / |entry_price − stop_loss|
```

### Wavelet denoising

[wavelet_denoiser.py](wavelet_denoiser.py) — Donoho-Johnstone universal
threshold:

1. Discrete wavelet transform decomposes price into a coarse approximation plus a pyramid of detail bands.
2. Estimate noise σ from the finest detail band via MAD: `σ = median(|cD₁|) / 0.6745`.
3. Apply threshold `σ · √(2 log n) · threshold_scale` to every detail band (soft shrinkage).
4. Inverse DWT reconstructs the cleaned price.

The causal `rolling_wavelet_denoise()` slides a 128-bar (Wavelet-2B) or 252-bar (older crossover strategies) window and keeps only the final reconstructed value at each step. No future data feeds the estimate at t.

### Chart

[chart.py](chart.py) — `plot_signals(strategy)` renders a Plotly candlestick chart with buy/sell markers and auto-overlays any DataFrame column not in `{open, high, low, close, volume, signal, avg_volume, atr, stop_loss}`. So
strategies surface their internal state (e.g. `wavelet_close`, `swing_high`, `swing_low`) just by storing it on the DataFrame.

---

## Limitations

- **Wavelet rolling-window cost.** `rolling_wavelet_denoise()` is O(window × N). Runs in a few seconds per 26-year daily history; an intraday version would need a faster causal pipeline (e.g. SWT or a fixed-level partial DWT).
- **Prominence is computed globally.** `scipy.find_peaks` reads the full denoised array when computing prominence. Pivot *selection* is therefore not strictly causal, even though the trade decision is. A fully-causal peak finder is straightforward to add.
- **Transaction costs.** 1 bp slippage per fill only. No commissions, borrow costs for shorts, exchange fees, or roll costs (relevant for futures).
- **Single-asset backtests.** All-in or all-tranche per ticker. Single-asset time-series forcasting. No portfolio effects, correlation-aware sizing, or cross-sectional allocation.
- **Data source.** `yfinance` daily continuous futures contracts. Adequate for a portfolio-piece backtest, not production quality. NG=F and LE=F in particular can have sparser histories on Yahoo.

---

## Dependencies

```
yfinance
pandas
numpy
plotly
hmmlearn     (used by regime_hmm.py for GaussianHMM)
pykalman
PyWavelets   (imported as pywt)
scipy        (used by wavelet_two_b.py for find_peaks)
```
