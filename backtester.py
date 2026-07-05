import pandas as pd
import numpy as np
from typing import Dict
from position_sizer import calculate_position_size


class Backtester:
    """
    Main backtesting engine.
    Entries on signal, exits on opposite signal.
    Position sizing via position_sizer().
    """

    def __init__(self, initial_balance: float = 6000.0, risk_pct: float = 0.02, slippage_pct: float = 0.0):
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct    # Combined spread + slippage cost per fill.
        self._reset()                       # Use a reset method so run() can call it cleanly.

    def _reset(self):
        """Reset all state — called before each backtest run."""
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.entry_date = None
        self.trade_open = False
        self.current_direction = 0          # 1 = long, -1 = short
        self.trades = []
        self.equity_curve = []

    def run(self, data: pd.DataFrame, strategy, verbose: bool = True) -> Dict:
        """Run the backtest on a dataset using a given strategy.

        data: OHLCV DataFrame from data_loader.
        strategy: any Strategy subclass instance.
        verbose: print trade-by-trade output if True.

        returns a dict of performance metrics and the trade log.
        """
        self._reset()

        strategy.set_data(data)
        df = strategy.generate_signals()

        if verbose:
            print(f"Strategy       : {strategy.name}")
            print(f"Initial balance: £{self.initial_balance:,.2f}")
            print(f"Risk per trade : {self.risk_pct * 100:.1f}%")
            print(f"Slippage       : {self.slippage_pct * 100:.3f}%\n")

        # pending_signal holds the signal from the previous bar, so we enter on the NEXT bar's open instead of the signal bar's close.
        # This removes look-ahead bias: you see the close, decide to trade, and the earliest you can realistically execute is the next bar's open.
        # Every variable with pending_ in the name is a one-day delay. Set today, used tomorrow.
        pending_signal = 0
        pending_sl = 0.0                    # Signal bar's low (long) or high (short).
        pending_exit = False

        # If the strategy provides a 'stop_loss' column, use it instead of bar low/high. Falls back to bar low/high when the column is absent or the value is NaN.
        has_custom_sl = 'stop_loss' in df.columns

# ========================================================================================== Backtesting loop starts here. ==========================================================================================
        for date, row in df.iterrows():     

            # ============================================================ EXIT RULE ============================================================
            # ============ Check: sl_hit or pending_exit or else ============
            if self.trade_open:
                sl_hit:bool = (self.current_direction == 1 and row['low'] <= self.stop_loss) or (self.current_direction == -1 and row['high'] >= self.stop_loss)
                # sl_hit = True if (going long AND the low hits stop loss price) OR (going short AND the high hits stop loss price)
                if sl_hit:
                    if self.current_direction == 1:     # Long position: exit immediately
                        exit_price = self.stop_loss * (1 - self.slippage_pct)
                    else:                               # Short position: exit immediately
                        exit_price = self.stop_loss * (1 + self.slippage_pct)
            
                elif pending_exit:          # Pending exit is set True/False. See line 114. If pending_exit = True, there is an opposite signal.
                    if self.current_direction == 1:     # Long position: exit at the next bar's open (ensure no lookahead bias).
                        exit_price = row['open'] * (1 - self.slippage_pct)
                    else:                               # Short position: exit at the next bar's open (ensure no lookahead bias).
                        exit_price = row['open'] * (1 + self.slippage_pct)
                
                else:       # If neither stop loss hit or opposite signal, there's no exit price. We keep the position.
                    exit_price = None

                if exit_price is not None:      # If there is an exit price (we close a position), we calculate and update the self.trades[] list.

                    # Multiply by direction: long profits when price rises, short when it falls
                    pnl = (exit_price - self.entry_price) * self.position * self.current_direction
                    pnl_pct = (pnl / self.current_balance) * 100   # % of balance AT entry, not initial

                    self.current_balance += pnl

                    self.trades.append({
                        'entry_date': self.entry_date,
                        'exit_date': date,
                        'direction': "long" if self.current_direction == 1 else "short",
                        'entry_price': self.entry_price,
                        'exit_price': round(exit_price, 2),
                        'position_size': self.position,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })

                    if verbose:
                        direction_label = "LONG" if self.current_direction == 1 else "SHORT"
                        exit_reason = "SL HIT" if sl_hit else "SIGNAL"
                        print(f"Balance: £{self.current_balance:,.2f} | CLOSED {direction_label} on {date.date()} | {self.position:.4f} units @ £{exit_price:.2f} | PnL: £{pnl:.2f} ({pnl_pct:.2f}%) | {exit_reason}")

                    self.trade_open = False
                    self.position = 0.0
                    self.current_direction = 0

            # ============ Opposite Signal ============
            if self.trade_open:
                opposite_signal:bool = (self.current_direction == 1 and row['signal'] == -1) or (self.current_direction == -1 and row['signal'] == 1)
                pending_exit = opposite_signal      # Sets whether pendind_exit is True or False based on whether or not opposite signal appears.
            else:
                pending_exit = False

            # ============================================================ ENTRY RULE ============================================================
            if not self.trade_open and pending_signal != 0 and self.current_balance > 0:
                direction = pending_signal

                # Entry at this bar's open, with slippage making the fill worse
                if direction == 1:
                    entry_price = row['open'] * (1 + self.slippage_pct)   # Long position: slips higher
                else:
                    entry_price = row['open'] * (1 - self.slippage_pct)   # Short position: slips lower

                # Stop loss: from the SIGNAL bar (t), not the entry bar (t+1). We already know this level when we decide to trade.
                stop_loss = pending_sl          # pending_sl is set below. See line 155.

                # Skip trade if entry price equals stop loss — no room for a valid SL.
                if stop_loss == entry_price:
                    pending_signal = 0
                    self.equity_curve.append({'date': date, 'balance': self.current_balance})
                    continue

                # Call the position sizer to calculate appropriate units to trade.
                sizing = calculate_position_size(account_balance=self.current_balance, risk_pct=self.risk_pct, entry_price=entry_price, stop_loss_price=stop_loss)

                max_units = (self.current_balance * 20) / entry_price  # 20x max leverage cap. Can't go more than this. 
                self.position = min(sizing['units_to_trade'], max_units)
                self.entry_price = entry_price
                self.stop_loss = stop_loss
                self.entry_date = date
                self.trade_open = True
                self.current_direction = direction

                if verbose:
                    direction_label = sizing['direction'].upper()
                    print(f"Balance: £{self.current_balance:,.2f} | OPENED {direction_label} on {date.date()} | {self.position:.4f} units @ £{entry_price:.2f} | SL: £{stop_loss:.2f}")

            # Capture this bar's signal and SL level for execution on the NEXT bar
            pending_signal = int(row['signal'])
            if pending_signal == 1:
                pending_sl = row['stop_loss'] if has_custom_sl and pd.notna(row['stop_loss']) else row['low']
            elif pending_signal == -1:
                pending_sl = row['stop_loss'] if has_custom_sl and pd.notna(row['stop_loss']) else row['high']

            # === Record equity AFTER processing this bar ===
            self.equity_curve.append({
                'date': date,
                'balance': self.current_balance
            })

        # === Close any open position at end of data ===
        if self.trade_open:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * self.position * self.current_direction
            self.current_balance += pnl
            self.equity_curve.append({'date': df.index[-1], 'balance': self.current_balance})
            if verbose:
                print(f"\nForce-closed open position at end of data | PnL: £{pnl:.2f}")

        # === Calculate performance metrics ===
        metrics = self._calculate_metrics()

        if verbose:
            print(f"\n{'='*40}")
            print(f"Final balance  : £{metrics['final_balance']:,.2f}")
            print(f"Total return   : {metrics['total_return_pct']:.2f}%")
            print(f"Total trades   : {metrics['num_trades']}")
            print(f"Win rate       : {metrics['win_rate_pct']:.1f}%")
            print(f"Sharpe ratio   : {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown   : {metrics['max_drawdown_pct']:.2f}%")
            print(f"Profit factor  : {metrics['profit_factor']}")
            print(f"Expectancy     : £{metrics['expectancy']:,.2f} per trade")
            print(f"Avg win        : £{metrics['avg_win']:,.2f}")
            print(f"Avg loss       : £{metrics['avg_loss']:,.2f}")
            print(f"Largest win    : £{metrics['largest_win']:,.2f}")
            print(f"Largest loss   : £{metrics['largest_loss']:,.2f}")

        return metrics

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics from completed trades and equity curve."""

        total_return_pct = (
            (self.current_balance - self.initial_balance) / self.initial_balance
        ) * 100

        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]  # list of trades where pnl > 0
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

        # Sharpe ratio: average daily return divided by std deviation of daily returns
        # Annualised by multiplying by sqrt(252) — 252 trading days in a year
        equity_df = pd.DataFrame(self.equity_curve)         # converts equity curve list into a DataFrame
        daily_returns = equity_df['balance'].pct_change().dropna()  # pct_change() calculates % change between each row

        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown: largest peak-to-trough fall in balance
        rolling_max = equity_df['balance'].cummax()         # cummax() = running maximum up to each point
        drawdown = (equity_df['balance'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100             # most negative value = worst drawdown

        # Profit factor: gross profit / gross loss
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss   = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        # Average win and average loss
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        avg_win  = round(gross_profit / len(winning_trades), 2) if winning_trades else 0
        avg_loss = round(gross_loss   / len(losing_trades),  2) if losing_trades  else 0

        # Expectancy: average PnL per trade
        expectancy = round(sum(t['pnl'] for t in self.trades) / num_trades, 2) if num_trades > 0 else 0

        # Largest win and largest loss
        largest_win  = round(max((t['pnl'] for t in self.trades), default=0), 2)
        largest_loss = round(min((t['pnl'] for t in self.trades), default=0), 2)

        return {
            'final_balance': self.current_balance,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate_pct': round(win_rate * 100, 1),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


if __name__ == "__main__":
    from data_loader import load_historical_data    
    from strategy_folder.kalman_cross import KalmanCrossover
    from strategy_folder.ma_cross import MovingAverageCrossover
    from strategy_folder.kalman_ma_hybrid import KalmanMAHybrid
    from strategy_folder.wavelet_kalman_cross import WaveletKalmanCrossover
    from strategy_folder.wavelet_ma_cross import WaveletMACrossover
    from strategy_folder.two_b import TwoB

    df = load_historical_data("SI=F", "2000-01-01", "2026-04-15")
    strategy = TwoB(lookback=20, confirmation_days=3)
    backtester = Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001)
    results = backtester.run(df, strategy, verbose=True)

    # --- Interactive Chart ---
    from chart import plot_signals
    plot_signals(strategy)