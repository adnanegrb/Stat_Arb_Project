import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ccxt
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from dataclasses import dataclass
from typing import List, Tuple

PAIR_A      = "BTC/USDT"
PAIR_B      = "ETH/USDT"
TIMEFRAME   = "1h"
LOOKBACK    = 60
Z_ENTRY     = 2.0
Z_EXIT      = 0.5
KELLY_FRAC  = 0.60
CAPITAL     = 100.0
FEES        = 0.001
LEVERAGE    = 2
N_CANDLES   = 43800
START_DATE  = "2021-01-01T00:00:00Z"
RESULTS_DIR = "results"


def fetch_prices(symbol: str, timeframe: str, limit: int) -> pd.Series:
    exchange = ccxt.binance({"enableRateLimit": True})
    all_bars = []
    since    = exchange.parse8601(START_DATE)
    print(f"Fetching {symbol}...")
    while len(all_bars) < limit:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not bars:
            break
        all_bars += bars
        since = bars[-1][0] + 1
        print(f"   {len(all_bars)} candles...", end="\r")
    all_bars = all_bars[:limit]
    index    = pd.to_datetime([b[0] for b in all_bars], unit="ms")
    closes   = [b[4] for b in all_bars]
    s        = pd.Series(closes, index=index)
    print(f"OK {symbol} -- {len(s)} candles ({s.index[0].date()} -> {s.index[-1].date()})")
    return s


def kalman_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    obs_mat = np.vstack([log_b, np.ones(len(log_b))]).T[:, np.newaxis, :]
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[1.0, 0.0],
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=1e-4 * np.eye(2),
    )
    state_means, _ = kf.filter(log_a[:, np.newaxis])
    return state_means[:, 0]


def compute_signals(price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
    log_a  = np.log(price_a.values)
    log_b  = np.log(price_b.values)
    print("Running Kalman Filter...")
    betas  = kalman_hedge_ratio(log_a, log_b)
    spread = log_a - betas * log_b
    s      = pd.Series(spread, index=price_a.index)
    mu     = s.rolling(LOOKBACK).mean()
    sigma  = s.rolling(LOOKBACK).std()
    zscore = (s - mu) / sigma
    df = pd.DataFrame({
        "price_a": price_a.values,
        "price_b": price_b.values,
        "beta":    betas,
        "spread":  spread,
        "zscore":  zscore.values,
    }, index=price_a.index)
    return df.dropna()


@dataclass
class Trade:
    entry_idx: int
    exit_idx:  int
    side:      str
    entry_z:   float
    exit_z:    float
    pnl:       float
    pnl_pct:   float


def kelly_size(z: float, capital: float) -> float:
    W = 0.88
    R = 4.7
    f = (W - (1 - W) / R) * KELLY_FRAC
    return min(capital * f, capital * 0.70)


def run_backtest(df: pd.DataFrame) -> Tuple[pd.Series, List[Trade]]:
    equity       = np.zeros(len(df))
    equity[0]    = CAPITAL
    capital      = CAPITAL
    trades       = []
    in_position  = False
    side         = ""
    entry_idx    = 0
    entry_z      = 0.0
    entry_size   = 0.0
    entry_spread = 0.0
    zscores      = df["zscore"].values
    spreads      = df["spread"].values

    for i in range(1, len(df)):
        z         = zscores[i]
        equity[i] = capital

        if not in_position:
            if z > Z_ENTRY:
                side         = "short_spread"
                entry_idx    = i
                entry_z      = z
                entry_spread = spreads[i]
                entry_size   = kelly_size(z, capital)
                in_position  = True
            elif z < -Z_ENTRY:
                side         = "long_spread"
                entry_idx    = i
                entry_z      = z
                entry_spread = spreads[i]
                entry_size   = kelly_size(z, capital)
                in_position  = True
        else:
            if abs(z) < Z_EXIT:
                exit_spread = spreads[i]
                if side == "short_spread":
                    raw_pnl = (entry_spread - exit_spread) * entry_size * LEVERAGE
                else:
                    raw_pnl = (exit_spread - entry_spread) * entry_size * LEVERAGE
                fee       = entry_size * FEES * 4 * LEVERAGE
                net_pnl   = raw_pnl - fee
                capital  += net_pnl
                equity[i] = capital
                trades.append(Trade(
                    entry_idx=entry_idx, exit_idx=i,
                    side=side, entry_z=entry_z, exit_z=z,
                    pnl=net_pnl, pnl_pct=net_pnl / entry_size * 100,
                ))
                in_position = False

    eq = pd.Series(equity, index=df.index)
    eq = eq.replace(0, np.nan).ffill().fillna(CAPITAL)
    return eq, trades


def compute_metrics(equity: pd.Series, trades: List[Trade]) -> dict:
    returns       = equity.pct_change().dropna()
    sharpe        = returns.mean() / returns.std() * np.sqrt(8760)
    roll_max      = equity.cummax()
    drawdown      = (equity - roll_max) / roll_max
    max_dd        = drawdown.min()
    pnls          = [t.pnl for t in trades]
    win_rate      = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
    avg_win       = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss      = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    return {
        "Total Return (%)":  round((equity.iloc[-1] / CAPITAL - 1) * 100, 2),
        "Final Capital ($)": round(equity.iloc[-1], 2),
        "Sharpe Ratio":      round(sharpe, 3),
        "Max Drawdown (%)":  round(max_dd * 100, 2),
        "Nb Trades":         len(trades),
        "Win Rate (%)":      round(win_rate * 100, 2),
        "Avg Win ($)":       round(avg_win, 4),
        "Avg Loss ($)":      round(avg_loss, 4),
        "Profit Factor":     round(profit_factor, 3),
    }


def save_results_txt(metrics: dict, trades: List[Trade], pvalue: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "results.txt")
    with open(path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("  STAT ARB BTC/ETH | 2021-2026 | Levier x2\n")
        f.write("=" * 50 + "\n\n")
        f.write("METRICS\n")
        f.write("-" * 30 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<22} {v}\n")
        f.write(f"\n  Cointegration p-value : {pvalue:.4f} {'PASS' if pvalue < 0.05 else 'FAIL'}\n")
        f.write(f"  Capital initial       : ${CAPITAL}\n")
        f.write(f"  Leverage              : x{LEVERAGE}\n")
        f.write(f"  Kelly fraction        : {KELLY_FRAC}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        f.write("TRADE LOG\n")
        f.write("-" * 30 + "\n")
        for i, t in enumerate(trades):
            status = "WIN " if t.pnl > 0 else "LOSS"
            f.write(f"  [{i+1:04d}] {t.side:<15} z_in={t.entry_z:+.2f}  z_out={t.exit_z:+.2f}  PnL=${t.pnl:+.4f}  {status}\n")
    print(f"Results saved -> {path}")


def plot_results(df: pd.DataFrame, equity: pd.Series, trades: List[Trade], metrics: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1.2, 1], hspace=0.4)
    C   = {"bg": "#0d1117", "grid": "#21262d", "green": "#3fb950",
           "red": "#f85149", "blue": "#58a6ff", "text": "#e6edf3", "yellow": "#d29922"}

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C["bg"])
    ax1.plot(equity.index, equity.values, color=C["green"], linewidth=1.5)
    ax1.axhline(CAPITAL, color=C["text"], linestyle="--", alpha=0.3, linewidth=1)
    ax1.fill_between(equity.index, CAPITAL, equity.values,
                     where=(equity.values >= CAPITAL), alpha=0.15, color=C["green"])
    ax1.fill_between(equity.index, CAPITAL, equity.values,
                     where=(equity.values < CAPITAL), alpha=0.15, color=C["red"])
    for t in trades:
        color = C["green"] if t.pnl > 0 else C["red"]
        ax1.axvline(df.index[t.entry_idx], color=color, alpha=0.12, linewidth=0.5)
    ax1.set_title("Equity Curve - Stat Arb BTC/ETH | 2021-2026 | Levier x2 | Kelly 60%",
                  color=C["text"], fontsize=13, pad=10)
    ax1.set_ylabel("Capital ($)", color=C["text"])
    ax1.tick_params(colors=C["text"])
    ax1.grid(color=C["grid"], linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(C["grid"])
    metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    ax1.text(0.01, 0.97, metrics_str, transform=ax1.transAxes,
             fontsize=8, verticalalignment="top", color=C["text"],
             bbox=dict(boxstyle="round", facecolor=C["grid"], alpha=0.8),
             fontfamily="monospace")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C["bg"])
    ax2.plot(df.index, df["zscore"].values, color=C["blue"], linewidth=0.5, alpha=0.9)
    ax2.axhline(Z_ENTRY,  color=C["red"],    linestyle="--", linewidth=1, alpha=0.7, label=f"+{Z_ENTRY}")
    ax2.axhline(-Z_ENTRY, color=C["green"],  linestyle="--", linewidth=1, alpha=0.7, label=f"-{Z_ENTRY}")
    ax2.axhline(Z_EXIT,   color=C["yellow"], linestyle=":",  linewidth=0.8, alpha=0.5)
    ax2.axhline(-Z_EXIT,  color=C["yellow"], linestyle=":",  linewidth=0.8, alpha=0.5)
    ax2.fill_between(df.index, Z_ENTRY, df["zscore"].values,
                     where=(df["zscore"].values > Z_ENTRY), alpha=0.2, color=C["red"])
    ax2.fill_between(df.index, -Z_ENTRY, df["zscore"].values,
                     where=(df["zscore"].values < -Z_ENTRY), alpha=0.2, color=C["green"])
    ax2.set_title("Z-Score du Spread", color=C["text"], fontsize=11)
    ax2.set_ylabel("Z-Score", color=C["text"])
    ax2.tick_params(colors=C["text"])
    ax2.grid(color=C["grid"], linewidth=0.5)
    ax2.legend(fontsize=8, facecolor=C["grid"], labelcolor=C["text"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(C["grid"])

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(C["bg"])
    pnls       = [t.pnl for t in trades]
    t_idx      = [df.index[t.exit_idx] for t in trades]
    bar_colors = [C["green"] if p > 0 else C["red"] for p in pnls]
    ax3.bar(t_idx, pnls, color=bar_colors, alpha=0.8, width=pd.Timedelta(hours=5))
    ax3.axhline(0, color=C["text"], linewidth=0.8, alpha=0.5)
    ax3.set_title("PnL par Trade ($)", color=C["text"], fontsize=11)
    ax3.set_ylabel("PnL ($)", color=C["text"])
    ax3.tick_params(colors=C["text"])
    ax3.grid(color=C["grid"], linewidth=0.5)
    for spine in ax3.spines.values():
        spine.set_edgecolor(C["grid"])

    out = os.path.join(RESULTS_DIR, "equity_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"Plot saved -> {out}")
    plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("  STAT ARB BACKTEST | BTC/ETH | 2021-2026 | Levier x2")
    print("=" * 55)

    price_a = fetch_prices(PAIR_A, TIMEFRAME, N_CANDLES)
    price_b = fetch_prices(PAIR_B, TIMEFRAME, N_CANDLES)

    df_raw  = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    price_a = df_raw["a"]
    price_b = df_raw["b"]

    _, pvalue, _ = coint(np.log(price_a.values), np.log(price_b.values))
    print(f"Cointegration p-value: {pvalue:.4f} {'PASS' if pvalue < 0.05 else 'FAIL'}")

    df             = compute_signals(price_a, price_b)
    print(f"Signal DataFrame: {len(df)} rows")

    print("Running backtest...")
    equity, trades = run_backtest(df)

    metrics = compute_metrics(equity, trades)
    print("\n" + "=" * 35)
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print("=" * 35)

    save_results_txt(metrics, trades, pvalue)
    plot_results(df, equity, trades, metrics)
