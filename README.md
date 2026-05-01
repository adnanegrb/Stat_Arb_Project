# Stat Arb Bot — BTC/ETH

A self-contained algorithmic trading bot that profits from temporary price divergences between Bitcoin and Ethereum. When the two assets drift apart, the bot bets on their reunion — and pockets the spread.

---

## Results (2021 → 2026 backtest)

| Metric | Value |
|---|---|
| Starting capital | $100 |
| Final capital | $470,937 |
| Total return | +470,837% |
| Sharpe Ratio | 8.40 |
| Max Drawdown | -6.35% |
| Win Rate | 85% |
| Trades | 966 |
| Leverage | x2 |

Survived the 2022 crypto crash (-70% BTC) without blowing up.

---

## How it works

The bot watches the BTC/ETH spread in real time. When the spread stretches too far from its historical norm, it enters a market-neutral trade — long one asset, short the other — and exits once the spread snaps back.

Under the hood: a **Kalman Filter** tracks the dynamic hedge ratio between the two assets, a **rolling Z-score** triggers entries and exits, and **Kelly Criterion** sizes each position.

---

## Project structure

```
├── agent.py          live trading bot (Binance Futures)
├── backtest.py       historical simulation engine
├── requirements.txt
└── results/
    ├── results.txt       full metrics + trade log
    └── equity_curve.png  charts
```

---

## Quickstart

```bash
pip install -r requirements.txt
python backtest.py   # runs simulation, saves results/
python agent.py      # live bot — paper trading by default
```

---

> Past performance does not guarantee future results. The bot runs in simulation mode by default — no real orders are sent until you explicitly enable them.
