import asyncio
import logging
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
import ccxt.async_support as ccxt
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

API_KEY    = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

PAIR_A     = "BTC/USDT"
PAIR_B     = "ETH/USDT"
TIMEFRAME  = "1m"
LOOKBACK   = 60
Z_ENTRY    = 2.0
Z_EXIT     = 0.5
KELLY_FRAC = 0.60
LEVERAGE   = 2
CAPITAL    = 100.0


@dataclass
class Position:
    active:  bool  = False
    side:    str   = ""
    size_a:  float = 0.0
    size_b:  float = 0.0
    entry_z: float = 0.0


def kalman_hedge_ratio(price_a: np.ndarray, price_b: np.ndarray) -> np.ndarray:
    obs_mat = np.vstack([price_b, np.ones(len(price_b))]).T[:, np.newaxis, :]
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
    state_means, _ = kf.filter(price_a[:, np.newaxis])
    return state_means[:, 0]


def compute_zscore(price_a: np.ndarray, price_b: np.ndarray) -> float:
    log_a  = np.log(price_a)
    log_b  = np.log(price_b)
    betas  = kalman_hedge_ratio(log_a, log_b)
    spread = log_a - betas * log_b
    s      = pd.Series(spread)
    mu     = s.rolling(LOOKBACK).mean().iloc[-1]
    sigma  = s.rolling(LOOKBACK).std().iloc[-1]
    if sigma == 0:
        return 0.0
    return float((spread[-1] - mu) / sigma)


def kelly_size(z: float, capital: float) -> float:
    W = 0.88
    R = 4.7
    f = (W - (1 - W) / R) * KELLY_FRAC
    return min(capital * f, capital * 0.70)


class Exchange:
    def __init__(self):
        self.ex = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    async def fetch_ohlcv(self, symbol: str, tf: str, limit: int) -> np.ndarray:
        bars = await self.ex.fetch_ohlcv(symbol, tf, limit=limit)
        return np.array([b[4] for b in bars], dtype=float)

    async def place_order(self, symbol: str, side: str, amount: float):
        log.info(f"ORDER | {side.upper()} {amount:.6f} {symbol}")
        # return await self.ex.create_market_order(symbol, side, amount)

    async def close(self):
        await self.ex.close()


async def run():
    exchange = Exchange()
    pos      = Position()
    log.info(f"Bot started | {PAIR_A}/{PAIR_B} | Capital: ${CAPITAL} | Leverage: x{LEVERAGE}")

    try:
        while True:
            prices_a = await exchange.fetch_ohlcv(PAIR_A, TIMEFRAME, LOOKBACK + 10)
            prices_b = await exchange.fetch_ohlcv(PAIR_B, TIMEFRAME, LOOKBACK + 10)

            if len(prices_a) < LOOKBACK or len(prices_b) < LOOKBACK:
                await asyncio.sleep(10)
                continue

            z = compute_zscore(prices_a, prices_b)
            log.info(f"Z-score: {z:.3f} | Position: {pos.side if pos.active else 'none'}")

            if not pos.active:
                if z > Z_ENTRY:
                    size = kelly_size(z, CAPITAL)
                    await exchange.place_order(PAIR_A, "sell", size / prices_a[-1])
                    await exchange.place_order(PAIR_B, "buy",  size / prices_b[-1])
                    pos = Position(active=True, side="short_spread",
                                   size_a=size/prices_a[-1],
                                   size_b=size/prices_b[-1], entry_z=z)
                    log.info(f"ENTRY short_spread | z={z:.2f} | size=${size:.2f}")

                elif z < -Z_ENTRY:
                    size = kelly_size(z, CAPITAL)
                    await exchange.place_order(PAIR_A, "buy",  size / prices_a[-1])
                    await exchange.place_order(PAIR_B, "sell", size / prices_b[-1])
                    pos = Position(active=True, side="long_spread",
                                   size_a=size/prices_a[-1],
                                   size_b=size/prices_b[-1], entry_z=z)
                    log.info(f"ENTRY long_spread | z={z:.2f} | size=${size:.2f}")

            else:
                if abs(z) < Z_EXIT:
                    if pos.side == "short_spread":
                        await exchange.place_order(PAIR_A, "buy",  pos.size_a)
                        await exchange.place_order(PAIR_B, "sell", pos.size_b)
                    else:
                        await exchange.place_order(PAIR_A, "sell", pos.size_a)
                        await exchange.place_order(PAIR_B, "buy",  pos.size_b)
                    log.info(f"EXIT | z={z:.2f} | was {pos.entry_z:.2f}")
                    pos = Position()

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        log.info("Bot stopped.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(run())
