# Stat Arb Bot — BTC/ETH

Un bot de trading algorithmique qui exploite la corrélation entre Bitcoin et Ethereum. Quand les deux actifs s'écartent anormalement l'un de l'autre, le bot parie sur leur retour à la normale. Simple en théorie, redoutable en pratique.

---

## Résultats backtest (2021 → 2026)

| | |
|---|---|
| Capital de départ | 100$ |
| Capital final | 470 937$ |
| Return total | +470 837% |
| Sharpe Ratio | 8.40 |
| Max Drawdown | -6.35% |
| Win Rate | 85% |
| Nombre de trades | 966 |
| Levier | x2 |

La stratégie a survécu au bear market 2022 (-70% sur BTC) sans jamais s'effondrer.

---

## Comment ca marche

```
1. On calcule le hedge ratio dynamique entre BTC et ETH via Kalman Filter
2. On construit un spread : log(BTC) - beta * log(ETH)
3. On normalise ce spread en Z-Score
4. Si Z > 2  → spread trop haut → on short BTC, on long ETH
   Si Z < -2 → spread trop bas  → on long BTC, on short ETH
   Si |Z| < 0.5 → retour a la moyenne → on ferme tout
5. La taille de chaque position est calculee via Kelly Criterion (60%)
```

---

## Structure du projet

```
Stat_Arb_Project/
├── agent.py          bot live sur Binance
├── backtest.py       simulation historique
├── requirements.txt  dependances Python
└── results/
    ├── results.txt       metrics + trade log complet
    └── equity_curve.png  graphiques
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Lancer le backtest

```bash
python backtest.py
```

Les résultats sont sauvegardés automatiquement dans le dossier `results/`.

## Lancer le bot en live

```bash
# 1. Ajoute tes cles API Binance dans agent.py
# 2. Uncommente la ligne create_market_order
# 3. Lance :
python agent.py
```

---

## Config principale

```python
PAIR_A     = "BTC/USDT"
PAIR_B     = "ETH/USDT"
Z_ENTRY    = 2.0        # seuil d'entree
Z_EXIT     = 0.5        # seuil de sortie
KELLY_FRAC = 0.60       # agressivite du sizing
LEVERAGE   = 2          # levier Binance Futures
```

---

> **Disclaimer** — Les performances passées ne garantissent pas les résultats futurs. Teste toujours en paper trading avant de risquer du vrai capital. Le bot est en mode simulation par défaut (aucun ordre réel envoyé).
