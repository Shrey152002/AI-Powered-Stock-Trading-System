# AAPL Trading Bot — Project Guide

A plain-English walkthrough of what this project does, how it's trained and evaluated, and how the MLOps tooling (MLflow, Docker, tests, CI) fits around it — with small examples for every concept.

---

## 1. What this project actually does

Imagine a student who has never traded a stock. You hand them 10 years of Apple's daily price history and say: *"Every day, decide: hold the stock, or don't. I'll tell you afterward whether that made or lost money."* They try this millions of times, slowly getting better at noticing patterns. That's the whole idea — except the "student" is a PPO (Proximal Policy Optimization) reinforcement-learning agent, and the "practice" is a training loop.

Once trained, the bot can be used in three ways:

| Command | What it does |
|---|---|
| `train` | Practice on historical data, save what it learned as a model file |
| `evaluate` | Test the trained model on data it has never seen, and compare it to just buying and holding |
| `trade` | Run it live against a paper (fake-money) brokerage account |

**Scope:** the project is trained and validated primarily on **AAPL**. It can point at any ticker (`SYMBOL=MSFT`), and §5b documents what happened when it was actually tried on four others — worth reading before assuming this generalizes.

---

## 2. How it works, step by step

```
historical prices  →  trading environment  →  PPO agent  →  buy/hold decision
   (yfinance)          (10-day window)        (the "brain")
```

**Example, concretely:** the agent looks at the last 10 trading days. For each day it sees five numbers:

| Day | % price change | Volume (0–1) | RSI (0–1) | Trend vs 20‑day avg | Volatility |
|---|---|---|---|---|---|
| -9 | +0.4% | 0.62 | 0.58 | +1.2% | 0.011 |
| -8 | -1.1% | 0.71 | 0.51 | +0.4% | 0.013 |
| ... | ... | ... | ... | ... | ... |
| -1 (today) | +0.8% | 0.55 | 0.63 | +2.1% | 0.010 |

That's a 10×5 grid of numbers — the *only* thing the model sees (earlier versions of this project used just the first two columns; see §5a for why the rest were added). Based on that grid, it outputs one of two choices: **0** (go flat) or **1** (go long/hold the stock). Still no news, no fundamentals — just price behavior and momentum.

Each time it acts, it gets a **reward**: the realized gain or loss when it closes a position, *minus* a small fee — the same fee that would actually apply in evaluation. Positive reward nudges the model toward repeating that decision; negative reward (including the fee) nudges it away from acting too often. Over hundreds of thousands of these tiny nudges, a policy emerges.

---

## 3. Training, in simple terms

- **`total_timesteps`** = how many of these single-day decisions it practices. 40,000 is like cramming for 40 minutes; 500,000 is closer to a full semester. More practice (usually) means a better-tuned policy — up to a point.
- **`seed`** = the random starting point for training. Two runs with identical settings but different seeds can land on meaningfully different policies — see §5a. Never trust a single seed's result.
- **Training window** = which years of history it practices on. Practicing only on a rising market (say, 2020–2023, a strong bull run for AAPL) teaches it fewer lessons than practicing across ups, downs, and sideways stretches (this project uses 2015–2024).
- **Algorithm** = PPO, a well-established reinforcement-learning method. Nothing custom in the algorithm itself — a small neural network (`MlpPolicy`) with default Stable-Baselines3 settings. The custom part is the environment: the 5 features above, and a reward that charges the agent its own trading fee (§7, bug #7).

---

## 4. Metrics explained, with tiny examples

### Total / Percent Return
*"If I'd put $1 in at the start, what do I have at the end?"*
**Example:** portfolio value goes from $1.00 → $1.14 → that's a **14% return**.

### Sharpe Ratio
*"Was that return worth the bumpy ride?"*
**Example:** Two strategies both end the year up 20%.
- Strategy A: gains a little every single day. Smooth. High Sharpe.
- Strategy B: up 60%, crashes to down 10%, claws back to +20%. Same ending number, way scarier journey. Low Sharpe.

Formula in words: *average daily return ÷ how spread-out those daily returns are*, scaled up to a yearly number. Higher is better; above ~1 is generally considered decent, above 2 is very good.

### Maximum Drawdown
*"What's the worst dip I'd have sat through?"*
**Example:** your portfolio peaks at $150, then falls to $120 before recovering. That's a drawdown of `(150-120)/150 = 20%` — the number that answers "how bad could this have felt at the worst moment?", regardless of how it ends up.

### Win Rate — ⚠️ the one that's easy to misread
*"What fraction of individual days had the portfolio's value go up?"*
**Example:** out of 250 trading days, only 10 had a value increase → **4% win rate**. That sounds terrible, but it isn't the same as "the strategy lost money 96% of the time" — most days a hold-or-flat strategy simply doesn't move, and the real profit comes from a handful of days that mattered. Don't judge the strategy by this number alone — look at total return and Sharpe instead.

### Total Trades
*"How many times did it actually flip its position?"*
**Example:** early versions of this project reported "238 trades" for every single stock tested, regardless of how that stock actually moved — because the code was accidentally counting *trading days*, not real position flips (§7, bug #6). After the fix, real trade counts range from 0 to ~20 a year depending on the stock and seed.

---

## 5. Real evaluation results (from this project)

### 5a. AAPL — the headline result, before and after

| | Before | After |
|---|---|---|
| Features | price % change, volume | + RSI, trend, volatility |
| Reward | price gain only | price gain **minus trading fee** |
| Timesteps | 300,000 | 500,000 |
| Seeds tested | 1 | 3 |
| **Sharpe Ratio** | -0.29 | **1.50 ± 0.16** |
| **Trades/year** | 53 | **~5** |
| **Return** | -7.46% | **13.59% ± 6.27** |
| **Beat buy & hold** | No | **2 of 3 seeds** |

This is the number worth trusting: Sharpe went from *negative* (losing money on a risk-adjusted basis) to a consistent ~1.5 with a small spread across three independent training runs — not a one-off lucky result. Trading frequency dropped by ~90% once the reward actually charged the agent its own trading fee.

**Being precise about what this claim covers:** beating buy & hold is genuinely inconsistent (2 of 3 seeds, not 3 of 3) — call this "usually helps, not guaranteed," not "beats the market." The Sharpe and overtrading improvements, however, are robust across every seed tested.

### 5b. Does it generalize to other stocks? — a real limitation, not hidden

The same treatment (5 features, fee-aware reward, 500k timesteps, 3 seeds) was run on **MSFT, AMZN, GOOGL, and META** too:

| Symbol | Return | Sharpe | Trades (avg) | Outperformance | Beat B&H |
|---|---|---|---|---|---|
| AAPL | 13.59% ± 6.27 | 1.50 ± 0.16 | 5.3 | +1.10 ± 6.27 | 2/3 |
| MSFT | 1.09% ± 6.12 | 0.38 ± 0.47 | 10.0 | -16.23 ± 6.12 | 0/3 |
| AMZN | -0.24% ± 0.34 | -0.34 ± 0.49 | 0.0 | -5.83 ± 0.34 | 0/3 |
| GOOGL | 26.36% ± 19.90 | 0.96 ± 0.70 | 10.7 | -39.97 ± 19.90 | 0/3 |
| META | 4.11% ± 1.16 | 0.29 ± 0.03 | 4.0 | -7.37% ± 1.16 | 0/3 |

**Beat buy & hold in only 2 of 15 runs total — both on AAPL.** Two honest findings here:

1. **It doesn't generalize.** 2025 was a strong year for MSFT/AMZN/GOOGL/META buy-and-hold specifically, and the model couldn't keep pace on any of them. This is real evidence, not a training bug — don't claim "beats the market" as a general property of this approach.
2. **A new failure mode: policy collapse.** AMZN produced **zero trades in all 3 seeds**, and GOOGL did the same in 1 of 3 — the model learned to never act at all. The likely cause: the trading-fee penalty (added to fix overtrading) can overcorrect if the agent's willingness to explore (PPO's entropy) decays before it finds a trade worth the fee. Fixing this — e.g. an entropy-bonus floor, or a smaller fee coefficient — is the natural next experiment (§9).

**Why keep this section instead of just reporting AAPL:** an interviewer asking "does this generalize?" deserves this exact answer — tested, didn't generalize, found a specific and explainable failure mode. That's stronger than not having checked.

---

## 6. How MLOps is integrated (and why each piece matters)

Think of MLOps here as answering one question per piece: *"how do I stop this from quietly breaking, becoming impossible to reproduce, or fooling me with a bad result?"*

### Config via `.env` (not hardcoded)
**Example:** instead of editing code to test a new date range or seed, you just run:
```bash
TRAIN_START=2015-01-01 TOTAL_TIMESTEPS=500000 SEED=1 python -m trading_agent.cli train
```
Same code, different behavior — and no risk of accidentally committing an API key to git.

### Data caching (a simple stand-in for data versioning)
**Example:** the first time you request AAPL data for 2015–2024, it's saved to `data/cache/AAPL_2015-01-01_2024-12-31.csv`. Every future run using that exact range reads the same file — so "why did my results change?" is never answered by "the data silently changed underneath me."

### MLflow (experiment tracking)
**Example:** comparing a 40k-timestep run against a 500k-timestep run — or three different seeds of the same config — used to mean scrolling back through terminal output. With MLflow every run sits in one queryable table, params and metrics together:

| Run | timesteps | seed | eval_sharpe_ratio | eval_percent_return |
|---|---|---|---|---|
| ppo-AAPL-seed1 | 500000 | 1 | 1.72 | 18.13 |
| ppo-AAPL-seed2 | 500000 | 2 | 1.42 | 4.72 |
| ppo-AAPL-seed3 | 500000 | 3 | 1.35 | 17.93 |

Open it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. This table is exactly how §5a's mean ± std was computed — not from memory.

### Tests + CI
**Example:** one test checks that a risk-limit function correctly *rejects* a trade that would put 90% of the portfolio into one stock:
```python
def test_check_risk_limits_blocks_oversized_position():
    portfolio = {"cash": 1_000, "positions": {}}
    trades = {"prices": {"AAPL": 100.0}, "target_positions": {"AAPL": 100}}
    assert check_risk_limits(portfolio, trades, max_position_pct=0.25) is False
```
This runs automatically on every GitHub push. If someone accidentally breaks the risk check while refactoring, the pull request shows a red ❌ instead of the bug reaching production.

### Docker (containerization)
**Example:** "it works on my machine" stops being an excuse — the same `Dockerfile` that builds and runs on this project's laptop will build and run identically on any other machine with Docker installed, because every dependency version is frozen inside the image.

### Risk management + logging
**Example:** instead of `print("buying stock")`, every decision is written to a timestamped log line:
```
2026-09-20 11:14:08 - INFO - Placing order: buy 12.4 shares of AAPL
```
saved to a file — so a bad trading day can be reconstructed afterward instead of lost when the terminal closes.

---

## 7. Real bugs hit while building this (with simple examples)

| # | What happened | Simple explanation | Fix |
|---|---|---|---|
| 1 | `df['Close']` returned a table instead of a column | Newer `yfinance` changed its output format without warning — like a spreadsheet vendor silently adding an extra header row | Flatten the column format right after download |
| 2 | Docker build looked frozen for 40+ minutes | It was downloading ~3–4GB of GPU (CUDA) software the project doesn't need — like ordering a truck to deliver a letter | Installed the CPU-only version of PyTorch instead |
| 3 | `MlflowException: filesystem tracking... maintenance mode` | The newest MLflow version stopped supporting the simple "just save to a folder" method | Switched to a small local database file (SQLite) instead |
| 4 | `ModuleNotFoundError: alpha_vantage.sectorperformance` | Two libraries that depend on each other were installed at versions that don't actually match, like a phone case that doesn't fit the new phone model | Pinned the older, compatible library version |
| 5 | Docker said it saved the model — but the file never appeared on the computer | On Windows, the terminal (Git Bash) was quietly rewriting part of the Docker command before running it | Added a flag (`MSYS_NO_PATHCONV=1`) that tells the terminal not to do that |
| 6 | `Key ID must be given to access Alpaca trade API` | The `.env` file with the API keys existed — but was completely empty (0 bytes) | Filled it in with real values and saved it properly |
| 7 | Every stock showed a suspicious, identical "238 trades," and two stocks showed an impossible "0% drawdown" | The evaluation code was checking for a data field (`portfolio_value`) the trading library never provides. It silently fell back to a homemade guess that treated raw dollar price-swings as if they were 1%-scaled daily returns — and separately, it was counting *trading days*, not real trades | Read the library's actual, fee-aware profit tracker (`total_profit`) instead of guessing, and count real position flips |
| 8 | Backtesting the *same saved model* twice gave two different results | The model's decision function samples randomly by default — like asking the same trained student the same question twice and letting them roll a die each time before answering | Told it to always give its single best answer (`deterministic=True`) |

**Why #7 and #8 matter most:** every "the model beats the market" number reported earlier in this project's development (including a headline "+28.92% outperformance") was a direct artifact of these two bugs, not real model skill. Fixing them reversed the conclusion entirely — see §5a for what the honest number actually is.

---

## 8. How to run it

```bash
# one-time setup (use Python 3.10 or 3.11 — not the newest release)
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# train, evaluate, or paper-trade
python -m trading_agent.cli train
python -m trading_agent.cli evaluate
python -m trading_agent.cli trade

# reproduce the §5a result: 3 seeds, one symbol
python scripts/run_seeds.py

# reproduce the §5b result: 3 seeds x 4 more symbols
python scripts/run_full_sweep.py

# compare every run
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 9. Resume bullets (ready to use)

- *"Built and trained a PPO reinforcement-learning trading agent (Stable-Baselines3) with a custom Gym environment, improving mean Sharpe ratio from -0.29 to 1.50 (± 0.16 across 3 seeds) and cutting trading frequency ~90% by adding technical indicators and a fee-aware reward signal."*
- *"Found and fixed two evaluation bugs — a wrong data-field lookup and non-deterministic inference — that had been silently inflating backtest results by ~50 percentage points; re-validated all results after the fix."*
- *"Ran a 5-symbol × 3-seed generalization study, discovering and root-causing a policy-collapse failure mode under a new reward penalty."*
- *"Added MLflow experiment tracking, pytest/CI, and a CPU-optimized Docker image (678MB) to a research notebook, making every training run reproducible and comparable."*

---

## 10. What to improve next

- **Fix the zero-trade collapse (§5b).** Try a smaller fee coefficient or an entropy-bonus floor so the agent doesn't overcorrect into never trading.
- **Fail fast on bad config.** Right now a missing API key causes a confusing error three layers deep instead of a clear "ALPACA_API_KEY is not set" message at startup.
- **Test across more time periods.** One train/test split per symbol is one data point each. Walk-forward validation (rolling windows across multiple years) would give a real confidence interval instead of a single split.
- **Track multiple models properly.** Right now every training run overwrites the same file per symbol/seed. MLflow's Model Registry can tag one as "production" while others stay archived for comparison.
- **Isolate the §5a improvement further.** Features, reward, and timesteps all changed together. Ablating them one at a time (already easy — each is an independent config value) would show which change contributed the most.
