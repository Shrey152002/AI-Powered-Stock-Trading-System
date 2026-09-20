# 📈 AI-Powered Stock Trading System 🚀

This project is a **Reinforcement Learning-based trading system** that uses historical and real-time data to **train**, **backtest**, and **live trade** AAPL (and, experimentally, other tickers). It combines **PPO (Proximal Policy Optimization)** with an MLOps layer (MLflow, Docker, CI, tests) to make every result reproducible.

📖 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — a full plain-English walkthrough: how it works, what the metrics mean, real (honest, including negative) evaluation results, and every bug found while building this.

---

## 📆 Features

- 🤖 **Reinforcement Learning**: PPO agent trades AAPL using `Stable-Baselines3`, with RSI/trend/volatility features and a fee-aware reward.
- 📊 **Backtesting**: Evaluate strategies on a configurable historical window, with multi-seed and multi-symbol sweep scripts included.
- ♻️ **Live Trading**: Real-time trading through the **Alpaca** brokerage API (paper trading).
- 📉 **Performance Metrics**: Sharpe Ratio, Win Rate, Max Drawdown, computed from the environment's real (fee-aware) profit tracker.
- 🧠 **Comparison with Buy & Hold**: See the real, sometimes unflattering, comparison — see PROJECT_GUIDE.md §5.

---

## 📁 Project Structure

```bash
src/trading_agent/
    config.py         # Settings loaded from environment / .env (pydantic-settings)
    data.py           # yfinance download + on-disk cache (simple data versioning)
    env.py            # Custom gym_anytrading environment
    train.py          # Trains PPO, logs params/metrics/model to MLflow
    evaluate.py        # Backtests the model, plots performance, compares to buy & hold
    risk.py           # Position-size / daily-loss checks + market-hours check
    live_trading.py   # Alpaca paper-trading loop using the trained model
    cli.py            # `python -m trading_agent.cli {train,evaluate,trade}`
scripts/
    run_seeds.py       # Train/evaluate one symbol across multiple seeds, report mean +/- std
    run_multi_symbol.py  # Train/evaluate across several tickers
    run_full_sweep.py  # Multi-symbol x multi-seed sweep (see PROJECT_GUIDE.md §5b)
tests/                # pytest unit tests (env, risk, config) — no network required
stocks_trading.ipynb  # Original exploratory notebook, kept for reference
Dockerfile            # Container image for training/evaluation/trading
.github/workflows/ci.yml  # Runs pytest on every push
```

---

## ⚙️ Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

> Use Python 3.10 or 3.11 — `stable-baselines3` / `gym_anytrading` don't yet support the newest Python releases.

---

## 🔧 Configuration

Copy `.env.example` to `.env` and fill in your keys — nothing is hardcoded in source anymore:

```bash
cp .env.example .env
```

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper trading only
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

Every other setting (symbol, training window, risk limits, MLflow experiment name, etc.) has a sane default in `src/trading_agent/config.py` and can be overridden the same way, e.g. `SYMBOL=MSFT`, `TOTAL_TIMESTEPS=100000`.

---

## 🚀 How to Use

### 📚 1. Train the PPO model

```bash
python -m trading_agent.cli train
```

Saves the model to `models/ppo_aapl_trading.zip` and logs the run (params, metrics, model artifact) to MLflow, tracked in a local `mlflow.db` SQLite file. Inspect runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

### 🧪 2. Backtest the model

```bash
python -m trading_agent.cli evaluate
```

Backtests on the configured test window, saves a performance chart to `reports/performance.png`, compares against a buy-and-hold baseline, and logs both sets of metrics to the same MLflow run.

---

### 💹 3. Run live (paper) trading

```bash
python -m trading_agent.cli trade
```

Polls the market every `UPDATE_INTERVAL_SECONDS` (default 60s), asks the trained PPO model for a buy/hold decision, and submits the order through Alpaca — paper trading only unless you change `ALPACA_BASE_URL`.

---

### 🐳 Run any of the above in Docker

```bash
docker build -t trading-agent .
docker run --rm --env-file .env \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data/cache:/app/data/cache" \
  -v "$PWD/reports:/app/reports" \
  trading-agent train    # or: evaluate / trade
```

> **Windows + Git Bash users:** prefix the command with `MSYS_NO_PATHCONV=1` (e.g. `MSYS_NO_PATHCONV=1 docker run ...`). Git Bash silently mangles the container-side half of `-v host:container` paths otherwise, so the volume mount looks fine but nothing actually gets written back to the host — this is a real gotcha, verified while building this out.

---

### ✅ Run the tests

```bash
pytest -v
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push/PR.

---

## 📊 Key Performance Metrics

- **Total Return**
- **Annual Return**
- **Sharpe Ratio**
- **Max Drawdown**
- **Win Rate**
- **Normalized Portfolio Growth**

---

## 📌 Live Trading Workflow

1. ✅ Check if market is open
2. 📈 Fetch live stock data
3. 🤖 PPO model decides (Buy/Hold)
4. 💸 Trades sent to Alpaca (paper/live)
5. 📊 Track performance, value, and trades

---

## 🧠 Use Cases

- Automated algorithmic trading bots
- Quantitative trading research
- RL experimentation in financial markets
- Personal trading assistant
- Academic projects

---

## 🛡️ Risk Management

- Max position: 25% of total capital
- Max daily loss: 3%
- Uses logging and backtesting to verify model decisions before real deployment

---

## 📋 License

This project is for **educational and research purposes only**. Use at your own risk. Trading involves financial risk, and past performance does not guarantee future results.

---

## 👨‍💼 Author

**Shreyash Verma**  
3rd Year B.Tech, Indian Institute of Information Technology, Nagpur  
AI/ML Enthusiast | Quant Research | Full-Stack ML Developer

---

## 📬 Feedback or Questions?

Feel free to [connect on LinkedIn](https://www.linkedin.com/in/shreyash-verma01) or drop an issue in this repo!

