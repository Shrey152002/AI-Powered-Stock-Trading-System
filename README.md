# 📈 AI-Powered Stock Trading System 🚀

This project is a **Reinforcement Learning-based trading system** that uses historical and real-time data to **train**, **backtest**, and **live trade** stock portfolios. It combines **PPO (Proximal Policy Optimization)** with **modern portfolio theory** to make intelligent investment decisions.

---

## 📆 Features

- 🤖 **Reinforcement Learning**: PPO agent learns to trade AAPL stock using `Stable-Baselines3`.
- 📊 **Backtesting**: Evaluate strategies on historical data (2020–2023).
- ♻️ **Live Trading**: Real-time trading through the **Alpaca** brokerage API.
- 💼 **Portfolio Optimization**: Maximize Sharpe ratio or minimize volatility across multiple assets.
- 📉 **Performance Metrics**: Sharpe Ratio, Win Rate, Max Drawdown, etc.
- 🧠 **Comparison with Buy & Hold**: See if AI can outperform traditional investing.
- 📈 **Technical Indicators**: SMA, normalized volume, etc.

---

## 📁 Project Structure

```bash
stocks.py              # Main file with full logic
ppo_aapl_trading.zip   # Trained PPO model (optional)
```

---

## ⚙️ Installation

```bash
pip install gymnasium gym_anytrading
pip install stable-baselines3[extra]
pip install yfinance alpha_vantage alpaca-trade-api ta
```

---

## 🔧 Configuration

Update your Alpaca and Alpha Vantage API keys in the `Config` class inside `stocks.py`:

```python
ALPACA_API_KEY = "your_key"
ALPACA_API_SECRET = "your_secret"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading only

ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"
```

---

## 🚀 How to Use

### 📚 1. Train PPO Model

```python
python stocks.py  # or run the bottom "Train PPO" block
```

Saves model to: `ppo_aapl_trading.zip`

---

### 🧪 2. Backtest the Model

```python
# Evaluate performance on 2023 test set
evaluate_model_accuracy(
    model_path="ppo_aapl_trading",
    test_data_start="2023-01-01",
    test_data_end="2023-12-31"
)
```

Also compares with Buy & Hold:
```python
compare_to_baseline(model_metrics)
```

---

### 💹 3. Run Live Trading (Paper Trading)

```python
run_live_trading_system()
```

This executes trades every 60 seconds during market hours using live AAPL prices and your PPO model.

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

