# 📉 Portfolio Stress Lab & Strategic Advisor

A **professional-grade Python dashboard** for analyzing investment portfolios.

This tool combines **Monte Carlo risk simulation**, **Scenario Analysis**, **Smart Stock Screening**, and **Rules-Based Rebalancing** into a clean **Streamlit** interface.

Designed for investors who want to move beyond simple spreadsheets and visualize **“What-If” scenarios** with institutional-grade logic.

---

## ✨ Features

### 🔍 Portfolio Inspection
- **Deep Dive Visualization**  
  Interactive **Sunburst** and **Treemap** charts for sector and asset allocation.
- **Rich Metadata**  
  Automatic enrichment with **Dividend Yield, Sector, and Country** via Yahoo Finance.
- **Income Analysis**  
  Calculates the **weighted average dividend yield**  
  (supports manual yield overrides for Cash / Fixed Income).
- **Report Generation**  
  One-click download of a detailed **HTML Inspection Report**.

---

### 🌪️ Stress Lab (Risk Engine)
- **Monte Carlo Simulation**
  - Value at Risk (**VaR**)
  - Conditional VaR (**CVaR**)
  - Max Drawdown probabilities
- **Scenario Stress Testing**
  - Equity Crash (**-30%**)
  - Inflation Shock / **Stagflation**
  - **Credit Spread Widening**
  - **Interest Rate Hikes (+200 bps)**
  - **Bull Market** (upside comparison)
- **Rolling Regime Analysis**
  - Rolling **Volatility** (“Fear”)
  - Rolling **Sharpe Ratio** (“Efficiency”)
- **Correlation Heatmap**
  - Identifies diversification breakdowns and highly correlated assets.

---

### ⚖️ Strategic Advisor (Smart 4.0)
- **Composite Scoring Engine** (0–100)
  - **Valuation**: P/E relative to fair value
  - **Momentum**: RSI (“buy the dip”) + trend vs 200-day SMA
  - **Diversification**: Penalties for sector concentration
  - **Risk / Reward**: Downside risk added vs upside potential gained
- **Rules-Based Rebalancing**
  - Automated waterfall logic to trim positions
  - Refill cash buffers
  - Fund new ideas
- **Deep Dive Workflow**
  - One-click **Analyze** buttons to inspect recommendations instantly.

---

### 🔬 Advanced Stock Screener
- **Sector Benchmarking**
  - Compares **P/E**, **Profit Margins**, and **ROE** against industry averages
- **Technical Dashboard**
  - RSI (14)
  - SMA 50 / 200 distance
  - Beta
- **Live Market Data**
  - Real-time pricing
  - Daily price change
  - Analyst recommendations

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- `pip`

### Clone the Repository
```bash
git clone https://github.com/yourusername/portfolio-stress-lab.git
cd portfolio-stress-lab
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage
```bash
streamlit run app.py
```

Open:
http://localhost:8501

---

## 📊 Data Input Methods

### Manual Entry
```txt
AAPL, 10
MSFT, 5
BAS.DE, 20
```

### Excel Upload
Supports Name, Ticker, Weight, Value (optional), Yield (optional).

---

## 🧠 Logic & Methodology

### Smart Advisor Scoring
- Scenario Impact
- Sector Penalty
- Technicals
- Fundamentals

### Rebalancing Waterfall
1. Trim oversized positions
2. Secure cash buffer
3. Fund new ideas
4. Reinvest surplus

---

## ⚠️ Disclaimer
Educational use only. No financial advice.

---

## 📄 License
MIT License
