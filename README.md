# 📉 Portfolio Stress Lab & Strategic Advisor

A **professional-grade Python dashboard** for analyzing investment portfolios.  
This tool combines **Monte Carlo risk simulation**, **Scenario Analysis**, and **Rules-Based Rebalancing** into a clean **Streamlit** interface.

Designed for investors who want to move beyond simple spreadsheets and visualize **"What‑If" scenarios**.

---

## ✨ Features

### 🔍 Portfolio Inspection
- **Deep Dive Visualization**: Interactive **Sunburst** and **Treemap** charts for sector and asset allocation.
- **Rich Metadata**: Automatic enrichment with **Dividend Yield, Sector, and Country** via Yahoo Finance.
- **Income Analysis**: Calculates the **weighted average dividend yield** of the portfolio.
- **Report Generation**: One‑click download of a detailed **HTML Inspection Report**.

### 🌪️ Stress Lab (Risk Engine)
- **Monte Carlo Simulation**: Calculates **Value at Risk (VaR)**, **Conditional VaR**, and **Max Drawdown probabilities**.
- **Scenario Stress Testing**:
  - Equity Crash (**‑30%**)
  - Inflation Shock / **Stagflation**
  - **Credit Spread Widening**
  - **Interest Rate Hikes (+200 bps)**
- **Rolling Regime Analysis**:
  - Rolling **Volatility** ("Fear")
  - Rolling **Sharpe Ratio** ("Efficiency")
- **Correlation Heatmap**: Identifies diversification breakdowns and highly correlated assets.

### ⚖️ Advisor & Rebalancing
- **Rules‑Based Rebalancing** using configurable constraints:
  - **Minimum Cash Buffer** (e.g., 15%)
  - **Maximum Position Cap** (e.g., no asset > 12%)
- **Ghost Position Simulation**:
  - Simulate hypothetical additions (e.g., *"What if I add 3% Microsoft?"*)
  - Automatically determines funding trades.
- **Strategic Screener**:
  - Curated universe of ~100 high‑quality stocks
  - Filters by:
    - **Vision**: Income | Growth | Crisis‑Ready
    - **Hedge Need**: Stagflation, Crash Protection, etc.
    - **Sector Preference**: Healthcare, Tech, Energy, etc.
- **Impact Analysis**:
  - Quantifies how each suggested trade improves or worsens portfolio resilience under stress scenarios.

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- `pip`

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/portfolio-stress-lab.git
cd portfolio-stress-lab
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### Recommended `requirements.txt`
```text
streamlit
pandas
numpy
yfinance
plotly
scipy
```

---

## 🖥️ Usage

Run the dashboard locally:
```bash
streamlit run app.py
```

The application will open automatically at:
```
http://localhost:8501
```

---

## 📊 Data Input

The app supports:
- CSV portfolio uploads
- Manual editing of holdings directly in `app.py`

### Supported CSV Format
The CSV file must include at least the following columns:

| Column | Description |
|------|------------|
| `Ticker` | Asset ticker (e.g., AAPL, MSFT) |
| `Name` | Asset name |
| `Weight` | Decimal weight (e.g., 0.05 = 5%) |
| `Kind` | EQUITY \| CASH \| ALT |

---

## 🧠 Logic & Methodology

### Risk Metrics
- **Value at Risk (VaR)**  
  - Historical simulation using **bootstrap resampling**
- **Volatility**
  - Annualized standard deviation of daily returns

### Rebalancing Logic — *Waterfall Method*
1. **Trim** positions exceeding the maximum cap.
2. **Secure Cash** to meet the minimum buffer.
3. **Fund New Ideas** (ghost positions first).
4. **Reinvest** remaining surplus into underweight assets.

---

## ⚠️ Disclaimer

This software is provided **for educational and informational purposes only**.  
It does **not** constitute financial advice.

Market data is sourced from **Yahoo Finance** and may be delayed or inaccurate.  
Past performance and simulated stress tests are **not indicative of future results**.

---

## 📄 License

MIT License
