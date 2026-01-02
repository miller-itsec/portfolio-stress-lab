import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import io
import requests

# ==========================================
# 1. CONSTANTS & DATA
# ==========================================

CASH_KEYWORDS = [
    "tagesgeld", "giro", "verrechnungskonto", "sparkasse", "comdirect",
    "postbank", "direktbank", "hypo-vereinsbank", "barclays", "santander", "ing", "volkswagen financial",
]
ALT_KEYWORDS = ["gold", "silver", "jewelry", "art"]

TICKER_RE = re.compile(r"\(([A-Z]+):([A-Z0-9\.\-]+)\)")
SIMPLE_PAREN_TICKER_RE = re.compile(r"\(([A-Z0-9]{1,10}(\.[A-Z]{1,4})?)\)\s*$")

# Mapping European/obscure symbols to Yahoo Finance
TICKER_MAP = {
    "2PP": "PYPL", "W3U": "WU", "PJXA": "PBR", "1KN": "VICI", "LTP": "LTC",
    "BMT": "BTI", "CLE": "CSL", "ONK": "OKE", "BAYN": "BAYN.DE", "BAC": "VZ",
}

# Approximate Sector Benchmarks (P/E, Profit Margin, ROE)
SECTOR_BENCHMARKS = {
    "Technology": {"PE": 35.0, "Margin": 0.20, "ROE": 0.25},
    "Healthcare": {"PE": 25.0, "Margin": 0.15, "ROE": 0.18},
    "Financial Services": {"PE": 12.0, "Margin": 0.25, "ROE": 0.12},
    "Energy": {"PE": 10.0, "Margin": 0.10, "ROE": 0.15},
    "Consumer Cyclical": {"PE": 22.0, "Margin": 0.08, "ROE": 0.15},
    "Consumer Defensive": {"PE": 24.0, "Margin": 0.10, "ROE": 0.20},
    "Industrials": {"PE": 20.0, "Margin": 0.08, "ROE": 0.14},
    "Utilities": {"PE": 18.0, "Margin": 0.12, "ROE": 0.10},
    "Basic Materials": {"PE": 15.0, "Margin": 0.07, "ROE": 0.12},
    "Real Estate": {"PE": 35.0, "Margin": 0.30, "ROE": 0.08}, # High P/E due to depreciation
    "Communication Services": {"PE": 20.0, "Margin": 0.15, "ROE": 0.12}
}

ADVISOR_PALETTE = [
    # --- HEALTHCARE (Defensive / Income) ---
    {"name": "Johnson & Johnson", "ticker": "JNJ", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["balanced", "income"], "scenarios": ["equity_crash", "credit_event", "stagflation"], "why": "AAA-rated balance sheet; defensive stabilizer."},
    {"name": "UnitedHealth Group", "ticker": "UNH", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["balanced", "growth"], "scenarios": ["equity_crash", "soft_landing"], "why": "Dominant managed care compounder."},
    {"name": "Eli Lilly", "ticker": "LLY", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["growth"], "scenarios": ["soft_landing", "recovery"], "why": "High-growth pharma (obesity/diabetes)."},
    {"name": "Novo Nordisk", "ticker": "NVO", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["growth", "balanced"], "scenarios": ["soft_landing", "recovery"], "why": "GLP-1 market leader; structural growth."},
    {"name": "Merck & Co.", "ticker": "MRK", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["balanced", "income"], "scenarios": ["equity_crash"], "why": "Oncology dominance; reasonable valuation."},
    {"name": "AbbVie", "ticker": "ABBV", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["income", "balanced"], "scenarios": ["equity_crash", "stagflation"], "why": "High yield; strong immunology franchise."},
    {"name": "Sanofi", "ticker": "SNY", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["income", "balanced"], "scenarios": ["equity_crash"], "why": "Cheap valuation; EU defensive exposure."},
    {"name": "Pfizer", "ticker": "PFE", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["income"], "scenarios": ["equity_crash"], "why": "Deep value; high yield turnaround play."},
    {"name": "Gilead Sciences", "ticker": "GILD", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["income", "balanced"], "scenarios": ["equity_crash"], "why": "Cashflow machine; HIV/Oncology value."},
    {"name": "Thermo Fisher", "ticker": "TMO", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Life sciences pick-and-shovel leader."},
    {"name": "Danaher", "ticker": "DHR", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Quality serial acquirer; recurring revenue."},
    {"name": "CVS Health", "ticker": "CVS", "bucket": "EQUITY", "sector": "Healthcare", "vision": ["income"], "scenarios": ["recovery", "equity_crash"], "why": "Deep value vertical integration."},

    # --- TECHNOLOGY (Growth / Cyclical) ---
    {"name": "Microsoft", "ticker": "MSFT", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth", "balanced"], "scenarios": ["soft_landing", "recovery"], "why": "Enterprise cloud & AI utility."},
    {"name": "Apple", "ticker": "AAPL", "bucket": "EQUITY", "sector": "Technology", "vision": ["balanced", "growth"], "scenarios": ["soft_landing"], "why": "Consumer staple of tech; cash rich."},
    {"name": "NVIDIA", "ticker": "NVDA", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery", "soft_landing"], "why": "AI infrastructure monopoly."},
    {"name": "Alphabet", "ticker": "GOOGL", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery", "soft_landing"], "why": "Search monopoly; reasonable valuation."},
    {"name": "Amazon", "ticker": "AMZN", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth"], "scenarios": ["recovery", "soft_landing"], "why": "Cloud + Logistics dominance."},
    {"name": "Meta Platforms", "ticker": "META", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery"], "why": "Social dominance; high free cash flow."},
    {"name": "Broadcom", "ticker": "AVGO", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth", "income"], "scenarios": ["soft_landing", "inflation_shock"], "why": "AI connectivity + software margins."},
    {"name": "ASML Holding", "ticker": "ASML", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery"], "why": "Semiconductor lithography monopoly."},
    {"name": "Salesforce", "ticker": "CRM", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "SaaS CRM leader; improving margins."},
    {"name": "Adobe", "ticker": "ADBE", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery"], "why": "Creative software monopoly."},
    {"name": "Oracle", "ticker": "ORCL", "bucket": "EQUITY", "sector": "Technology", "vision": ["balanced", "growth"], "scenarios": ["recovery"], "why": "Legacy database to cloud transition."},
    {"name": "SAP", "ticker": "SAP", "bucket": "EQUITY", "sector": "Technology", "vision": ["balanced"], "scenarios": ["recovery"], "why": "EU enterprise software powerhouse."},
    {"name": "Palo Alto Networks", "ticker": "PANW", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery", "soft_landing"], "why": "Cybersecurity platform leader."},
    {"name": "CrowdStrike", "ticker": "CRWD", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth"], "scenarios": ["recovery"], "why": "Cloud-native endpoint security."},
    {"name": "Infineon", "ticker": "IFX.DE", "bucket": "EQUITY", "sector": "Technology", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Power semis for EVs/Green energy."},

    # --- CONSUMER STAPLES (Defensive / Income) ---
    {"name": "Procter & Gamble", "ticker": "PG", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced", "income"], "scenarios": ["equity_crash", "stagflation"], "why": "The ultimate defensive staple."},
    {"name": "PepsiCo", "ticker": "PEP", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced", "income"], "scenarios": ["equity_crash"], "why": "Snacks & Bev diversification."},
    {"name": "Coca-Cola", "ticker": "KO", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced", "income"], "scenarios": ["equity_crash"], "why": "Brand power; pricing power."},
    {"name": "Costco", "ticker": "COST", "bucket": "EQUITY", "sector": "Staples", "vision": ["growth", "balanced"], "scenarios": ["soft_landing", "equity_crash"], "why": "Best-in-class retailer; membership moat."},
    {"name": "Walmart", "ticker": "WMT", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced"], "scenarios": ["equity_crash", "stagflation"], "why": "Defensive scale; grocery dominance."},
    {"name": "Philip Morris", "ticker": "PM", "bucket": "EQUITY", "sector": "Staples", "vision": ["income"], "scenarios": ["equity_crash"], "why": "High yield; smoke-free transition leader."},
    {"name": "Altria", "ticker": "MO", "bucket": "EQUITY", "sector": "Staples", "vision": ["income"], "scenarios": ["equity_crash"], "why": "Maximum yield; domestic US tobacco."},
    {"name": "British American Tobacco", "ticker": "BTI", "bucket": "EQUITY", "sector": "Staples", "vision": ["income"], "scenarios": ["equity_crash"], "why": "Deep value; high dividend yield."},
    {"name": "Unilever", "ticker": "UL", "bucket": "EQUITY", "sector": "Staples", "vision": ["income", "balanced"], "scenarios": ["equity_crash"], "why": "Emerging market staples exposure."},
    {"name": "Diageo", "ticker": "DEO", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced"], "scenarios": ["recovery"], "why": "Global spirits leader; compounding quality."},
    {"name": "Mondelez", "ticker": "MDLZ", "bucket": "EQUITY", "sector": "Staples", "vision": ["balanced"], "scenarios": ["equity_crash"], "why": "Snacking leader; emerging market growth."},

    # --- CONSUMER DISCRETIONARY (Cyclical) ---
    {"name": "LVMH", "ticker": "MC.PA", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Global luxury proxy; pricing power."},
    {"name": "Hermès", "ticker": "RMS.PA", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth"], "scenarios": ["recovery", "inflation_shock"], "why": "Ultra-luxury scarcity; Veblen good."},
    {"name": "McDonald's", "ticker": "MCD", "bucket": "EQUITY", "sector": "Disc", "vision": ["balanced", "income"], "scenarios": ["soft_landing", "stagflation"], "why": "Real estate & franchise cash cow."},
    {"name": "Starbucks", "ticker": "SBUX", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Global brand recovery play."},
    {"name": "Home Depot", "ticker": "HD", "bucket": "EQUITY", "sector": "Disc", "vision": ["balanced"], "scenarios": ["recovery", "rates_up"], "why": "Housing market proxy; efficiency leader."},
    {"name": "Tesla", "ticker": "TSLA", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth"], "scenarios": ["recovery"], "why": "EV & Robotics optionality."},
    {"name": "Booking Holdings", "ticker": "BKNG", "bucket": "EQUITY", "sector": "Disc", "vision": ["growth"], "scenarios": ["recovery"], "why": "Asset-light travel leader."},

    # --- FINANCIALS (Value / Rates) ---
    {"name": "Berkshire Hathaway", "ticker": "BRK-B", "bucket": "EQUITY", "sector": "Financials", "vision": ["balanced"], "scenarios": ["equity_crash", "stagflation"], "why": "Fortress balance sheet; diversified conglomerate."},
    {"name": "JPMorgan Chase", "ticker": "JPM", "bucket": "EQUITY", "sector": "Financials", "vision": ["balanced", "income"], "scenarios": ["rates_up", "soft_landing"], "why": "Best-in-breed global bank."},
    {"name": "Visa", "ticker": "V", "bucket": "EQUITY", "sector": "Financials", "vision": ["growth", "balanced"], "scenarios": ["recovery", "inflation_shock"], "why": "Payments duopoly; inflation hedge."},
    {"name": "Mastercard", "ticker": "MA", "bucket": "EQUITY", "sector": "Financials", "vision": ["growth", "balanced"], "scenarios": ["recovery", "inflation_shock"], "why": "Payments duopoly; secular trend."},
    {"name": "BlackRock", "ticker": "BLK", "bucket": "EQUITY", "sector": "Financials", "vision": ["balanced", "growth"], "scenarios": ["recovery"], "why": "Asset management dominance."},
    {"name": "Allianz", "ticker": "ALV.DE", "bucket": "EQUITY", "sector": "Financials", "vision": ["income", "balanced"], "scenarios": ["rates_up"], "why": "Insurance giant; steady yield."},
    {"name": "Munich Re", "ticker": "MUV2.DE", "bucket": "EQUITY", "sector": "Financials", "vision": ["balanced"], "scenarios": ["rates_up", "credit_event"], "why": "Reinsurance hard market beneficiary."},
    {"name": "Chubb", "ticker": "CB", "bucket": "EQUITY", "sector": "Financials", "vision": ["balanced"], "scenarios": ["rates_up"], "why": "Conservative P&C insurer."},
    {"name": "S&P Global", "ticker": "SPGI", "bucket": "EQUITY", "sector": "Financials", "vision": ["growth"], "scenarios": ["recovery"], "why": "Financial data & ratings moat."},

    # --- ENERGY (Inflation / Value) ---
    {"name": "Exxon Mobil", "ticker": "XOM", "bucket": "EQUITY", "sector": "Energy", "vision": ["income", "balanced"], "scenarios": ["stagflation", "inflation_shock"], "why": "Integrated energy scale; strong balance sheet."},
    {"name": "Chevron", "ticker": "CVX", "bucket": "EQUITY", "sector": "Energy", "vision": ["income", "balanced"], "scenarios": ["stagflation", "inflation_shock"], "why": "Capital discipline; dividend aristocrat."},
    {"name": "Shell", "ticker": "SHEL", "bucket": "EQUITY", "sector": "Energy", "vision": ["income", "balanced"], "scenarios": ["stagflation", "inflation_shock"], "why": "LNG leader; valuation discount vs US."},
    {"name": "TotalEnergies", "ticker": "TTE", "bucket": "EQUITY", "sector": "Energy", "vision": ["income"], "scenarios": ["stagflation"], "why": "Best-in-class EU integrated; renewable hedge."},
    {"name": "ConocoPhillips", "ticker": "COP", "bucket": "EQUITY", "sector": "Energy", "vision": ["balanced"], "scenarios": ["inflation_shock"], "why": "Pure-play E&P; variable cash return."},
    {"name": "EOG Resources", "ticker": "EOG", "bucket": "EQUITY", "sector": "Energy", "vision": ["balanced"], "scenarios": ["inflation_shock"], "why": "Premium driller; tech-focused E&P."},
    {"name": "Schlumberger", "ticker": "SLB", "bucket": "EQUITY", "sector": "Energy", "vision": ["growth", "balanced"], "scenarios": ["recovery", "inflation_shock"], "why": "Global services leader."},
    {"name": "Canadian Natural", "ticker": "CNQ", "bucket": "EQUITY", "sector": "Energy", "vision": ["income", "balanced"], "scenarios": ["inflation_shock"], "why": "Long-life low-decline reserves."},

    # --- INDUSTRIALS & MATERIALS (Cyclical / Real Assets) ---
    {"name": "Caterpillar", "ticker": "CAT", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["inflation_shock", "recovery"], "why": "Infrastructure & mining proxy."},
    {"name": "Deere & Co", "ticker": "DE", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["inflation_shock"], "why": "Ag-tech leader; precision agriculture."},
    {"name": "Siemens", "ticker": "SIE.DE", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["recovery", "soft_landing"], "why": "Industrial automation & grid play."},
    {"name": "Honeywell", "ticker": "HON", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["soft_landing"], "why": "Industrial conglomerate quality."},
    {"name": "Union Pacific", "ticker": "UNP", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced", "income"], "scenarios": ["soft_landing"], "why": "Rail monopoly moat."},
    {"name": "Lockheed Martin", "ticker": "LMT", "bucket": "EQUITY", "sector": "Industrials", "vision": ["income", "balanced"], "scenarios": ["credit_event", "stagflation"], "why": "Defense spending hedge."},
    {"name": "RTX Corp", "ticker": "RTX", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["credit_event", "recovery"], "why": "Aerospace & Defense mix."},
    {"name": "Airbus", "ticker": "AIR.PA", "bucket": "EQUITY", "sector": "Industrials", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Commercial aviation duopoly."},
    {"name": "Vinci", "ticker": "DG.PA", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced", "income"], "scenarios": ["inflation_shock"], "why": "Concessions & toll roads (inflation linked)."},
    {"name": "Waste Management", "ticker": "WM", "bucket": "EQUITY", "sector": "Industrials", "vision": ["balanced"], "scenarios": ["equity_crash", "stagflation"], "why": "Recession-proof trash utility."},
    {"name": "Linde", "ticker": "LIN", "bucket": "EQUITY", "sector": "Materials", "vision": ["balanced", "growth"], "scenarios": ["soft_landing"], "why": "Industrial gases monopoly; pricing power."},
    {"name": "Air Liquide", "ticker": "AI.PA", "bucket": "EQUITY", "sector": "Materials", "vision": ["balanced"], "scenarios": ["soft_landing"], "why": "EU industrial gases leader."},
    {"name": "Freeport-McMoRan", "ticker": "FCX", "bucket": "EQUITY", "sector": "Materials", "vision": ["growth", "balanced"], "scenarios": ["inflation_shock", "recovery"], "why": "Copper play for electrification."},
    {"name": "Rio Tinto", "ticker": "RIO", "bucket": "EQUITY", "sector": "Materials", "vision": ["income"], "scenarios": ["inflation_shock", "recovery"], "why": "Iron ore & copper diversification."},
    {"name": "Newmont", "ticker": "NEM", "bucket": "EQUITY", "sector": "Materials", "vision": ["balanced"], "scenarios": ["stagflation", "credit_event"], "why": "Largest gold miner (fiat hedge)."},

    # --- UTILITIES & REAL ESTATE (Income / Rates) ---
    {"name": "NextEra Energy", "ticker": "NEE", "bucket": "EQUITY", "sector": "Utilities", "vision": ["growth", "income"], "scenarios": ["soft_landing"], "why": "Renewables growth + regulated stability."},
    {"name": "Duke Energy", "ticker": "DUK", "bucket": "EQUITY", "sector": "Utilities", "vision": ["income"], "scenarios": ["equity_crash"], "why": "Regulated utility ballast."},
    {"name": "Southern Company", "ticker": "SO", "bucket": "EQUITY", "sector": "Utilities", "vision": ["income"], "scenarios": ["equity_crash"], "why": "Nuclear regulatory completion; steady yield."},
    {"name": "Iberdrola", "ticker": "IBE.MC", "bucket": "EQUITY", "sector": "Utilities", "vision": ["balanced", "income"], "scenarios": ["soft_landing"], "why": "Global renewables leader."},
    {"name": "Realty Income", "ticker": "O", "bucket": "EQUITY", "sector": "Real Estate", "vision": ["income"], "scenarios": ["soft_landing"], "why": "Monthly dividend; retail triple-net lease."},
    {"name": "Prologis", "ticker": "PLD", "bucket": "EQUITY", "sector": "Real Estate", "vision": ["growth", "balanced"], "scenarios": ["recovery"], "why": "Logistics warehousing dominance."},
    {"name": "American Tower", "ticker": "AMT", "bucket": "EQUITY", "sector": "Real Estate", "vision": ["growth", "income"], "scenarios": ["soft_landing"], "why": "Cell tower infra; data growth."},
    {"name": "Public Storage", "ticker": "PSA", "bucket": "EQUITY", "sector": "Real Estate", "vision": ["income", "balanced"], "scenarios": ["stagflation"], "why": "Self-storage pricing power."},
    {"name": "VICI Properties", "ticker": "VICI", "bucket": "EQUITY", "sector": "Real Estate", "vision": ["income"], "scenarios": ["soft_landing"], "why": "Casino/Experiential real estate."},
]

SCENARIO_SHOCKS = {
    "equity_crash": {"EQUITY": -0.30, "FIN": -0.40, "REIT": -0.35, "BONDS": -0.05, "ALT": -0.10, "CASH": 0.0, "CASHLIKE": 0.0},
    "rates_up":     {"EQUITY": -0.10, "FIN": -0.12, "REIT": -0.18, "BONDS": -0.12, "ALT": -0.02, "CASH": 0.0, "CASHLIKE": 0.0},
    "stagflation":  {"EQUITY": -0.15, "FIN": -0.20, "REIT": -0.18, "BONDS": -0.08, "ALT": +0.05, "CASH": 0.0, "CASHLIKE": 0.0},
    "credit_event": {"EQUITY": -0.25, "FIN": -0.35, "REIT": -0.30, "BONDS": -0.07, "ALT": -0.05, "CASH": 0.0, "CASHLIKE": 0.0},
	"bull_market":  {"EQUITY": 0.20, "FIN": 0.25, "REIT": 0.15, "BONDS": 0.05, "ALT": 0.05, "CASH": 0.0, "CASHLIKE": 0.0},
}

@dataclass
class Holding:
    name: str
    weight: float
    kind: str
    ticker: Optional[str] = None
    yield_val: Optional[float] = None

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) for a price series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def excel_col_to_idx(col: str) -> int:
    col = col.strip().upper()
    idx = 0
    for c in col:
        if not ("A" <= c <= "Z"): raise ValueError(f"Bad Excel column: {col}")
        idx = idx * 26 + (ord(c) - ord("A") + 1)
    return idx - 1

def parse_percent_de(value) -> float:
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float, np.floating)):
        return float(value) if value <= 1.5 else float(value) / 100.0
    s = str(value).strip().replace(" ", "").replace("%", "").replace(",", ".")
    try: v = float(s)
    except ValueError: return np.nan
    return v / 100.0 if v > 1.5 else v

def parse_value_clean(value) -> float:
    """Cleans currency strings like '$1,200.00' or '1.200,00' to floats."""
    if pd.isna(value): return 0.0
    if isinstance(value, (int, float, np.floating)): return float(value)
    
    s = str(value).strip()
    # Simple regex to keep digits, dots, commas, minus
    s = re.sub(r"[^\d,\.-]", "", s)
    if not s: return 0.0
    
    # European format heuristic (1.000,00)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): 
            s = s.replace(".", "").replace(",", ".")
        else: 
            s = s.replace(",", "")
    elif "," in s:
        # Check if comma is decimal separator (e.g. 12,50 vs 1,200)
        # If comma is followed by 2 digits, likely decimal
        if len(s.split(",")[-1]) == 2: s = s.replace(",", ".")
        else: s = s.replace(",", "")
        
    try: return float(s)
    except: return 0.0

def classify_name(name: str) -> str:
    n = name.strip().lower()
    if any(k in n for k in CASH_KEYWORDS): return "CASH"
    if any(k in n for k in ALT_KEYWORDS): return "ALT"
    return "TICKER"

def extract_ticker_extended(name: str) -> Optional[str]:
    m = TICKER_RE.search(name)
    if m: return m.group(2)
    m2 = SIMPLE_PAREN_TICKER_RE.search(name.strip())
    if m2: return m2.group(1)
    return None

def get_col_series(df: pd.DataFrame, col_spec: str) -> pd.Series:
    if re.fullmatch(r"[A-Za-z]+", col_spec.strip()):
        idx = excel_col_to_idx(col_spec)
        return df.iloc[:, idx] if idx < df.shape[1] else pd.Series()
    matches = [c for c in df.columns if str(c).strip().lower() == col_spec.strip().lower()]
    return df[matches[0]] if matches else pd.Series()

@st.cache_data(ttl=3600*4)
def fetch_advisor_live_data(tickers: List[str]) -> pd.DataFrame:
    """Batch fetches P/E, RSI, and SMA200 values."""
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="2y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    except: return pd.DataFrame()

    results = []
    try: objs = yf.Tickers(" ".join(tickers))
    except: objs = None

    for t in tickers:
        try:
            # Handle MultiIndex vs Single Index
            if len(tickers) > 1:
                if t in data.columns.get_level_values(0): df_t = data[t].copy()
                else: continue
            else: df_t = data.copy()

            df_t = df_t.dropna(how="all")
            if df_t.empty: continue

            curr_px = df_t["Close"].iloc[-1]
            
            # RSI
            delta = df_t["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # SMA 200
            sma200 = df_t["Close"].rolling(200).mean().iloc[-1]
            
            # PE
            pe = np.nan
            if objs:
                try: pe = objs.tickers[t].info.get('trailingPE') or objs.tickers[t].info.get('forwardPE')
                except: pass
            
            results.append({
                "Ticker": t, "Price": curr_px, "RSI": rsi, 
                "SMA200_Val": sma200, "PE": pe if pe else np.nan
            })
        except: continue
        
    return pd.DataFrame(results).set_index("Ticker")

def fetch_current_prices_batch(tickers: List[str], target_currency="USD") -> Dict[str, float]:
    """
    Ultra-robust price fetcher with Currency Normalization.
    1. Tries Fast Info -> History -> Info to get a price.
    2. Detects currency (e.g. EUR).
    3. Converts to 'target_currency' (default USD) using live FX rates.
    """
    if not tickers: return {}
    price_map = {}
    currency_map = {} # Ticker -> Currency string
    
    # Remove duplicates
    unique_tkrs = list(set(tickers))
    
    # 1. BATCH FETCH DATA
    try:
        tickers_str = " ".join(unique_tkrs)
        dat = yf.Tickers(tickers_str)
        
        # --- A. Get Raw Prices & Currencies ---
        for t in unique_tkrs:
            price = None
            curr = None
            ticker_obj = dat.tickers[t]
            
            # METHOD A: Fast Info (Most efficient)
            try:
                if hasattr(ticker_obj, "fast_info"):
                    # Price
                    price = getattr(ticker_obj.fast_info, "last_price", None)
                    if price is None and "last_price" in ticker_obj.fast_info:
                        price = ticker_obj.fast_info["last_price"]
                    
                    # Currency
                    curr = getattr(ticker_obj.fast_info, "currency", None)
                    if curr is None and "currency" in ticker_obj.fast_info:
                        curr = ticker_obj.fast_info["currency"]
            except: pass
            
            # METHOD B: 1-Day History (Definitive fallback)
            if price is None:
                try:
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        price = hist["Close"].iloc[-1]
                        # History doesn't always give currency easily, fallback to Info
                except: pass

            # METHOD C: Standard Info (Slowest, last resort)
            if price is None or curr is None:
                try:
                    info = ticker_obj.info
                    if price is None:
                        price = info.get("currentPrice") or info.get("regularMarketPreviousClose")
                    if curr is None:
                        curr = info.get("currency")
                except: pass
            
            # Store found data
            if price is not None and price > 0:
                price_map[t] = price
                currency_map[t] = curr.upper() if curr else "USD" # Default to USD if missing
            else:
                print(f"Failed to fetch price for: {t}")

        # --- B. Normalize Currencies ---
        # Identify what currencies we have that are NOT the target
        needed_currencies = set(currency_map.values())
        if target_currency in needed_currencies:
            needed_currencies.remove(target_currency)
        
        # Fetch FX Rates (e.g. "EURUSD=X")
        fx_rates = {}
        if needed_currencies:
            fx_tickers = [f"{c}{target_currency}=X" for c in needed_currencies]
            try:
                fx_dat = yf.Tickers(" ".join(fx_tickers))
                for c in needed_currencies:
                    fx_symbol = f"{c}{target_currency}=X"
                    
                    # Try getting rate
                    rate = None
                    fx_obj = fx_dat.tickers[fx_symbol]
                    try:
                        rate = getattr(fx_obj.fast_info, "last_price", None)
                        if not rate: rate = fx_obj.info.get("regularMarketPrice")
                    except: pass
                    
                    # Fallback for common ones if YF fails
                    if not rate:
                        if c == "EUR": rate = 1.08 # Safety fallback
                        elif c == "GBP": rate = 1.25
                    
                    if rate:
                        fx_rates[c] = rate
            except: pass

        # Apply Conversion
        final_prices = {}
        for t, raw_px in price_map.items():
            c = currency_map.get(t, "USD")
            
            # Handle GBp (Pence) -> GBP -> USD
            if c == "GBP":
                # Some feeds denote pence as GBP with low price, or 'GBp'
                # Logic: If price is massive (e.g. 1500), it's likely pence. 
                # But safer to rely on exact string 'GBp' if provided, or assume standard GBP.
                # Standard YF usually uses 'GBP' for pounds.
                pass 
            elif c == "GBp" or c == "GPB": # Typos or Pence
                raw_px = raw_px / 100.0
                c = "GBP"

            if c == target_currency:
                final_prices[t] = raw_px
            elif c in fx_rates:
                final_prices[t] = raw_px * fx_rates[c]
            else:
                # If we lack FX rate, return raw price but warn (or assuming 1:1)
                final_prices[t] = raw_px 
        
        return final_prices

    except Exception as e:
        print(f"Batch fetch error: {e}")
        return {}

@st.cache_data(ttl=3600*24) # Cache names for 24 hours
def fetch_ticker_names(tickers: List[str]) -> Dict[str, str]:
    """
    Batch fetches long names for tickers (e.g. 'AAPL' -> 'Apple Inc.').
    """
    if not tickers: return {}
    name_map = {}
    
    # yfinance allows batching via Tickers object
    try:
        string_list = " ".join(tickers)
        yt = yf.Tickers(string_list)
        
        for t in tickers:
            try:
                # Access the ticker object from the batch
                ticker_obj = yt.tickers[t]
                # Try fast_info first (faster, no extra request usually)
                name = None
                if hasattr(ticker_obj, "fast_info"):
                    # fast_info is a lazy dictionary in recent yfinance
                    # We try 'shortName' or 'longName' equivalent if available
                    # Note: fast_info keys vary, 'currency'/'exchange' etc are common. 
                    # Often info is safer for names.
                    pass 
                
                # Fallback to standard info (requires API call)
                if not name:
                    info = ticker_obj.info
                    name = info.get("longName") or info.get("shortName") or info.get("displayName")
                
                if name:
                    name_map[t] = name
            except:
                continue
    except Exception:
        pass
        
    return name_map

@st.cache_data(ttl=3600*24)
def fetch_rich_metadata(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches Sector, Country, and Yield for a list of tickers.
    Returns a DataFrame indexed by Ticker.
    """
    if not tickers: return pd.DataFrame()
    
    data = []
    # Batching with yf.Tickers is faster
    try:
        yt = yf.Tickers(" ".join(tickers))
        for t in tickers:
            try:
                info = yt.tickers[t].info
                data.append({
                    "Ticker": t,
                    "Sector": info.get("sector", "Other"),
                    "Country": info.get("country", "Global"),
                    "Yield": info.get("dividendYield", 0.0) or 0.0,
                    "Name": info.get("shortName") or info.get("longName") or t
                })
            except:
                # Fallback if ticker fails
                data.append({"Ticker": t, "Sector": "Unknown", "Country": "Unknown", "Yield": 0.0, "Name": t})
    except: pass
    
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).set_index("Ticker")

def load_portfolio(file_obj, sheet: str, name_col: str, weight_col: str, ticker_col: str, value_col: str = None, yield_col: str = None) -> Tuple[List[Holding], Optional[float]]:
    try:
        df = pd.read_excel(file_obj, sheet_name=sheet, header=0, engine="openpyxl")
    except Exception: return [], None

    name_s = get_col_series(df, name_col)
    w_s = get_col_series(df, weight_col)
    t_s = get_col_series(df, ticker_col) if ticker_col else None
    
    # Optional Columns
    v_s = get_col_series(df, value_col) if value_col else None
    y_s = get_col_series(df, yield_col) if yield_col else None

    if name_s.empty or w_s.empty: return [], None

    holdings = []
    tickers_to_lookup = []
    total_calculated_value = 0.0

    for i, (nm, w) in enumerate(zip(name_s.tolist(), w_s.tolist())):
        nm_str = str(nm).strip() if pd.notna(nm) else ""
        raw_t = t_s.iloc[i] if t_s is not None else None
        tkr_str = str(raw_t).strip() if pd.notna(raw_t) else ""

        if (not nm_str or nm_str.lower() == "nan") and tkr_str and tkr_str.lower() != "nan":
            nm_str = tkr_str.upper()

        if not nm_str or nm_str.lower() == "nan": continue
        wt = parse_percent_de(w)
        if np.isnan(wt): continue
        
        # Accumulate Total Value
        if v_s is not None and len(v_s) > i:
            val = parse_value_clean(v_s.iloc[i])
            total_calculated_value += val
            
        # Parse Manual Yield (Strict / 100)
        manual_yield = None
        if y_s is not None and len(y_s) > i:
            raw_y = y_s.iloc[i]
            # Only process if not empty
            if pd.notna(raw_y) and str(raw_y).strip() != "":
                # 1. Clean the number (handle currency symbols, commas)
                cleaned_y = parse_value_clean(raw_y)
                # 2. Strict Normalization: 5.5 -> 0.055
                manual_yield = cleaned_y / 100.0

        kind = classify_name(nm_str)
        tkr = None
        if kind == "TICKER":
            if tkr_str and tkr_str.lower() != "nan": tkr = tkr_str.upper()
            if not tkr: tkr = extract_ticker_extended(nm_str)
        
        if tkr: 
            tkr = TICKER_MAP.get(tkr.upper(), tkr.upper())
            tickers_to_lookup.append(tkr)

        holdings.append(Holding(name=nm_str, weight=float(wt), kind=kind, ticker=tkr, yield_val=manual_yield))

    if tickers_to_lookup:
        name_map = fetch_ticker_names(list(set(tickers_to_lookup)))
        for h in holdings:
            if h.kind == "TICKER" and h.ticker in name_map:
                if h.name.upper() == h.ticker.upper() or len(h.name) < 6:
                    h.name = name_map[h.ticker]

    total = sum(h.weight for h in holdings)
    if total > 0:
        for h in holdings: h.weight /= total
        
    return holdings, (total_calculated_value if total_calculated_value > 0 else None)

@st.cache_data(ttl=3600*12)
def fetch_prices_for_tickers(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px_data = data["Close"] if "Close" in data.columns else data
    if isinstance(px_data, pd.Series): px_data = px_data.to_frame()
    return px_data.dropna(how="all").ffill().dropna()

def build_price_matrix(holdings: List[Holding], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # FIX: Accept both "TICKER" (from Excel) and "EQUITY" (from Manual Entry)
    valid_kinds = ["TICKER", "EQUITY"]
    
    wanted = sorted({h.ticker for h in holdings if h.kind in valid_kinds and h.ticker})
    
    if not wanted: return pd.DataFrame(), {}
    
    px_all = fetch_prices_for_tickers(wanted, start, end)
    
    resolved = {}
    price_df = pd.DataFrame(index=px_all.index)
    
    # Map back to holding names
    holding_map = {h.name: h.ticker for h in holdings if h.kind in valid_kinds and h.ticker}
    
    for hname, t in holding_map.items():
        if t in px_all.columns:
            price_df[hname] = px_all[t]
            resolved[hname] = t
            
    return price_df.dropna(how="all").ffill().dropna(), resolved

# --- Stats & Charts ---

def bucket_holding(name: str) -> str:
    n = name.lower()
    if any(k in n for k in CASH_KEYWORDS): return "CASH"
    if any(k in n for k in ALT_KEYWORDS): return "ALT"
    if any(x in n for x in ["bond", "govt", "inflation linked"]): return "BONDS"
    if any(x in n for x in ["properties", "reit"]): return "REIT"
    if any(x in n for x in ["bank", "financial"]): return "FIN"
    return "EQUITY"

def get_risk_metrics(price_df, weights):
    rets = price_df.pct_change().dropna()
    cov = rets.cov() * 252
    w = weights.values.reshape(-1, 1)
    port_var = (w.T @ cov.values @ w).item()
    mrc = cov.values @ w
    rc = (w * mrc) / port_var
    rc = pd.Series(rc.flatten(), index=weights.index).sort_values(ascending=False)
    return rc, rets

def calculate_portfolio_impact(current_weights: pd.Series, price_df: pd.DataFrame, scenario: str) -> Dict[str, float]:
    """
    Calculates scenario P&L and Risk Metrics for a given weight distribution.
    Used for Before/After comparisons.
    """
    # 1. Scenario P&L
    shockmap = SCENARIO_SHOCKS.get(scenario, SCENARIO_SHOCKS["equity_crash"])
    scenario_pnl = 0.0
    
    for name, wt in current_weights.items():
        b = bucket_holding(name)
        shock = shockmap.get(b, shockmap.get("EQUITY", 0.0))
        scenario_pnl += wt * shock
        
    # 2. Risk Metrics (Requires Price Data)
    # Filter weights to those present in price_df
    valid_keys = [k for k in current_weights.index if k in price_df.columns]
    
    if len(valid_keys) > 2:
        w_sub = current_weights[valid_keys]
        w_sub = w_sub / w_sub.sum() # renormalize for risk calc
        
        rets = price_df[valid_keys].pct_change().dropna()
        if not rets.empty:
            # Avg Corr
            corr_mat = rets.corr()
            # extract upper triangle off-diagonals
            mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            avg_corr = corr_mat.where(mask).stack().mean()
            
            # Top 5 Risk
            cov = rets.cov() * 252
            w_vec = w_sub.values.reshape(-1, 1)
            port_var = (w_vec.T @ cov.values @ w_vec).item()
            mrc = cov.values @ w_vec
            rc = (w_vec * mrc) / port_var
            top5_risk = np.sum(sorted(rc.flatten(), reverse=True)[:5])
        else:
            avg_corr, top5_risk = np.nan, np.nan
    else:
        avg_corr, top5_risk = np.nan, np.nan
        
    return {
        "scenario_pnl": scenario_pnl,
        "avg_corr": avg_corr,
        "top5_risk": top5_risk
    }

def stress_scenarios_df(holdings: List[Holding]) -> pd.DataFrame:
    buckets = {h.name: bucket_holding(h.name) for h in holdings}
    scenarios = {
        "Equity -30%": {"EQUITY": -0.30, "FIN": -0.40, "REIT": -0.35, "BONDS": -0.05, "ALT": -0.10, "CASH": 0.0},
        "Equity -20%": {"EQUITY": -0.20, "FIN": -0.25, "REIT": -0.22, "BONDS": -0.03, "ALT": -0.06, "CASH": 0.0},
        "Rates +200bp": {"EQUITY": -0.10, "FIN": -0.12, "REIT": -0.18, "BONDS": -0.12, "ALT": -0.02, "CASH": 0.0},
        "Stagflation": {"EQUITY": -0.15, "FIN": -0.20, "REIT": -0.18, "BONDS": -0.08, "ALT": +0.05, "CASH": 0.0},
    }
    rows = []
    for sname, shockmap in scenarios.items():
        pnl = 0.0
        for h in holdings:
            b = buckets.get(h.name, "EQUITY")
            shock = shockmap.get(b, shockmap.get("EQUITY", 0.0))
            pnl += h.weight * shock
        rows.append({"Scenario": sname, "Portfolio Shock P&L": pnl})
    return pd.DataFrame(rows).sort_values("Portfolio Shock P&L")

# --- PLOTLY INTERACTIVE CHARTS ---

def get_interactive_allocation(holdings):
    df = pd.DataFrame([h.__dict__ for h in holdings]).sort_values("weight", ascending=False)
    df = df[df['weight'] > 0]
    fig = px.bar(df, x="name", y="weight", color="kind", 
                 title="Portfolio Allocation", labels={"weight": "Weight", "name": "Holding"},
                 hover_data=["ticker"])
    # INCREASED HEIGHT to 600 and added bottom margin for legend space
    fig.update_layout(xaxis_tickangle=-45, height=600, margin=dict(b=150))
    return fig

def get_interactive_risk(rc_ser):
    df = rc_ser.to_frame(name="Risk Contribution").reset_index().rename(columns={"index": "Ticker"})
    fig = px.bar(df, x="Ticker", y="Risk Contribution", 
                 title="Risk Contribution (Variance Share)",
                 color="Risk Contribution", color_continuous_scale="Reds")
    return fig

def get_interactive_heatmap(price_data):
    corr = price_data.pct_change().dropna().corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Heatmap (Hover for details)")
    return fig

def get_interactive_equity_curve(port_rets, bench_rets=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=("Performance vs Benchmark", "Drawdown"))

    # 1. Cumulative Returns
    port_eq = (1 + port_rets).cumprod()
    fig.add_trace(go.Scatter(x=port_eq.index, y=port_eq, mode='lines', name='Portfolio', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    
    if bench_rets is not None:
        # Align benchmark to portfolio start
        bench_rets = bench_rets.reindex(port_rets.index).fillna(0.0)
        bench_eq = (1 + bench_rets).cumprod()
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq, mode='lines', name='S&P 500 (SPY)', line=dict(color='#ff7f0e', width=1.5, dash='dot')), row=1, col=1)

    # 2. Drawdown
    dd = port_eq / port_eq.cummax() - 1
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode='lines', name='Drawdown', line=dict(color='#d62728', width=1), fill='tozeroy'), row=2, col=1)

    fig.update_yaxes(title_text="Growth ($1 inv)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(hovermode="x unified", height=600)
    return fig

def get_rolling_stats_fig(port_rets, window=63): # 63d = ~3 months
    # Rolling Vol
    vol = port_rets.rolling(window).std() * np.sqrt(252)
    # Rolling Sharpe (Simplified)
    ret = port_rets.rolling(window).mean() * 252
    sharpe = ret / vol
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"Rolling {window}d Volatility", f"Rolling {window}d Sharpe Ratio"))
    
    # Vol - ORANGE for visibility in both modes
    fig.add_trace(go.Scatter(x=vol.index, y=vol, name="Vol (Ann)", line=dict(color='#FFA15A', width=2)), row=1, col=1)
    fig.add_hline(y=vol.mean(), line_dash="dash", line_color="gray", annotation_text="Avg", row=1, col=1)
    
    # Sharpe - TEAL for visibility
    fig.add_trace(go.Scatter(x=sharpe.index, y=sharpe, name="Sharpe", line=dict(color='#00CC96', width=2)), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=1)
    
    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    return fig

def get_attribution_fig(holdings, scenario_name="Equity -30%"):
    # Re-use bucket logic locally
    scenarios = {
        "Equity -30%": {"EQUITY": -0.30, "FIN": -0.40, "REIT": -0.35, "BONDS": -0.05, "ALT": -0.10, "CASH": 0.0},
        "Rates +200bp": {"EQUITY": -0.10, "FIN": -0.12, "REIT": -0.18, "BONDS": -0.12, "ALT": -0.02, "CASH": 0.0},
        "Stagflation": {"EQUITY": -0.15, "FIN": -0.20, "REIT": -0.18, "BONDS": -0.08, "ALT": +0.05, "CASH": 0.0},
        "Credit Event": {"EQUITY": -0.25, "FIN": -0.35, "REIT": -0.30, "BONDS": -0.07, "ALT": -0.05, "CASH": 0.0},
    }
    shockmap = scenarios.get(scenario_name, scenarios["Equity -30%"])
    
    rows = []
    for h in holdings:
        if h.kind in ["ALT"]: continue # optional skip
        b = bucket_holding(h.name)
        shock = shockmap.get(b, shockmap.get("EQUITY", 0.0))
        contrib = h.weight * shock
        rows.append({"Holding": h.name, "P&L": contrib})
    
    df = pd.DataFrame(rows).sort_values("P&L").head(15)
    # Using Plotly for attribution too
    fig = px.bar(df, x="P&L", y="Holding", orientation='h', title=f"{scenario_name}", color="P&L", color_continuous_scale="RdBu")
    return fig

def generate_report_html(context):
    """
    Generates a simple HTML dashboard from the context data (metrics + figures).
    """
    # Helper to convert plotly fig to html div
    def fig_to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content = f"""
    <html>
    <head>
        <title>Portfolio Stress Test Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            .metric-box {{ display: inline-block; padding: 20px; background: #f0f2f6; margin: 10px; border-radius: 8px; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #000; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .chart-container {{ margin-bottom: 50px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Portfolio Stress Test Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Key Metrics</h2>
        <div>
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-val">{context['mdd']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">VaR (95%)</div>
                <div class="metric-val">{context['var']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Ann. Volatility</div>
                <div class="metric-val">{context['vol']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-val">{context['sharpe']}</div>
            </div>
        </div>

        <h2>Performance & Risk</h2>
        <div class="chart-container">{fig_to_html(context['fig_eq'])}</div>
        <div class="chart-container">{fig_to_html(context['fig_roll'])}</div>
        
        <h2>Scenario Analysis</h2>
        <p>Projected impact of specific market shocks:</p>
        {"".join([f'<div class="chart-container">{fig_to_html(f)}</div>' for f in context['scenario_figs']])}

        <hr>
        <p style="text-align: center; color: #888;">Portfolio Stress Lab | Generated locally.</p>
        
        <script>
            // Auto open print dialog optionally
            // window.print();
        </script>
    </body>
    </html>
    """
    return html_content

def generate_inspection_report(df, metrics, fig_pie, fig_tree):
    """Generates HTML report for Portfolio Inspection."""
    def fig_to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs='cdn') if fig else ""

    # Convert Dataframe to HTML Table
    table_html = df.to_html(classes="table", float_format="{:.2%}".format)

    html = f"""
    <html>
    <head>
        <title>Portfolio Inspection Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; color: #333; }}
            .metric {{ display: inline-block; padding: 15px; background: #f0f2f6; margin-right: 10px; border-radius: 5px; }}
            .val {{ font-size: 20px; font-weight: bold; }}
            .label {{ font-size: 12px; color: #666; }}
            .chart {{ margin-top: 30px; border: 1px solid #eee; padding: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Portfolio Inspection Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
        
        <div style="margin-bottom: 30px;">
            <div class="metric"><div class="label">Holdings</div><div class="val">{metrics['count']}</div></div>
            <div class="metric"><div class="label">Est. Yield</div><div class="val">{metrics['yield']}</div></div>
            <div class="metric"><div class="label">Top Sector</div><div class="val">{metrics['sector']}</div></div>
        </div>

        <div class="chart">{fig_to_html(fig_pie)}</div>
        <div class="chart">{fig_to_html(fig_tree)}</div>
        
        <h3>Holdings Detail</h3>
        {table_html}
    </body>
    </html>
    """
    return html

def suggest_rebalance_robust(holdings, cash_buffer, max_weight, new_adds=None, entry_size=0.03):
    """
    Suggests trades to rebalance portfolio.
    new_adds: List of dicts [{'ticker': 'AAPL', 'name': 'Apple'}] to simulate adding.
    entry_size: Target weight for new additions.
    """
    # 1. Prepare Dataframe
    df = pd.DataFrame([h.__dict__ for h in holdings])
    
    # Merge Cash
    df.loc[df["kind"] == "CASH", "name"] = "CASH (Total)"
    df.loc[df["kind"] == "CASH", "ticker"] = "CASH" 

    # Group existing
    df_grouped = df.groupby("name", as_index=False).agg({
        "weight": "sum", "kind": "first", "ticker": "first"
    })
    
    # 2. Inject "Ghost" Positions (New Adds)
    if new_adds:
        for item in new_adds:
            # Only add if not already in portfolio
            if item['name'] not in df_grouped['name'].values:
                new_row = pd.DataFrame([{
                    "name": item['name'], 
                    "weight": 0.0, 
                    "kind": "EQUITY", 
                    "ticker": item['ticker']
                }])
                df_grouped = pd.concat([df_grouped, new_row], ignore_index=True)

    w = df_grouped.set_index("name")["weight"]
    
    # Filters
    is_cash = df_grouped.set_index("name")["kind"] == "CASH"
    is_alt  = df_grouped.set_index("name")["kind"] == "ALT"
    
    # Track New Adds for special handling
    new_names = [x['name'] for x in new_adds] if new_adds else []
    
    target = w.copy()
    
    # ---------------------------------------------------------
    # STEP 1: CAP MAX WEIGHT (Generate Cash)
    # ---------------------------------------------------------
    excess_cash = 0.0
    for name, wt in w.sort_values(ascending=False).items():
        if is_alt.get(name, False): continue
        if wt > max_weight and not is_cash.get(name, False):
            sell_amt = wt - max_weight
            target[name] -= sell_amt
            excess_cash += sell_amt

    cash_idx = target[is_cash].index
    if not cash_idx.empty:
        target[cash_idx[0]] += excess_cash

    # ---------------------------------------------------------
    # STEP 2: DISTRIBUTE CASH (Waterfall)
    # ---------------------------------------------------------
    current_cash_sum = target[is_cash].sum()
    
    # Waterfall A: Ensure Buffer
    if current_cash_sum < cash_buffer:
        shortfall = cash_buffer - current_cash_sum
        liquid = target[~is_cash & ~is_alt].sort_values(ascending=False)
        for name, wt in liquid.items():
            if shortfall <= 1e-9: break
            sell = min(shortfall, wt)
            target[name] -= sell
            shortfall -= sell
            if not cash_idx.empty: target[cash_idx[0]] += sell
            
    # Waterfall B: Invest Surplus
    elif current_cash_sum > cash_buffer + 0.01:
        surplus = current_cash_sum - cash_buffer
        
        # Priority 1: Fund New Candidates (Ghost Positions)
        # We give them their 'entry_size' immediately if we have cash
        for name in new_names:
            if surplus <= 0: break
            # Cost to buy full entry size
            cost = entry_size 
            buy = min(surplus, cost)
            
            target[name] += buy
            surplus -= buy
            if not cash_idx.empty: target[cash_idx[0]] -= buy
            
        # Priority 2: Distribute Remaining Surplus to Existing Undercaps
        if surplus > 0.005: # If meaningful cash left
            eligible = target[~is_cash & ~is_alt & ~target.index.isin(new_names)]
            eligible = eligible[eligible < max_weight]
            
            if eligible.sum() > 0:
                for name, wt in eligible.items():
                    room = max_weight - wt
                    # Proportional fill
                    share = (wt / eligible.sum()) * surplus
                    buy = min(share, room)
                    target[name] += buy
                    if not cash_idx.empty: target[cash_idx[0]] -= buy
                    
                    if not cash_idx.empty and target[cash_idx[0]] < cash_buffer: 
                        target[cash_idx[0]] += buy; target[name] -= buy; break

    # ---------------------------------------------------------
    # STEP 3: OUTPUT
    # ---------------------------------------------------------
    target = target / target.sum()
    meta = df_grouped.set_index("name")[["ticker"]]
    res = pd.DataFrame({"Current": w, "Target": target}).join(meta)
    
    res["Delta"] = res["Target"] - res["Current"]
    
    conditions = [res["Delta"] > 0.002, res["Delta"] < -0.002]
    res["Action"] = np.select(conditions, ["BUY", "SELL"], default="-")
    
    res["SortKey"] = res["Action"].map({"BUY": 0, "SELL": 1, "-": 2})
    return res.sort_values(["SortKey", "Delta"], ascending=[True, False]).drop(columns="SortKey")

def advisor_rank_with_impact(holdings, price_df, vision, scenario, add_weight, sectors=None):
    """
    Smart Advisor 4.0: 
    - Risk/Reward Trade-off analysis.
    - Contextual MA (Distance to SMA).
    - Score factors in Upside Potential.
    """
    # 1. Base Stats
    base_w = pd.Series({h.name: h.weight for h in holdings})
    
    # Calculate Base Risk (Selected Scenario) AND Base Reward (Bull Market)
    base_risk = calculate_portfolio_impact(base_w, price_df, scenario)["scenario_pnl"]
    base_reward = calculate_portfolio_impact(base_w, price_df, "bull_market")["scenario_pnl"]
    
    # Sector Exposure
    current_tickers = [h.ticker for h in holdings if h.ticker]
    meta = fetch_rich_metadata(current_tickers)
    current_sector_weights = {}
    if not meta.empty:
        ticker_sector_map = meta["Sector"].to_dict()
        for h in holdings:
            s = "Unknown"
            if h.ticker and h.ticker in ticker_sector_map: s = ticker_sector_map[h.ticker]
            elif h.kind == "CASH": s = "Cash"
            current_sector_weights[s] = current_sector_weights.get(s, 0) + h.weight

    rows = []
    exist_tkrs = {h.ticker for h in holdings if h.ticker}
    
    candidates = []
    for c in ADVISOR_PALETTE:
        if vision not in c["vision"]: continue
        if scenario not in c["scenarios"]: continue
        if c["ticker"] in exist_tkrs: continue
        if sectors and c["sector"] not in sectors: continue
        candidates.append(c)
        
    if not candidates: return pd.DataFrame()
    
    live_data = fetch_advisor_live_data([c["ticker"] for c in candidates])
    
    for c in candidates:
        t = c["ticker"]
        sec = c["sector"]
        
        # --- A. Simlate Trade ---
        new_w = base_w.copy()
        new_w[c["name"]] = add_weight
        cash_names = [h.name for h in holdings if h.kind == "CASH"]
        if cash_names: new_w[cash_names[0]] = max(0, new_w[cash_names[0]] - add_weight)
        new_w = new_w / new_w.sum()
        
        # --- B. Calculate Deltas ---
        # 1. Risk Delta (Did we make the crash worse?)
        new_risk = calculate_portfolio_impact(new_w, price_df, scenario)["scenario_pnl"]
        delta_risk = new_risk - base_risk # Usually negative (worse) if buying equity
        
        # 2. Reward Delta (Did we improve upside?)
        new_reward = calculate_portfolio_impact(new_w, price_df, "bull_market")["scenario_pnl"]
        delta_reward = new_reward - base_reward # Usually positive (better)
        
        # --- C. Scoring ---
        score = 50.0 
        
        # Reward the trade if Upside > Downside risk added
        # Multiplier arbitrary to scale to 0-100
        net_benefit = (delta_reward * 1.2) + delta_risk 
        if net_benefit > 0: score += 15
        
        # Sector Diversity
        curr_sec_w = current_sector_weights.get(sec, 0.0)
        if curr_sec_w > 0.30: score -= 30 
        elif curr_sec_w > 0.20: score -= 15
        elif curr_sec_w < 0.05: score += 10
            
        # Technicals
        rsi_val = 50
        pe_val = np.nan
        ma_dist_str = "-"
        
        if t in live_data.index:
            row = live_data.loc[t]
            rsi_val = row["RSI"]
            pe_val = row["PE"]
            px = row["Price"]
            sma = row["SMA200_Val"]
            
            # RSI
            if rsi_val < 35: score += 15 
            elif rsi_val > 75: score -= 15
            
            # MA Distance (Contextual)
            if pd.notna(sma) and pd.notna(px) and sma > 0:
                dist = (px - sma) / sma
                color_dot = "🟢" if dist > 0 else "🔴"
                ma_dist_str = f"{color_dot} {dist:+.1%}"
                
                # Trend Score
                if dist > 0: score += 10 # Uptrend
                if dist < -0.15: score += 5 # Deep value / Mean reversion candidate?
            
            # Valuation
            if pd.notna(pe_val):
                if pe_val < 20: score += 10 
                elif pe_val > 60: score -= 10
                
        score = max(0, min(100, score))
        
        # Readable Impact String
        # Shows: "📉 -0.2% | 📈 +0.5%"
        risk_str = f"📉 {delta_risk*100:+.2f}%"
        rew_str = f"📈 {delta_reward*100:+.2f}%"
        
        # Highlight the "Net" effect simply for the user
        impact_str = f"{risk_str} | {rew_str}"
            
        rows.append({
            "Score": int(score),
            "Ticker": t,
            "Name": c["name"],
            "Sector": sec,
            "Sector_Exp": curr_sec_w,
            "RSI": f"{rsi_val:.0f}",
            "P/E": f"{pe_val:.1f}" if pd.notna(pe_val) else "-",
            "vs SMA200": ma_dist_str, # New Contextual Column
            "Risk/Reward": impact_str, # New Combined Column
            "Why": c["why"],
            "Raw_Delta_Risk": delta_risk,
            "Raw_Delta_Reward": delta_reward
        })
        
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Score", "Raw_Delta_Reward"], ascending=[False, False])

# ==========================================
# 3. STREAMLIT APP
# ==========================================

st.set_page_config(page_title="Portfolio Stress Lab", layout="wide", page_icon="📈")
st.title("📈 Portfolio Stress Lab")

# --- 1. SIDEBAR NAVIGATION (MOVED TO TOP) ---
st.sidebar.header("Navigation")

# Initialize State
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🔎 Inspection"

# Render Menu at the VERY TOP of the sidebar
page = st.sidebar.radio(
    "Go to", 
    ["🔎 Inspection", "📉 Stress Lab", "⚖️ Advisor & Actions", "🔬 Stock Screener"], 
    key="active_page",
    label_visibility="collapsed" # Clean look (hides the "Go to" label)
)

st.sidebar.markdown("---")

# --- 2. SIDEBAR DATA IMPORT (APPEARS BELOW NAV) ---
st.sidebar.header("📂 Data Import")
input_method = st.sidebar.radio("Input Method:", ["Excel Upload", "Manual Entry (Qty)"])

# 1. ALWAYS Initialize holdings to empty first
holdings = []

# 2. LOAD FROM STATE IMMEDIATELY (Persistence)
if input_method == "Manual Entry (Qty)" and "manual_holdings" in st.session_state:
    holdings = st.session_state["manual_holdings"]

if input_method == "Excel Upload":
    if "manual_holdings" in st.session_state: del st.session_state["manual_holdings"]
    
    uploaded_file = st.sidebar.file_uploader("Upload Portfolio (.xlsx)", type=["xlsx"])
    sheet_name = "Portfolio"
    
    if uploaded_file:
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            uploaded_file.seek(0)
            
            with st.sidebar.expander("Column Mapping", expanded=False):
                name_col = st.text_input("Name Column", "A")
                ticker_col = st.text_input("Ticker Column", "B")
                weight_col = st.text_input("Weight Column", "AQ")
                value_col = st.text_input("Value Column (Optional)", "H")
                yield_col = st.text_input("Yield Column (Optional)", "V")
            
            # NEW: Assumption Toggle
            excel_curr = st.sidebar.radio("Excel Values are in:", ["EUR (€)", "USD ($)"], horizontal=True)
            
            # Modified call to accept value_col
            holdings, calc_total = load_portfolio(uploaded_file, sheet_name, name_col, weight_col, ticker_col, value_col, yield_col)
            
            if calc_total:
                # NORMALIZE TO USD FOR INTERNAL STORAGE
                # The app uses USD as the base for the "portfolio_total_value" state variable.
                final_total_usd = calc_total
                
                if "EUR" in excel_curr:
                    # Fetch FX Rate to convert Input EUR -> Base USD
                    try:
                        fx_obj = yf.Ticker("EURUSD=X")
                        rate = None
                        if hasattr(fx_obj, "fast_info"):
                            rate = getattr(fx_obj.fast_info, "last_price", None)
                        if not rate: 
                            rate = fx_obj.info.get("regularMarketPrice")
                        
                        # Fallback
                        if not rate: rate = 1.08
                        
                        final_total_usd = calc_total * rate
                    except:
                        # Fallback if fetch fails
                        final_total_usd = calc_total * 1.08
                
                st.session_state["portfolio_total_value"] = final_total_usd
                
            elif "portfolio_total_value" in st.session_state:
                # Clear legacy value if we switched files and no value col found
                del st.session_state["portfolio_total_value"]
                
        except Exception as e: st.sidebar.error(f"Error reading Excel: {e}")

elif input_method == "Manual Entry (Qty)":
    st.sidebar.info("Enter your share counts. We'll fetch prices to calculate portfolio weights.")
    
    # Inputs
    total_cash = st.sidebar.number_input("Total Cash Position ($)", min_value=0.0, value=10000.0, step=500.0)
    
    raw_holdings = st.sidebar.text_area(
        "Holdings (Ticker, Quantity)", 
        value="AAPL, 10\nMSFT, 5\nNVDA, 20",
        help="Format: TICKER, SHARES (one per line). Example: \nAAPL, 10\nMSFT, 5"
    )
    
    if st.sidebar.button("Build Portfolio"):
        with st.spinner("Fetching data..."):
            # A. PARSE INPUTS
            lines = [line.strip() for line in raw_holdings.split('\n') if line.strip()]
            parsed_data = []
            unique_tickers = set()
            debug_log = []

            for line in lines:
                if "," in line: parts = line.split(',')
                else: parts = line.split()
                
                if len(parts) >= 2:
                    raw_tkr = parts[0].strip().upper()
                    final_tkr = TICKER_MAP.get(raw_tkr, raw_tkr)
                    try:
                        qty_str = re.sub(r"[^0-9\.]", "", parts[1])
                        qty = float(qty_str)
                        if qty > 0:
                            parsed_data.append({"ticker": final_tkr, "qty": qty})
                            unique_tickers.add(final_tkr)
                            debug_log.append(f"✅ Parsed: {final_tkr} x {qty}")
                        else:
                            debug_log.append(f"⚠️ Skipped {final_tkr}: Qty 0")
                    except: 
                        debug_log.append(f"❌ Error parsing line: {line}")
            
            # B. FETCH DATA
            tkr_list = list(unique_tickers)
            if not tkr_list:
                st.sidebar.error("No valid tickers found.")
            else:
                price_map = fetch_current_prices_batch(tkr_list)
                name_map = fetch_ticker_names(tkr_list)
                
                # C. BUILD PORTFOLIO
                equity_val = 0.0
                temp_holdings = []
                
                for p in parsed_data:
                    tkr = p["ticker"]
                    px = price_map.get(tkr, 0.0)
                    if px > 0:
                        val = px * p["qty"]
                        equity_val += val
                        display_name = name_map.get(tkr, tkr)
                        temp_holdings.append({"name": display_name, "ticker": tkr, "value": val, "kind": "EQUITY"})
                        debug_log.append(f"💰 Priced: {tkr} @ ${px:.2f} = ${val:.2f}")
                    else:
                        debug_log.append(f"⚠️ Price Not Found: {tkr}")

                # D. FINALIZE
                total_port_val = equity_val + total_cash
                new_holdings = []
                
                if total_port_val > 0:
                    for h in temp_holdings:
                        w = h["value"] / total_port_val
                        new_holdings.append(Holding(name=h["name"], ticker=h["ticker"], weight=w, kind="EQUITY"))
                    
                    if total_cash > 0:
                        w_cash = total_cash / total_port_val
                        new_holdings.append(Holding(name="CASH (Input)", ticker="CASH", weight=w_cash, kind="CASH"))
                    
                    # SAVE EVERYTHING TO STATE
                    st.session_state["manual_holdings"] = new_holdings
                    st.session_state["portfolio_total_value"] = total_port_val  # <--- NEW: SAVING TOTAL VALUE
                    st.rerun()
                else:
                    st.sidebar.error("Total Value is 0. Check inputs.")
                    with st.sidebar.expander("Debug Log"):
                        for l in debug_log: st.write(l)

# --- SIMULATION SETTINGS ---
# (Keep this section unique!)
if "sim_settings_shown" not in st.session_state:
    st.sidebar.header("⚙️ Simulation")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-01"), key="sim_start_date_sidebar")
    alpha_conf = st.sidebar.slider("VaR Significance Level (Alpha)", 0.01, 0.10, 0.05, 0.01)
    st.sidebar.caption(f"Calculating **{1.0 - alpha_conf:.0%} Confidence VaR**.")

# --- FINAL CHECK ---
if not holdings:
    st.info("👋 Please upload an Excel file OR use 'Manual Entry' to build your portfolio.")
    st.stop()

# --- CHECK LOAD ---
if not holdings:
    st.info("👋 Please upload an Excel file OR use 'Manual Entry' to build your portfolio.")
    st.stop()

# --- INSPECTION ---
if page == "🔎 Inspection":
    # 1. Prepare Data
    df_h = pd.DataFrame([h.__dict__ for h in holdings])
    df_h["kind"] = df_h["kind"].replace("TICKER", "EQUITY")
    
    tkrs = df_h[df_h["kind"] == "EQUITY"]["ticker"].dropna().unique().tolist()
    
    with st.spinner("Fetching Sector & Yield data..."):
        meta_df = fetch_rich_metadata(tkrs)
    
    # Merge metadata
    df_merged = df_h.merge(meta_df, left_on="ticker", right_index=True, how="left")
    
    # Safety: Ensure cols exist
    for col in ["Sector", "Yield", "Country"]:
        if col not in df_merged.columns: df_merged[col] = np.nan

    # Sector Cleaning
    def clean_sector(row):
        sec = row["Sector"] if pd.notna(row["Sector"]) else ""
        name = str(row["name"]).lower()
        if sec and sec not in ["Unknown", "Other", ""]: return sec
        if any(x in name for x in ["bond", "govt", "treasury", "fixed income", "gilt", "bund"]): return "Fixed Income"
        if "reit" in name or "real estate" in name: return "Real Estate"
        if "gold" in name or "silver" in name or "precious" in name: return "Commodities"
        if row["kind"] == "EQUITY": return "Unclassified Equity"
        return row["kind"]

    df_merged["Sector"] = df_merged.apply(clean_sector, axis=1)
    
    # Yield Fix
    df_merged["Yield"] = df_merged["Yield"].fillna(0.0)
    if df_merged["Yield"].max() > 1.0: 
        df_merged["Yield"] = df_merged["Yield"] / 100.0
    
    if "yield_val" in df_merged.columns:
        # We override the 'Yield' column ONLY where:
        # 1. We have a manual yield (yield_val is not NaN)
        # 2. The position is CASH (This preserves live data for Equities)
        mask_cash = (df_merged["kind"] == "CASH") & (df_merged["yield_val"].notna())
        df_merged.loc[mask_cash, "Yield"] = df_merged.loc[mask_cash, "yield_val"]

    # --- CURRENCY TOGGLE & VALUE CALCULATION ---
    
    # 1. Fetch Live Exchange Rate (EUR/USD)
    fx_rate = 1.08 # Fallback default
    try:
        fx_obj = yf.Ticker("EURUSD=X")
        if hasattr(fx_obj, "fast_info"):
            f = fx_obj.fast_info.last_price
            if f: fx_rate = f
    except: pass
    
    # 2. Controls Layout
    c_inv, c_curr = st.columns([2, 1])
    
    # Currency Toggle
    curr_view = c_curr.radio("View Currency", ["USD ($)", "EUR (€)"], horizontal=True)
    is_eur = "EUR" in curr_view
    
    # 3. Determine Total Value
    # Check if we have a hard-calculated value from data import
    has_calculated_total = "portfolio_total_value" in st.session_state
    base_val = float(st.session_state.get("portfolio_total_value", 10000.0))
    
    # Apply FX conversion if viewing in EUR
    display_val = base_val / fx_rate if is_eur else base_val
    curr_sym = "€" if is_eur else "$"

    if has_calculated_total:
        # READ-ONLY MODE: Data provided exact values
        c_inv.metric(f"Total Portfolio Value ({'EUR' if is_eur else 'USD'})", f"{curr_sym}{display_val:,.2f}")
        port_total_val = display_val
    else:
        # SIMULATION MODE: User sets the scale
        port_total_val = c_inv.number_input(
            f"Total Portfolio Value ({'EUR' if is_eur else 'USD'})", 
            value=display_val, 
            step=1000.0, 
            format="%.2f",
            help="Enter a total value to simulate dollar amounts for your percentage-based portfolio."
        )
    
    # 4. Calculate Column
    df_merged["Value"] = df_merged["weight"] * port_total_val
    
    # 5. Metrics
    port_yield = (df_merged["weight"] * df_merged["Yield"]).sum()
    equity_subset = df_merged[df_merged["kind"]=="EQUITY"]
    top_sec = equity_subset["Sector"].mode()[0] if not equity_subset.empty else "N/A"
    
    m1, m2, m3, m4 = st.columns([1,1,1,1])
    m1.metric("Total Holdings", len(df_h))
    m2.metric("Est. Total Yield", f"{port_yield:.2%}")
    m3.metric("Top Sector", top_sec)

    # 6. Visuals
    c_left, c_right = st.columns([1.4, 1])
    
    with c_left:
        st.subheader("Holdings Detail")
        display_cols = ["name", "ticker", "weight", "Value", "Sector", "Yield"]
        valid_cols = [c for c in display_cols if c in df_merged.columns]
        
        # Dynamic Formatting Symbol
        curr_sym = "€" if is_eur else "$"
        
        st.dataframe(
            df_merged[valid_cols].sort_values("weight", ascending=False)
            .style.format({
                "weight": "{:.2%}", 
                "Yield": "{:.2%}",
                "Value": curr_sym + "{:,.2f}" # Dynamic Currency Symbol
            })
            .background_gradient(subset=["weight"], cmap="Greens"),
            height=600, use_container_width=True
        )

    with c_right:
        st.subheader("Allocation Analysis")
        fig_pie = px.pie(df_merged, values="weight", names="kind", hole=0.4, 
                         title="Asset Class Weights", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        fig_tree = None
        if not equity_subset.empty:
            equity_subset = equity_subset.copy()
            equity_subset["ShortLabel"] = equity_subset["ticker"].fillna(equity_subset["name"])
            fig_tree = px.treemap(
                equity_subset, path=[px.Constant("All Holdings"), "Sector", "ShortLabel"], 
                values="weight", title="Sector Diversification",
                color="Sector", color_discrete_sequence=px.colors.qualitative.Prism,
                hover_name="name", hover_data={"weight": ":.2%", "ShortLabel": False, "Sector": False}
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Add Equities to see Sector Breakdown.")

    # 7. Download Report Logic
    with m4:
        st.write("") 
        valid_cols = [c for c in display_cols if c in df_merged.columns]
        
        insp_html = generate_inspection_report(
            df_merged[valid_cols].sort_values("weight", ascending=False),
            {"count": len(df_h), "yield": f"{port_yield:.2%}", "sector": top_sec},
            fig_pie, fig_tree
        )
        st.download_button("📥 Download Report", insp_html, "Inspection_Report.html", "text/html")

# --- STRESS LAB ---
if page == "📉 Stress Lab":
    # 1. Controls (Top Row) - Added 3rd column for Download Button
    c_btn, c_bm, c_dl = st.columns([1, 1.5, 1])
    
    refresh_stress = c_btn.button("🔄 Refresh Stress Test")
    show_spy = c_bm.checkbox("Compare vs S&P 500 (SPY)", value=True)
    
    # Placeholder: Reserves space at the top right for the button
    dl_placeholder = c_dl.empty()

    # 2. Initialize State
    if "stress_results" not in st.session_state:
        st.session_state["stress_results"] = None

    # 3. Auto-Run Logic
    should_run = (st.session_state["stress_results"] is None) or refresh_stress

    if should_run:
        with st.spinner("Crunching market data..."):
            price_df, resolved = build_price_matrix(holdings, str(start_date), None)
            
            bench_rets = None
            if show_spy:
                try:
                    spy_df = fetch_prices_for_tickers(["SPY"], str(start_date), None)
                    if not spy_df.empty:
                        s_col = spy_df["SPY"] if "SPY" in spy_df.columns else spy_df.iloc[:, 0]
                        bench_rets = s_col.pct_change().dropna()
                except: pass

            if not price_df.empty:
                w_ser = pd.Series({h.name: h.weight for h in holdings if h.name in price_df.columns})
                w_ser = w_ser / w_ser.sum()
                rc_ser, rets = get_risk_metrics(price_df, w_ser)
                port_rets = (rets * w_ser).sum(axis=1)
                
                st.session_state["stress_results"] = {
                    "price_df": price_df,
                    "rc_ser": rc_ser,
                    "port_rets": port_rets,
                    "bench_rets": bench_rets
                }
            else:
                st.warning("No price data found.")

    # 4. Render Dashboard
    if st.session_state["stress_results"]:
        data = st.session_state["stress_results"]
        port_rets = data["port_rets"]
        
        # --- A. Calculate Metrics (First) ---
        dd = (1+port_rets).cumprod() / (1+port_rets).cumprod().cummax() - 1
        var_val = port_rets.mean() + norm.ppf(alpha_conf) * port_rets.std()
        ann_ret = port_rets.mean() * 252
        ann_vol = port_rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        downside_rets = port_rets[port_rets < 0]
        downside_vol = downside_rets.std() * np.sqrt(252)
        sortino = ann_ret / downside_vol if downside_vol > 0 else 0
        
        # --- B. Generate Figures (First) ---
        fig_eq = get_interactive_equity_curve(port_rets, data["bench_rets"])
        fig_roll = get_rolling_stats_fig(port_rets)
        
        scenario_figs = []
        s_keys = list(SCENARIO_SHOCKS.keys())
        for s in s_keys:
            map_name = {"equity_crash": "Equity -30%", "rates_up": "Rates +200bp", "stagflation": "Stagflation", "credit_event": "Credit Event"}
            scenario_figs.append(get_attribution_fig(holdings, map_name.get(s, s)))

        # --- C. Generate Report & Fill Top Placeholder ---
        report_html = generate_report_html({
            "mdd": f"{dd.min():.2%}",
            "var": f"{var_val:.2%}",
            "vol": f"{ann_vol:.1%}",
            "sharpe": f"{sharpe:.2f}",
            "fig_eq": fig_eq,
            "fig_roll": fig_roll,
            "scenario_figs": scenario_figs
        })
        
        # This renders the button in the top right corner
        dl_placeholder.download_button(
            label="📄 Download Report (HTML)",
            data=report_html,
            file_name="Portfolio_Stress_Report.html",
            mime="text/html",
            help="Download a full report. Open the file and Print to PDF (Ctrl+P) to save."
        )

        # --- D. Render Page Layout ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Max Drawdown", f"{dd.min():.2%}", help="Worst peak-to-trough decline.")
        k2.metric(f"VaR ({alpha_conf:.0%})", f"{var_val:.2%}", help="Estimated worst daily loss.")
        k3.metric("Ann. Volatility", f"{ann_vol:.1%}", help="Standard deviation of returns.")
        k4.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Return per unit of total risk.")
        k5.metric("Sortino Ratio", f"{sortino:.2f}", help="Return per unit of downside risk.")

        st.markdown("### 📈 Performance")
        st.plotly_chart(fig_eq, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Correlation Heatmap")
            st.caption("Hover over cells to see correlation values.")
            st.plotly_chart(get_interactive_heatmap(data["price_df"]), use_container_width=True)
        with c2:
            st.markdown("#### Risk Contributors")
            st.plotly_chart(get_interactive_risk(data["rc_ser"]), use_container_width=True)

        st.markdown("#### Rolling Regime (3-Month Window)")
        st.plotly_chart(fig_roll, use_container_width=True)
        
        st.info("""
        **How to read this chart:**
        * **Orange Line (Volatility):** Measures "Fear." Spikes indicate instability. 
        * **Teal Line (Sharpe):** Measures "Efficiency." Drops below 0 mean risk without reward.
        """)
        
        st.markdown("---")
        st.markdown("### 🌪️ Scenario Analysis")
        st.caption("Projected impact of various market shocks. (Showing top detractors per scenario)")

        # Render ALL Scenarios sequentially in a grid
        for i in range(0, len(scenario_figs), 2):
            cols = st.columns(2)
            cols[0].plotly_chart(scenario_figs[i], use_container_width=True)
            
            if i+1 < len(scenario_figs):
                cols[1].plotly_chart(scenario_figs[i+1], use_container_width=True)

# --- ADVISOR & ACTIONS ---
if page == "⚖️ Advisor & Actions":
    st.header("⚖️ Portfolio Actions & Strategy")
    
    # ==========================================
    # 1. RULES-BASED REBALANCING
    # ==========================================
    st.subheader("1. Maintenance Rebalancing")
    st.caption("Automated hygiene trades to keep risk limits and cash buffers in check.")
    
    c_set1, c_set2, c_set3 = st.columns(3)
    c_buff = c_set1.slider("Min Cash Buffer", 0.0, 0.8, 0.35, help="Minimum % of portfolio to keep in Cash.")
    m_w = c_set2.slider("Max Position Cap", 0.03, 0.5, 0.05, help="Maximum % allowed for any single Equity position.")
    
    # NEW: Candidate Selector (Uses ALL options, Sorted Alphabetically)
    # This allows you to simulate adding ANY stock, regardless of your Advisor filters below.
    sorted_palette = sorted(ADVISOR_PALETTE, key=lambda x: x['ticker'])
    adv_options = [f"{c['ticker']} - {c['name']}" for c in sorted_palette]
    
    sel_cands = c_set3.multiselect(
        "Simulate Adding (What-If)", 
        adv_options, 
        help="Select stocks to inject into the calculation (Target Size: 3%)."
    )
    
    # Parse selection back to list of dicts
    new_adds_list = []
    if sel_cands:
        for s in sel_cands:
            parts = s.split(" - ", 1)
            if len(parts) == 2:
                new_adds_list.append({"ticker": parts[0], "name": parts[1]})

    st.info("""
    **📋 Logic Waterfall:**
    1.  **Trim Overweights:** Sell positions > Max Cap.
    2.  **Secure Cash:** Refill Buffer to Min %.
    3.  **Fund New Ideas:** Buy selected "Simulated Adds" up to 3% each.
    4.  **Reinvest Remainder:** Distribute any leftover cash to existing stocks.
    """)
    
    try:
        # Pass the new candidates list to the function
        reb_df = suggest_rebalance_robust(holdings, c_buff, m_w, new_adds=new_adds_list, entry_size=0.03)
        
        # Diagnostics
        curr_cash = pd.DataFrame([h.__dict__ for h in holdings]).query("kind == 'CASH'")["weight"].sum()
        st.markdown(f"**Current Cash:** `{curr_cash:.1%}` vs Target `{c_buff:.1%}`")
        
        def highlight(row):
            if row["Action"] == "BUY": return ['background-color: #d4edda; color: black']*len(row)
            if row["Action"] == "SELL": return ['background-color: #f8d7da; color: black']*len(row)
            return ['']*len(row)
        
        display_df = reb_df.reset_index().rename(columns={"name": "Name", "ticker": "Ticker"})
        cols = ["Ticker", "Name", "Current", "Target", "Delta", "Action"]
        
        st.dataframe(
            display_df[cols].style.apply(highlight, axis=1)
            .format("{:.2%}", subset=["Current","Target","Delta"]), 
            use_container_width=True
        )
        st.caption("Note: 'Jewelry/Art' (ALT) excluded from trade suggestions.")
        
    except Exception as e:
        st.error(f"Rebalance Error: {e}")

    # ==========================================
    # 2. STRATEGIC ADVISOR (With Sector Filter)
    # ==========================================
    st.subheader("2. Strategic Additions")
    st.caption("Ideas to hedge specific risks or tilt the portfolio (Impact analysis included).")
    
    # 1. Filters Layout
    c_vis, c_scen, c_sec = st.columns([1, 1, 1.5])
    
    # Vision & Hedge
    vision = c_vis.selectbox("Vision", ["balanced", "income", "growth", "crisis-ready"])
    scen = c_scen.selectbox("Hedge Against", ["equity_crash", "stagflation", "rates_up"])
    
    # Sector Filter (Dynamic)
    avail_sectors = sorted(list({c["sector"] for c in ADVISOR_PALETTE}))
    sectors = c_sec.multiselect("Filter by Sector", avail_sectors, placeholder="All Sectors (Default)")
    
    # 2. Calculation
    # Get price data if available for risk calc, otherwise empty
    px_ref = pd.DataFrame()
    if "stress_results" in st.session_state and st.session_state["stress_results"]:
        px_ref = st.session_state["stress_results"]["price_df"]

    # Run Logic
    recs = advisor_rank_with_impact(holdings, px_ref, vision, scen, 0.03, sectors)
    
    # --- Helper for Navigation Callback ---
    def go_to_screener(ticker):
        st.session_state["screener_ticker"] = ticker
        st.session_state["active_page"] = "🔬 Stock Screener"
        st.session_state["screener_back_button"] = True

    # 3. Display Results
    if not recs.empty:
        # A. Explainer
        with st.expander("ℹ️ How to read this table", expanded=False):
            st.markdown(f"""
            * **Risk/Reward:** Compares the *added downside* in a **{scen}** scenario vs. the *added upside* in a **Bull Market**.
            * **vs SMA200:** How far the current price is above/below the 200-Day Moving Average.
            * **RSI:** <30 (Oversold/Buy), >70 (Overbought/Sell).
            * **Score:** 0-100 Rating based on Value, Momentum, and Portfolio Fit.
            """)

        # B. Table (Overview)
        display_cols = ["Score", "Ticker", "Name", "Sector", "Risk/Reward", "P/E", "RSI", "vs SMA200"]
        
        st.dataframe(
            recs[display_cols].style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100), 
            use_container_width=True,
            hide_index=True
        )
        
        # C. Top Pick Highlight (With Deep Dive Button)
        top_pick = recs.iloc[0]
        st.success(f"💡 **Top Pick: {top_pick['Name']} ({top_pick['Ticker']})**")
        
        c_desc, c_act = st.columns([4, 1])
        with c_desc:
            st.markdown(f"""
            * **Why:** {top_pick['Why']}
            * **Trade-Off:** This trade adds **{top_pick['Raw_Delta_Risk']:.2%}** to your downside risk ({scen}), but adds **{top_pick['Raw_Delta_Reward']:.2%}** to your upside potential.
            * **Technical:** Price is **{top_pick['vs SMA200']}** relative to trend.
            """)
        with c_act:
            # DEEP DIVE BUTTON (Top Pick) - FIXED WITH CALLBACK
            st.button(
                f"🔍 Analyze\n{top_pick['Ticker']}", 
                key=f"btn_top_{top_pick['Ticker']}",
                on_click=go_to_screener,    # <--- Calls the helper
                args=(top_pick['Ticker'],)  # <--- Passes the ticker
            )

        # D. Other Actionable Ideas
        if len(recs) > 1:
            st.markdown("---")
            st.subheader("📋 More Opportunities")
            for i, row in recs.iloc[1:6].iterrows(): # Show next 5
                with st.expander(f"**{row['Ticker']}** - {row['Name']} (Score: {row['Score']})"):
                    c_info, c_btn = st.columns([4, 1])
                    with c_info:
                        st.write(f"_{row['Why']}_")
                        st.caption(f"Risk/Reward: {row['Risk/Reward']} | Sector: {row['Sector']}")
                    with c_btn:
                        # DEEP DIVE BUTTON (Others) - FIXED WITH CALLBACK
                        st.button(
                            f"🔍 Analyze", 
                            key=f"btn_{row['Ticker']}",
                            on_click=go_to_screener,   # <--- Calls the helper
                            args=(row['Ticker'],)      # <--- Passes the ticker
                        )
    else:
        st.info("No matching suggestions found. Try broadening your criteria.")

# --- SCREENER ---
if page == "🔬 Stock Screener":

    def go_back():
        st.session_state["active_page"] = "⚖️ Advisor & Actions"
        if "screener_back_button" in st.session_state:
            del st.session_state["screener_back_button"]
    
    # 1. NAVIGATION HEADER (Back Button)
    if st.session_state.get("screener_back_button"):
        c_back, c_head = st.columns([1, 5])
        with c_back:
            # FIXED: Use callback to change page safely
            st.button("⬅️ Return", on_click=go_back)
        with c_head:
            st.header("🔬 Screener")
    else:
        st.header("🔬 Screener")
    
    # Helper: Yahoo Search
    def search_yahoo(q):
        try:
            h = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={q}", headers=h)
            return r.json().get("quotes", [])
        except: return []

    # 2. Search Bar with Session State Logic
    col_s, col_r = st.columns([1,2])
    
    # CHECK FOR DEEP DIVE TARGET
    default_q = "BASF"
    if "screener_ticker" in st.session_state:
        default_q = st.session_state["screener_ticker"]
        del st.session_state["screener_ticker"] # Consume the state so it doesn't stick

    q = col_s.text_input("Ticker/Name", value=default_q).strip()
    
    sel_tkr = None
    if q:
        res = search_yahoo(q)
        if res:
            opts = {f"{r['symbol']} | {r.get('shortname','')}": r['symbol'] for r in res}
            # Auto-select the first result if it matches exactly, otherwise let user pick
            idx = 0
            s = col_r.selectbox("Result", list(opts.keys()), index=idx)
            if s: sel_tkr = opts[s]
    
    st.markdown("---")

    # 3. Data Rendering
    if sel_tkr:
        try:
            t = yf.Ticker(sel_tkr)
            i = t.info
            h = t.history(period="1y")
            
            if not h.empty and ("symbol" in i or "longName" in i):
                curr = i.get('currency','USD')
                sec = i.get('sector', 'Unknown')
                
                # Header
                c1, c2 = st.columns([3, 1])
                c1.subheader(f"{i.get('longName', sel_tkr)}")
                c1.caption(f"**{sec}** | {i.get('industry', '-')}")
                
                # Price
                px = h['Close'].iloc[-1]
                prev = i.get('regularMarketPreviousClose', px)
                chg = (px - prev) / prev if prev else 0
                c2.metric("Price", f"{px:,.2f} {curr}", f"{chg:+.2%}")

                # --- METRICS WITH SECTOR COMPARISON ---
                bench = SECTOR_BENCHMARKS.get(sec, {})
                
                st.subheader("Fundamentals vs Industry")
                k1, k2, k3, k4 = st.columns(4)
                
                # Market Cap
                mcap = i.get('marketCap')
                def fmt_b(n): return f"{n/1e9:.1f}B" if n and n>1e9 else "-"
                k1.metric("Mkt Cap", fmt_b(mcap))
                
                # P/E Ratio (Inverse Color: Lower is Green)
                pe = i.get('trailingPE')
                avg_pe = bench.get("PE")
                pe_delta = f"{(pe - avg_pe):+.1f} vs Avg" if (pe and avg_pe) else None
                k2.metric("P/E Ratio", f"{pe:.1f}x" if pe else "-", pe_delta, delta_color="inverse")

                # Profit Margin (Normal Color: Higher is Green)
                marg = i.get('profitMargins')
                avg_marg = bench.get("Margin")
                marg_delta = f"{(marg - avg_marg):+.1%} vs Avg" if (marg and avg_marg) else None
                k3.metric("Profit Margin", f"{marg:.1%}" if marg else "-", marg_delta)

                # ROE (Normal Color: Higher is Green)
                roe = i.get('returnOnEquity')
                avg_roe = bench.get("ROE")
                roe_delta = f"{(roe - avg_roe):+.1%} vs Avg" if (roe and avg_roe) else None
                k4.metric("ROE", f"{roe:.1%}" if roe else "-", roe_delta)

                # --- TECHNICALS ---
                st.subheader("Technicals")
                h["SMA50"] = h["Close"].rolling(50).mean()
                h["RSI"] = calculate_rsi(h["Close"])
                
                cur_rsi = h['RSI'].iloc[-1]
                cur_sma = h['SMA50'].iloc[-1]
                dist_sma = (px - cur_sma) / cur_sma if cur_sma else 0
                
                t1, t2, t3, t4 = st.columns(4)
                
                # RSI Color Logic
                rsi_st = "Neutral"
                if cur_rsi > 70: rsi_st = "Overbought"
                elif cur_rsi < 30: rsi_st = "Oversold"
                t1.metric("RSI (14)", f"{cur_rsi:.1f}", rsi_st, delta_color="off")
                
                t2.metric("vs SMA 50", f"{dist_sma:+.1%}", "Trend Strength")
                t3.metric("Beta", f"{i.get('beta'):.2f}" if i.get('beta') else "-")
                t4.metric("Div Yield", f"{i.get('dividendYield', 0):.2%}" if i.get('dividendYield') else "-")

                # Chart
                st.line_chart(h["Close"])
                
                # Business Summary
                with st.expander("🏢 Business Summary"):
                    st.write(i.get('longBusinessSummary', 'No summary available.'))
                    
            else:
                st.error("Data unavailable or Ticker invalid.")
        except Exception as e:
            st.error(f"Error loading data: {e}")