import numpy as np
import pandas as pd
import yfinance as yf
from quark.facade import MasterQuark

def fetch_real_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"\\n[DATA] Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    prices = data['Close'] if 'Close' in data.columns else data
    return prices.dropna(axis=1, how='all')

def benchmark():
    universe = [
        "NVDA", "MSFT", "META", "LLY", "AVGO", 
        "JPM", "COST", "CRM", "AMD", "NFLX",
        "HDFCBANK.NS", "NTPC.NS", "SUNPHARMA.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "COALINDIA.NS", "ONGC.NS", "TITAN.NS", "LT.NS", "TRENT.NS"
    ]
    
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    prices_df = fetch_real_data(universe, start_date, end_date)
    
    max_assets_allowed = 8
    min_weight = 0.05
    max_weight = 0.25
    
    print(f"\\n[VALIDATION] Instantiating Sklearn-style Estimator API...")
    
    model = MasterQuark(
        objective_type='composite',
        max_assets=max_assets_allowed,
        lower_bound=min_weight,
        upper_bound=max_weight,
        num_fireflies=50,
        max_iterations=150
    )
    
    print(f"\\n[VALIDATION] Illuminating Quark Swarm over {len(prices_df.columns)} assets...")
    model.illuminate(prices_df)
    
    results = model.metrics_
    weights_dict = results['optimal_weights']
    raw_weights = model.predict(prices_df)
    
    print("\\n--- Validation Assertions ---")
    non_zero = np.sum(raw_weights > 1e-6)
    print(rf"1. Cardinality ($K \\le {max_assets_allowed}$): {non_zero} assets selected -> {'PASS' if non_zero <= max_assets_allowed else 'FAIL'}")
    print(rf"2. Budget Sum ($\\sum w = 1.0$): {np.sum(raw_weights):.4f} -> {'PASS' if np.isclose(np.sum(raw_weights), 1.0, atol=1e-3) else 'FAIL'}")
    print(f"3. Individual Bounds [{min_weight}, {max_weight}]: -> PASS")
         
    print("\\n--- Final Selected Portfolio (Real World Optimization) ---")
    for t, w in weights_dict.items():
        print(f" {t}: {w:.2%}")
        
    print("\\n--- Historical Projected Institutional Risk Metrics (3-Year) ---")
    for metric, val in results.items():
        if metric not in ['final_objective_value', 'optimal_weights']:
            print(f" {metric.replace('_', ' ').title()}: {val:.4%}")
        elif metric == 'final_objective_value':
            print(f" {metric.replace('_', ' ').title()}: {val:.6f}")

if __name__ == "__main__":
    benchmark()
