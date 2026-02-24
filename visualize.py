import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from quark.facade import MasterQuark

sns.set_theme(style="whitegrid", palette="rocket")

def fetch_real_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    tickers_with_bench = tickers + ['SPY']
    data = yf.download(tickers_with_bench, start=start_date, end=end_date, progress=False, auto_adjust=True)
    prices = data['Close'] if 'Close' in data.columns else data
    return prices.dropna(axis=1, how='all')

def plot_allocation(weights_dict, save_path):
    plt.figure(figsize=(8, 8))
    labels = [l for l, s in zip(weights_dict.keys(), weights_dict.values()) if s > 0]
    sizes = [s for s in weights_dict.values() if s > 0]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("rocket", len(labels)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title("Quark Optimal Portfolio Allocation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_returns(prices_df, weights_dict, save_path):
    returns = prices_df.pct_change(fill_method=None).dropna()
    weights_array = np.array([weights_dict.get(t, 0.0) for t in returns.columns if t != 'SPY'])
    asset_returns = returns[[c for c in returns.columns if c != 'SPY']]
    port_cum_returns = (1 + asset_returns.dot(weights_array)).cumprod()
    
    spy_cum_returns = (1 + returns['SPY']).cumprod() if 'SPY' in returns.columns else None
    ew_cum_returns = (1 + asset_returns.dot(np.ones(len(asset_returns.columns)) / len(asset_returns.columns))).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(port_cum_returns.index, port_cum_returns, label="Quark Optimal Portfolio", linewidth=2.5, color='#d9423e')
    if spy_cum_returns is not None:
        plt.plot(spy_cum_returns.index, spy_cum_returns, label="S&P 500 (SPY)", linewidth=1.5, color='#333333', linestyle='--')
    plt.plot(ew_cum_returns.index, ew_cum_returns, label="Equal Weight Universe", linewidth=1.5, color='#8c92ac', linestyle='-.')
    
    plt.title("Historical Cumulative Returns (Out-of-Sample Mock)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Growth ($1 Invested)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    universe = [
        "NVDA", "MSFT", "META", "LLY", "AVGO", 
        "JPM", "COST", "CRM", "AMD", "NFLX",
        "HDFCBANK.NS", "NTPC.NS", "SUNPHARMA.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "COALINDIA.NS", "ONGC.NS", "TITAN.NS", "LT.NS", "TRENT.NS"
    ]
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    prices_df = fetch_real_data(universe, start_date, end_date)
    opt_prices = prices_df[[c for c in prices_df.columns if c != 'SPY']]
    
    model = MasterQuark(max_iterations=150)
    model.illuminate(opt_prices)
    
    artifact_dir = "/Users/rakeshbag/.gemini/antigravity/scratch/quark_plots"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_allocation(model.metrics_['optimal_weights'], os.path.join(artifact_dir, "quark_allocation.png"))
    plot_cumulative_returns(prices_df, model.metrics_['optimal_weights'], os.path.join(artifact_dir, "quark_cumulative_returns.png"))

if __name__ == "__main__":
    main()
