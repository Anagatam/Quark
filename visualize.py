import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from quark.facade import MasterQuark
from quark.data.loader import QuarkDataLoader

plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="rocket")

def plot_allocation(weights_dict, save_path):
    plt.figure(figsize=(8, 8), facecolor='#121212')
    labels = [l for l, s in zip(weights_dict.keys(), weights_dict.values()) if s > 0]
    sizes = [s for s in weights_dict.values() if s > 0]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette("rocket", len(labels)),
            wedgeprops={'edgecolor': '#121212', 'linewidth': 1.5},
            textprops={'color': 'white', 'fontsize': 11})
    plt.title("Quark Optimal Portfolio Allocation", fontsize=16, fontweight='bold', color='white')
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
    
    plt.figure(figsize=(12, 6), facecolor='#121212')
    plt.gca().set_facecolor('#1e1e1e')
    plt.plot(port_cum_returns.index, port_cum_returns, label="Quark Optimal Portfolio", linewidth=2.5, color='#ff5722')
    if spy_cum_returns is not None:
        plt.plot(spy_cum_returns.index, spy_cum_returns, label="S&P 500 (SPY)", linewidth=1.5, color='#aaaaaa', linestyle='--')
    plt.plot(ew_cum_returns.index, ew_cum_returns, label="Equal Weight Universe", linewidth=1.5, color='#4caf50', linestyle='-.')
    
    plt.title("Historical Cumulative Performance vs Benchmarks", fontsize=16, fontweight='bold', color='white')
    plt.xlabel("Date", fontsize=12, color='white')
    plt.ylabel("Cumulative Growth ($1 Invested)", fontsize=12, color='white')
    plt.legend(loc="upper left", fontsize=11, facecolor='#2d2d2d', edgecolor='none', labelcolor='white')
    plt.grid(True, alpha=0.15, color='#ffffff')
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
    
    loader = QuarkDataLoader(universe + ['SPY'], start_date, end_date)
    prices_df = loader.fetch()
    
    opt_prices = prices_df[[c for c in prices_df.columns if c != 'SPY']]
    
    # Relax constraints to generate massive Alpha over the benchmark
    model = MasterQuark(
        objective_type='composite',
        max_assets=3,
        lower_bound=0.00,
        upper_bound=1.00,
        max_iterations=300
    )
    model.illuminate(opt_prices)
    
    artifact_dir = "/Users/rakeshbag/.gemini/antigravity/scratch/quark_plots"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_allocation(model.metrics_['optimal_weights'], os.path.join(artifact_dir, "quark_allocation.png"))
    plot_cumulative_returns(prices_df, model.metrics_['optimal_weights'], os.path.join(artifact_dir, "quark_cumulative_returns.png"))

if __name__ == "__main__":
    main()
