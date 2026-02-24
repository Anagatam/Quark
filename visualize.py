import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from quark.facade import MasterQuark
from quark.data.loader import QuarkDataLoader

# Ensure local libs (kaleido) are available
sys.path.append("/Users/rakeshbag/.gemini/antigravity/scratch/.local_lib")

# Base Layout Template for Institutional Dark Theme
PREMIUM_DARK = dict(
    layout=go.Layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif", color='#e6edf3', size=13),
        title=dict(font=dict(size=22, color='#ffffff', weight='bold'), x=0.05),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(l=60, r=40, t=80, b=50),
        legend=dict(bgcolor='rgba(13, 17, 23, 0.8)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
    )
)

def calculate_metrics(returns):
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    ann_ret = (1 + returns.mean())**(252) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    cum_ret = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = drawdowns.min()
    
    return ann_ret, ann_vol, sharpe, max_dd, drawdowns, cum_ret

def plot_allocation_donut(weights_dict, save_path):
    sorted_weights = {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1]) if v > 0}
    labels = list(sorted_weights.keys())
    values = list(sorted_weights.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5,
        marker=dict(colors=px.colors.sequential.Plasma, line=dict(color='#0d1117', width=3)),
        textinfo='label+percent',
        hoverinfo='label+percent',
        textfont=dict(size=14, weight='bold')
    )])
    
    fig.update_layout(
        title="Quark Target Weights Allocation",
        template=PREMIUM_DARK,
        annotations=[dict(text='Quark', x=0.5, y=0.5, font_size=20, showarrow=False, font_weight='bold')]
    )
    fig.write_image(save_path, scale=3)


def plot_performance_tearsheet(prices_df, weights_dict, save_path):
    returns = prices_df.pct_change(fill_method=None).dropna()
    weights_array = np.array([weights_dict.get(t, 0.0) for t in returns.columns if t != 'SPY'])
    asset_returns = returns[[c for c in returns.columns if c != 'SPY']]
    
    port_returns = asset_returns.dot(weights_array)
    spy_returns = returns['SPY'] if 'SPY' in returns.columns else None
    ew_returns = asset_returns.dot(np.ones(len(asset_returns.columns)) / len(asset_returns.columns))
    
    pr_ret, pr_vol, pr_sr, pr_mdd, pr_drawdowns, port_cum = calculate_metrics(port_returns)
    ew_ret, ew_vol, ew_sr, ew_mdd, ew_drawdowns, ew_cum = calculate_metrics(ew_returns)
    if spy_returns is not None:
        sp_ret, sp_vol, sp_sr, sp_mdd, sp_drawdowns, spy_cum = calculate_metrics(spy_returns)
    else:
        spy_cum = None

    fig = make_subplots(
        rows=2, cols=1, 
        vertical_spacing=0.05, 
        shared_xaxes=True,
        row_heights=[0.7, 0.3]
    )
    
    # Cumulative Returns (Top)
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum, mode='lines', 
        name='Quark Optimal',
        line=dict(color='#00d1ff', width=3),  # Refined bright blue
        fill='tozeroy', fillcolor='rgba(0, 209, 255, 0.05)'
    ), row=1, col=1)
    
    if spy_cum is not None:
        fig.add_trace(go.Scatter(
            x=spy_cum.index, y=spy_cum, mode='lines', 
            name='S&P 500 Bench',
            line=dict(color='#8b949e', width=2, dash='dash')
        ), row=1, col=1)
        
    fig.add_trace(go.Scatter(
        x=ew_cum.index, y=ew_cum, mode='lines', 
        name='Equal Weight',
        line=dict(color='#b392f0', width=2, dash='dot')  # Soft purple
    ), row=1, col=1)

    # Drawdowns (Bottom)
    fig.add_trace(go.Scatter(
        x=pr_drawdowns.index, y=pr_drawdowns * 100, mode='lines',
        name='Quark Drawdown',
        line=dict(color='#f0883e', width=2),  # Muted orange
        fill='tozeroy', fillcolor='rgba(240, 136, 62, 0.2)'
    ), row=2, col=1)

    fig.update_layout(
        title="Institutional Performance Profile: Quark vs Benchmarks",
        template=PREMIUM_DARK,
        height=800,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Cumulative Growth ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.write_image(save_path, scale=3)


def plot_correlation_heatmap(prices_df, save_path):
    returns = prices_df.pct_change(fill_method=None).dropna()
    corr = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    
    fig.update_layout(
        title="Cross-Asset Correlation Matrix",
        template=PREMIUM_DARK,
        height=800,
        width=800
    )
    fig.write_image(save_path, scale=3)

def plot_risk_return_scatter(prices_df, weights_dict, save_path):
    returns = prices_df.pct_change(fill_method=None).dropna()
    assets = [c for c in returns.columns if c != 'SPY']
    
    data = []
    for asset in assets:
        ret, vol, _, _, _, _ = calculate_metrics(returns[asset])
        data.append({'Asset': asset, 'Annual Return (%)': ret*100, 'Volatility (%)': vol*100, 'Type': 'Asset'})
        
    weights_array = np.array([weights_dict.get(t, 0.0) for t in assets])
    port_returns = returns[assets].dot(weights_array)
    p_ret, p_vol, _, _, _, _ = calculate_metrics(port_returns)
    
    data.append({'Asset': 'Quark Optimal', 'Annual Return (%)': p_ret*100, 'Volatility (%)': p_vol*100, 'Type': 'Portfolio'})
    
    df = pd.DataFrame(data)
    
    fig = px.scatter(
        df, x='Volatility (%)', y='Annual Return (%)', text='Asset', 
        color='Type', color_discrete_map={'Asset': '#b392f0', 'Portfolio': '#00d1ff'},
        size=[15 if t == 'Portfolio' else 8 for t in df['Type']],
        title="Asset Efficiency & Risk-Return Topography"
    )
    
    fig.update_traces(textposition='top center', textfont=dict(color='white'))
    fig.update_layout(template=PREMIUM_DARK, height=700)
    fig.write_image(save_path, scale=3)



def main():
    universe = [
        "NVDA", "MSFT", "META", "LLY", "AVGO", 
        "JPM", "COST", "CRM", "AMD", "NFLX",
        "HDFCBANK.NS", "NTPC.NS", "SUNPHARMA.NS", "M&M.NS", "BAJAJ-AUTO.NS"
    ]
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
    
    loader = QuarkDataLoader(universe + ['SPY'], start_date, end_date)
    prices_df = loader.fetch()
    
    opt_prices = prices_df[[c for c in prices_df.columns if c != 'SPY']]
    
    model = MasterQuark(
        objective_type='composite',
        max_assets=4,
        lower_bound=0.00,
        upper_bound=1.00,
        max_iterations=300
    )
    model.illuminate(opt_prices)
    
    weights = model.metrics_['optimal_weights']
    
    artifact_dir = "assets"
    os.makedirs(artifact_dir, exist_ok=True)
    
    print("[VISUALS] Rendering Premium Plotly Graphics...")
    plot_allocation_donut(weights, os.path.join(artifact_dir, "quark_allocation_dark.png"))
    plot_performance_tearsheet(prices_df, weights, os.path.join(artifact_dir, "quark_cumulative_returns_dark.png"))
    plot_correlation_heatmap(opt_prices, os.path.join(artifact_dir, "quark_correlation_dark.png"))
    plot_risk_return_scatter(prices_df, weights, os.path.join(artifact_dir, "quark_efficiency_dark.png"))
    print("[VISUALS] ✅ Renders Complete.")

if __name__ == "__main__":
    main()
