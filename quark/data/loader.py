import pandas as pd
import yfinance as yf
from typing import List

class QuarkDataLoader:
    """
    Institutional Data Loader for fetching, aligning, and structurally 
    validating cross-market equity universes (e.g. NYSE & NSE).
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        print(f"\\n[DATA] 📡 Structuring Institutional Feeds for {len(self.tickers)} Equities...")
        # auto_adjust=True extracts clean Close prices dynamically
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        prices = data['Close'] if 'Close' in data.columns else data
        
        # Cross-market calendar synchronization (forward-fill non-trading days globally)
        prices_aligned = prices.ffill().dropna(axis=1, how='all')
        print(f"[DATA] ✅ Successfully synced {prices_aligned.shape[0]} trading days across {prices_aligned.shape[1]} active assets.")
        return prices_aligned
