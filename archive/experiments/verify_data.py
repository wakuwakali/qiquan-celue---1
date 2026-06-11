import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
import math

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def main():
    print("Fetching active BTC options from Deribit...")
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    response = requests.get(url).json()
    
    if 'result' not in response:
        print("Failed to fetch data from Deribit.")
        return
        
    data = response['result']
    
    # We need the current BTC spot price. Deribit provides an index price.
    index_price = data[0]['estimated_delivery_price']
    print(f"Current BTC Spot Price (Index): ${index_price:,.2f}")
    
    options = []
    now = datetime.now(timezone.utc)
    
    for item in data:
        inst = item['instrument_name']
        # e.g., BTC-28JUN24-65000-C
        parts = inst.split('-')
        if len(parts) != 4: continue
        coin, expiry_str, strike_str, type_str = parts
        
        if type_str != 'C': continue # Only Calls
        
        try:
            # Deribit expiry is usually 08:00 UTC
            expiry_date = datetime.strptime(expiry_str + " 08:00:00", "%d%b%y %H:%M:%S").replace(tzinfo=timezone.utc)
        except:
            continue
            
        time_to_expiry_days = (expiry_date - now).total_seconds() / (24 * 3600)
        
        if time_to_expiry_days <= 0: continue
        
        strike = float(strike_str)
        mark_val = item.get('mark_price')
        mark_price_btc = mark_val if mark_val is not None else 0
        mark_price_usd = mark_price_btc * index_price
        
        bid_val = item.get('bid_price')
        ask_val = item.get('ask_price')
        bid_price_usd = (bid_val if bid_val is not None else 0) * index_price
        ask_price_usd = (ask_val if ask_val is not None else 0) * index_price
        
        iv_val = item.get('mark_iv')
        iv = (iv_val if iv_val is not None else 0) / 100.0
        
        # The API doesn't always give delta in this endpoint directly if liquidity is low,
        # but let's check if it exists in 'greeks' or just calculate it using their IV
        # Actually, get_book_summary doesn't have greeks directly. Wait, sometimes it does.
        # Wait, let's use the public/ticker endpoint for specific instruments if we need greeks, 
        # or we just compute delta from their IV.
        # It's easier to just compute Delta from their IV to see which one is ~0.1
        r = 0.0
        T = time_to_expiry_days / 365.0
        d1 = (np.log(index_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T)) if iv > 0 else 0
        delta = norm.cdf(d1) if iv > 0 else 0
        
        options.append({
            'Instrument': inst,
            'Strike': strike,
            'DaysToExpiry': time_to_expiry_days,
            'T_years': T,
            'MarkPrice_USD': mark_price_usd,
            'Bid_USD': bid_price_usd,
            'Ask_USD': ask_price_usd,
            'Market_IV': iv,
            'Delta': delta
        })
        
    df = pd.DataFrame(options)
    
    # Filter for options expiring in roughly 1 to 3 days to match our 24h/short-term test
    df_short_term = df[(df['DaysToExpiry'] >= 0.5) & (df['DaysToExpiry'] <= 3.0)]
    
    if len(df_short_term) == 0:
        # If no 1-3 day expiries, just pick the nearest one
        nearest_days = df['DaysToExpiry'].min()
        df_short_term = df[(df['DaysToExpiry'] >= nearest_days) & (df['DaysToExpiry'] <= nearest_days + 1.0)]
        
    print(f"\nLooking at options expiring in {df_short_term['DaysToExpiry'].iloc[0]:.2f} days...")
    
    # Find the Call option with Delta closest to 0.1
    # Filter delta < 0.15 to just get a neighborhood
    df_near_01 = df_short_term[(df_short_term['Delta'] > 0.05) & (df_short_term['Delta'] < 0.15)]
    
    if len(df_near_01) == 0:
        print("No options with delta around 0.1 found for this expiry. Showing nearest:")
        df_near_01 = df_short_term.iloc[(df_short_term['Delta'] - 0.1).abs().argsort()[:3]]
        
    best_option = df_near_01.iloc[(df_near_01['Delta'] - 0.1).abs().argmin()]
    
    print("\n--- Actual Market Data (Deribit) ---")
    print(f"Instrument: {best_option['Instrument']}")
    print(f"Strike: ${best_option['Strike']:,.2f}")
    print(f"Delta: {best_option['Delta']:.4f}")
    print(f"Market Implied Volatility (IV): {best_option['Market_IV']:.2%}")
    print(f"Market Mark Price: ${best_option['MarkPrice_USD']:,.2f}")
    print(f"Market Bid/Ask: ${best_option['Bid_USD']:,.2f} / ${best_option['Ask_USD']:,.2f}")
    
    print("\n--- Backtest Simulation Model Comparison ---")
    # In our backtest, we used a rolling 30-day historical volatility.
    # Let's fetch the recent 30 days of BTC to get that.
    import yfinance as yf
    btc = yf.download('BTC-USD', period='60d', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        close_px = btc['Close'].iloc[:, 0]
    else:
        close_px = btc['Close']
    log_ret = np.log(close_px / close_px.shift(1))
    hist_vol = log_ret.tail(30).std() * np.sqrt(365)
    
    print(f"Our Model's 30-day Historical Volatility: {hist_vol:.2%}")
    
    simulated_price = black_scholes_call(index_price, best_option['Strike'], best_option['T_years'], 0.0, hist_vol)
    print(f"Our Model's Simulated Price (using Hist Vol): ${simulated_price:,.2f}")
    
    # Let's see the error
    diff = simulated_price - best_option['MarkPrice_USD']
    diff_pct = (diff / best_option['MarkPrice_USD']) if best_option['MarkPrice_USD'] > 0 else 0
    print(f"\nDifference: ${diff:,.2f} ({diff_pct:.2%})")
    
    if abs(diff_pct) > 0.5:
        print("\nConclusion: The simulation differs significantly from the market.")
        print("Reason: Crypto options suffer from 'Volatility Smile' and 'Volatility Premium'.")
        print("Market Implied Volatility (what traders actually price it at) is often much higher than Historical Volatility.")
        print("This means actual market premiums collected would likely be HIGHER than our simulation.")
    else:
        print("\nConclusion: The simulation is fairly close to actual market prices.")

if __name__ == "__main__":
    main()
