import ccxt
import pandas as pd
import time

def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    symbol = 'BTC/USDT'
    
    print("Fetching daily klines...")
    try:
        # Fetch perpetual klines
        since = exchange.parse8601('2020-01-01T00:00:00Z')
        all_klines = []
        while True:
            klines = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
            if not klines:
                break
            all_klines.extend(klines)
            since = klines[-1][0] + 86400000
            if len(klines) < 1000:
                break
        
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Fetched {len(df)} days of klines. First: {df['Date'].iloc[0]}, Last: {df['Date'].iloc[-1]}")
        
    except Exception as e:
        print(f"Error fetching klines: {e}")

    print("\nFetching funding rates...")
    try:
        all_rates = []
        since = exchange.parse8601('2020-01-01T00:00:00Z')
        while True:
            # Binance fapi endpoint for funding rates
            rates = exchange.fapiPublicGetFundingRate({'symbol': 'BTCUSDT', 'startTime': since, 'limit': 1000})
            if not rates:
                break
            all_rates.extend(rates)
            since = int(rates[-1]['fundingTime']) + 1
            if len(rates) < 1000:
                break
                
        df_rates = pd.DataFrame(all_rates)
        df_rates['Date'] = pd.to_datetime(pd.to_numeric(df_rates['fundingTime']), unit='ms')
        df_rates['fundingRate'] = pd.to_numeric(df_rates['fundingRate'])
        print(f"Fetched {len(df_rates)} funding rate records. First: {df_rates['Date'].iloc[0]}, Last: {df_rates['Date'].iloc[-1]}")
    except Exception as e:
        print(f"Error fetching funding rates: {e}")

if __name__ == "__main__":
    fetch_data()
