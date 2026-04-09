import yfinance as yf
btc = yf.download('BTC-USD', period='5d')
print(btc)
