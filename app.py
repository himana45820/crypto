from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURATION
# ==========================================
COINS = {
    'bitcoin': {
        'csv': 'Bitcoin.csv',
        'model': 'bitcoin_lstm_model.h5',
        'scaler': 'scaler.pkl',
        'features': ['Price', 'Open', 'High', 'Low']
    },
    'ethereum': {
        'csv': 'EthereumData.csv',
        'model': 'ethereum_lstm_model.keras',
        'scaler': 'ethereum_scaler.pkl',
        'features': ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    },
    'xrp': {
        'csv': 'XRPData.csv',
        'model': 'xrp_lstm_model.keras',
        'scaler': 'xrp_scaler.pkl',
        'features': ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    }
}

# Cache models and scalers globally so we do not load them on each request
loaded_models = {}
loaded_scalers = {}

def load_ml_assets(coin_key):
    if coin_key not in loaded_models:
        loaded_models[coin_key] = load_model(COINS[coin_key]['model'])
    if coin_key not in loaded_scalers:
        loaded_scalers[coin_key] = joblib.load(COINS[coin_key]['scaler'])
    return loaded_models[coin_key], loaded_scalers[coin_key]

def convert_volume(vol):
    if isinstance(vol, str):
        vol = vol.strip()
        if vol == '-' or vol == '':
            return np.nan
        if 'K' in vol:
            return float(vol.replace('K','')) * 1_000
        elif 'M' in vol:
            return float(vol.replace('M','')) * 1_000_000
        elif 'B' in vol:
            return float(vol.replace('B','')) * 1_000_000_000
    return vol

def update_csv_for_coin(coin_key, csv_file):
    try:
        df_raw = pd.read_csv(csv_file)
        if df_raw.empty:
            return df_raw
            
        last_date_str = df_raw['Date'].iloc[0]
        last_date = datetime.strptime(last_date_str, "%m/%d/%Y")
        today = datetime.now()
        
        if (today - last_date).days > 0:
            symbols = {'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT', 'xrp': 'XRPUSDT'}
            symbol = symbols.get(coin_key)
            if symbol:
                import urllib.request
                import json
                
                limit = min((today - last_date).days + 1, 1000)
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req)
                kline_data = json.loads(response.read().decode('utf-8'))
                
                new_rows = []
                for kline in kline_data:
                    timestamp = kline[0] / 1000.0
                    k_date = datetime.fromtimestamp(timestamp)
                    if k_date.date() <= last_date.date():
                        continue
                        
                    open_p, high_p, low_p, close_p, vol_p = float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])
                    
                    if vol_p >= 1_000_000_000:
                        vol_str = f"{vol_p/1_000_000_000:.2f}B"
                    elif vol_p >= 1_000_000:
                        vol_str = f"{vol_p/1_000_000:.2f}M"
                    elif vol_p >= 1_000:
                        vol_str = f"{vol_p/1_000:.2f}K"
                    else:
                        vol_str = f"{vol_p:.2f}"
                        
                    change_pct = (close_p - open_p) / open_p * 100
                    
                    new_rows.append({
                        "Date": k_date.strftime("%m/%d/%Y"),
                        "Price": str(close_p),
                        "Open": str(open_p),
                        "High": str(high_p),
                        "Low": str(low_p),
                        "Vol.": vol_str,
                        "Change %": f"{change_pct:.2f}%"
                    })
                
                if new_rows:
                    new_rows.reverse()
                    df_new = pd.DataFrame(new_rows)
                    df_combined = pd.concat([df_new, df_raw], ignore_index=True)
                    df_combined.to_csv(csv_file, index=False)
                    print(f"Updated {csv_file} with {len(new_rows)} new rows from Binance API.")
                    return df_combined
        return df_raw
    except Exception as e:
        print(f"Error updating CSV for {coin_key}: {e}")
        return pd.read_csv(csv_file)


def load_and_clean_data(coin_key):
    csv_file = COINS[coin_key]['csv']
    df = update_csv_for_coin(coin_key, csv_file)
    df.replace("###", np.nan, inplace=True)
    
    # Process numeric columns
    cols = ['Price', 'Open', 'High', 'Low']
    for col in cols:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
        
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].apply(convert_volume)

    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Calculate Volatility
    df['Returns'] = df['Price'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=30).std()
    
    # Drop rows without Volatility (first 30 rows)
    df = df.dropna()
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/<coin>', methods=['GET'])
def get_data(coin):
    coin = coin.lower()
    if coin not in COINS:
        return jsonify({"error": "Invalid coin"}), 400
        
    try:
        df = load_and_clean_data(coin)
        
        # Prepare data for charting
        # Returning last 365 days for lighter payload, or full depending on preference. Let's return full for now.
        dates = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['Price'].tolist()
        volatility = df['Volatility'].tolist()
        
        return jsonify({
            "coin": coin,
            "dates": dates,
            "prices": prices,
            "volatility": volatility
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    coin = data.get('coin', '').lower()
    target_date_str = data.get('date', '')
    
    if coin not in COINS:
        return jsonify({"error": "Invalid coin"}), 400
        
    try:
        selected_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
    try:
        # Load Data
        df = load_and_clean_data(coin)
        last_date = df.index[-1]
        
        days = (selected_date - last_date).days
        if days < 0:
            return jsonify({"error": "Please select a date on or after {}.".format(last_date.strftime("%Y-%m-%d"))}), 400
            
        # Prepare model input
        features = COINS[coin]['features']
        df_features = df[features]
        
        model, scaler = load_ml_assets(coin)
        
        # We need to reshape slightly differently for Bitcoin vs others because of feature length
        # Either way, we scale the complete data to get the last 60 days of scaled features
        scaled_data = scaler.transform(df_features)
        
        # PREDICTION LOGIC
        future_predictions = []
        if days == 0:
            last_60 = scaled_data[-61:-1]
            loop_days = 1
        else:
            last_60 = scaled_data[-60:]
            loop_days = days
        
        for _ in range(loop_days):
            input_data = last_60.reshape(1, 60, scaled_data.shape[1])
            pred = model(input_data, training=False)
            pred_value = float(pred[0][0])
            
            future_predictions.append(pred_value)
            
            # create new row with same number of features
            new_row = last_60[-1].copy()
            new_row[0] = pred_value
            last_60 = np.vstack((last_60, new_row))[1:]
            
        temp = np.zeros((len(future_predictions), scaled_data.shape[1]))
        temp[:, 0] = future_predictions
        real_prices = scaler.inverse_transform(temp)[:, 0]
        
        return jsonify({
            "coin": coin,
            "target_date": target_date_str,
            "predicted_price": round(float(real_prices[-1]), 2),
            "last_date": last_date.strftime('%Y-%m-%d'),
            "last_price": round(float(df['Price'].iloc[-1]), 2)
        })
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)
