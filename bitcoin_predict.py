# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib
# from datetime import datetime

# # Load model + scaler
# model = load_model("bitcoin_lstm_model.h5")
# scaler = joblib.load("scaler.pkl")

# # Load and clean data again (same steps!)
# df = pd.read_csv("Bitcoin.csv")

# df.replace("###", np.nan, inplace=True)

# cols = ['Price', 'Open', 'High', 'Low']
# for col in cols:
#     df[col] = df[col].str.replace(',', '')
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# df['Change %'] = df['Change %'].str.replace('%', '')
# df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')

# def convert_volume(vol):
#     if isinstance(vol, str):
#         vol = vol.strip()
        
#         if vol == '-' or vol == '':
#             return np.nan
        
#         if 'K' in vol:
#             return float(vol.replace('K','')) * 1_000
#         elif 'M' in vol:
#             return float(vol.replace('M','')) * 1_000_000
#         elif 'B' in vol:  
#             return float(vol.replace('B','')) * 1_000_000_000
    
#     return vol

# df['Vol.'] = df['Vol.'].apply(convert_volume)

# df = df.dropna()

# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date')
# df.set_index('Date', inplace=True)

# # Prepare data
# features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
# data = df[features]
# scaled_data = scaler.transform(data)

# # Function
# def predict_future(model, scaled_data, days, scaler):
#     future_predictions = []
#     last_60 = scaled_data[-60:]

#     for _ in range(days):
#         input_data = last_60.reshape(1, 60, scaled_data.shape[1])
        
#         pred = model.predict(input_data)
#         pred_value = pred[0][0]  
        
#         future_predictions.append(pred_value)
#         new_row = last_60[-1]   # copy last row
#         new_row[0] = pred_value  
#         last_60 = np.vstack((last_60, new_row))[1:]

#     temp = np.zeros((len(future_predictions), scaled_data.shape[1]))
#     temp[:, 0] = future_predictions

#     return scaler.inverse_transform(temp)[:, 0]

# user_date = input("Enter future date (YYYY-MM-DD): ")
# selected_date = datetime.strptime(user_date, "%Y-%m-%d")

# last_date = df.index[-1]
# days = (selected_date - last_date).days

# if days <= 0:
#     print("Enter a future date!")
# else:
#     future_prices = predict_future(model, scaled_data, days, scaler)
#     print(f"Predicted Price on {user_date}: {future_prices[-1]:.2f}")
    
    
    
    
#==================================================================================================
#===================================================================================================

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# ==============================
# Load model and scaler
# ==============================
model = load_model("bitcoin_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# ==============================
# Load and clean dataset
# ==============================
df = pd.read_csv("Bitcoin.csv")

df.replace("###", np.nan, inplace=True)

cols = ['Price', 'Open', 'High', 'Low']
for col in cols:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# ==============================
# Use selected features
# ==============================
features = ['Price', 'Open', 'High', 'Low']
data = df[features]

# Scale using saved scaler
scaled_data = scaler.transform(data)

# ==============================
# Prediction Function
# ==============================
def predict_future(model, scaled_data, days, scaler):
    future_predictions = []
    last_60 = scaled_data[-60:]

    for _ in range(days):
        input_data = last_60.reshape(1, 60, scaled_data.shape[1])

        pred = model.predict(input_data)
        pred_value = pred[0][0]   

        future_predictions.append(pred_value)
        new_row = last_60[-1].copy()
        new_row[0] = pred_value   

        last_60 = np.vstack((last_60, new_row))[1:]

    temp = np.zeros((len(future_predictions), scaled_data.shape[1]))
    temp[:, 0] = future_predictions

    real_prices = scaler.inverse_transform(temp)[:, 0]

    return real_prices

# ==============================
# User Input
# ==============================
user_date = input("Enter future date (YYYY-MM-DD): ")
selected_date = datetime.strptime(user_date, "%Y-%m-%d")

last_date = df.index[-1]
days = (selected_date - last_date).days

# ==============================
# Prediction Output
# ==============================
if days <= 0:
    print("Enter a future date!")
else:
    future_prices = predict_future(model, scaled_data, days, scaler)
    print(f"Predicted Bitcoin Price on {user_date}: {future_prices[-1]:.2f}")    