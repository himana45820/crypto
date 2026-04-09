import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("XRPData.csv")

df.replace("###", np.nan, inplace=True)

cols = ['Price', 'Open', 'High', 'Low']
for col in cols:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')

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

df['Vol.'] = df['Vol.'].apply(convert_volume)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df['Returns'] = df['Price'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=30).std()
df = df.dropna()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Volatility'])

plt.title("XRP Volatility (30-Day Rolling)")
plt.xlabel("Year")
plt.ylabel("Volatility")

plt.grid(True)   

plt.tight_layout()
plt.show()



features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
data = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i][0])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save
model.save("xrp_lstm_model.keras")
joblib.dump(scaler, "xrp_scaler.pkl")

print("XRP Model saved!")