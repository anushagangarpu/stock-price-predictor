import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# =============================================
# BACKGROUND IMAGE
# =============================================
def add_bg_image():
    try:
        with open("img.jpeg", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        bg_style = f"url('data:image/jpeg;base64,{encoded}')"
    except:
        bg_style = "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg_style};
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# =============================================
# LSTM MODEL
# =============================================
def predict_with_lstm(data_scaled, days_to_predict, scaler, lookback=30):

    X, y = [], []

    for i in range(lookback, len(data_scaled) - days_to_predict):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i:i + days_to_predict, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(days_to_predict)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_data = data_scaled[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_data)[0]

    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return predictions[:days_to_predict]

# =============================================
# IMPORTS
# =============================================
import yfinance as yf
from yahooquery import search

# =============================================
# SESSION STATE
# =============================================
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = ""

# =============================================
# UI
# =============================================
st.title("📈 Stock Price Predictor")

# Search
company = st.text_input("Search Company")
if st.button("Search"):
    results = search(company)
    if "quotes" in results:
        options = {r['shortname']: r['symbol'] for r in results['quotes'] if 'symbol' in r and 'shortname' in r}
        selected = st.selectbox("Select", list(options.keys()))
        if selected:
            st.session_state.selected_stock = options[selected]

# Prediction
if st.session_state.selected_stock:
    days = st.slider("Days", 1, 30, 7)

    if st.button("Predict"):

        with st.spinner("Fetching data..."):
            data = yf.download(st.session_state.selected_stock, period="2y", auto_adjust=True)

        if data.empty:
            st.error(f"Could not fetch data for **{st.session_state.selected_stock}**. Try a different stock.")
        else:
            close = data[['Close']].values

            if len(close) < 31:
                st.error(f"Not enough historical data. Only {len(close)} rows found.")
            else:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(close)

                preds = predict_with_lstm(scaled, days, scaler, lookback=30)

                df = pd.DataFrame(preds, columns=["Predicted"])
                df.index = pd.date_range(start=data.index[-1], periods=len(preds)+1, freq='B')[1:]

                st.line_chart(df)

                st.metric("Last Price", f"₹{close[-1][0]:.2f}")
                st.metric("Predicted", f"₹{preds[-1][0]:.2f}")
                