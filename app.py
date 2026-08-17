# import streamlit as st
# import pandas as pd
# import numpy as np
# import base64
# from datetime import datetime
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # =============================================
# # PAGE CONFIG
# # =============================================
# st.set_page_config(
#     page_title="Stock Price Predictor",
#     page_icon="📈",
#     layout="centered"
# )

# # =============================================
# # BACKGROUND IMAGE
# # =============================================
# def add_bg_image():
#     try:
#         with open("img.jpeg", "rb") as image_file:
#             encoded = base64.b64encode(image_file.read()).decode()
#         bg_style = f"url('data:image/jpeg;base64,{encoded}')"
#     except:
#         bg_style = "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)"

#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: {bg_style};
#             background-size: cover;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# add_bg_image()

# # =============================================
# # LSTM MODEL
# # =============================================
# def predict_with_lstm(data_scaled, days_to_predict, scaler, lookback=30):

#     X, y = [], []

#     for i in range(lookback, len(data_scaled) - days_to_predict):
#         X.append(data_scaled[i - lookback:i, 0])
#         y.append(data_scaled[i:i + days_to_predict, 0])

#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
#         LSTM(50),
#         Dense(days_to_predict)
#     ])

#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X, y, epochs=5, batch_size=32, verbose=0)

#     last_data = data_scaled[-lookback:].reshape(1, lookback, 1)
#     pred_scaled = model.predict(last_data)[0]

#     predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
#     return predictions[:days_to_predict]

# # =============================================
# # IMPORTS
# # =============================================
# import yfinance as yf
# from yahooquery import search

# # =============================================
# # SESSION STATE
# # =============================================
# if "selected_stock" not in st.session_state:
#     st.session_state.selected_stock = ""

# # =============================================
# # UI
# # =============================================
# st.title("📈 Stock Price Predictor")

# # Search
# company = st.text_input("Search Company")
# if st.button("Search"):
#     results = search(company)
#     if "quotes" in results:
#         options = {r['shortname']: r['symbol'] for r in results['quotes'] if 'symbol' in r and 'shortname' in r}
#         selected = st.selectbox("Select", list(options.keys()))
#         if selected:
#             st.session_state.selected_stock = options[selected]

# # Prediction
# if st.session_state.selected_stock:
#     days = st.slider("Days", 1, 30, 7)

#     if st.button("Predict"):

#         with st.spinner("Fetching data..."):
#             data = yf.download(st.session_state.selected_stock, period="2y", auto_adjust=True)

#         if data.empty:
#             st.error(f"Could not fetch data for **{st.session_state.selected_stock}**. Try a different stock.")
#         else:
#             close = data[['Close']].values

#             if len(close) < 31:
#                 st.error(f"Not enough historical data. Only {len(close)} rows found.")
#             else:
#                 scaler = MinMaxScaler()
#                 scaled = scaler.fit_transform(close)

#                 preds = predict_with_lstm(scaled, days, scaler, lookback=30)

#                 df = pd.DataFrame(preds, columns=["Predicted"])
#                 df.index = pd.date_range(start=data.index[-1], periods=len(preds)+1, freq='B')[1:]

#                 st.line_chart(df)

#                 st.metric("Last Price", f"₹{close[-1][0]:.2f}")
#                 st.metric("Predicted", f"₹{preds[-1][0]:.2f}")


import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf


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
            background-position: center;
            background-attachment: fixed;
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

    X = []
    y = []

    for i in range(lookback, len(data_scaled) - days_to_predict + 1):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i:i + days_to_predict, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Not enough data to train the model.")

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(days_to_predict)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=0
    )

    last_data = data_scaled[-lookback:].reshape(1, lookback, 1)

    pred_scaled = model.predict(last_data, verbose=0)[0]

    predictions = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    )

    return predictions[:days_to_predict]


# =============================================
# COMPANY NAME → NSE TICKER
# =============================================

company_tickers = {

    "wipro": "WIPRO.NS",
    "infosys": "INFY.NS",
    "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS",

    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",

    "hdfc bank": "HDFCBANK.NS",
    "hdfcbank": "HDFCBANK.NS",

    "icici bank": "ICICIBANK.NS",
    "icicibank": "ICICIBANK.NS",

    "sbi": "SBIN.NS",
    "state bank of india": "SBIN.NS",

    "axis bank": "AXISBANK.NS",
    "axisbank": "AXISBANK.NS",

    "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS",

    "tata steel": "TATASTEEL.NS",
    "tatasteel": "TATASTEEL.NS",

    "adani enterprises": "ADANIENT.NS",
    "adani enterprises limited": "ADANIENT.NS",

    "adani ports": "ADANIPORTS.NS",
    "adaniports": "ADANIPORTS.NS",

    "itc": "ITC.NS",

    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",

    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",

    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",

    "sun pharma": "SUNPHARMA.NS",
    "sunpharma": "SUNPHARMA.NS",

    "larsen toubro": "LT.NS",
    "l&t": "LT.NS",

    "titan": "TITAN.NS",

    "bajaj finance": "BAJFINANCE.NS",
    "bajajfinance": "BAJFINANCE.NS",

    "tech mahindra": "TECHM.NS",
    "techmahindra": "TECHM.NS"
}


# =============================================
# CONVERT USER INPUT TO TICKER
# =============================================

def get_ticker(company):

    company = company.strip().lower()

    # Check company dictionary
    if company in company_tickers:
        return company_tickers[company]

    # If user already entered NSE ticker
    if company.upper().endswith(".NS"):
        return company.upper()

    # If user entered BSE ticker
    if company.upper().endswith(".BO"):
        return company.upper()

    # Otherwise assume it is an NSE ticker
    return company.upper() + ".NS"


# =============================================
# SESSION STATE
# =============================================

if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = ""


# =============================================
# UI
# =============================================

st.title("📈 Stock Price Predictor")

st.write(
    "Enter a company name or NSE ticker to predict its future stock price."
)


# =============================================
# SEARCH
# =============================================

company = st.text_input(
    "Search Company",
    placeholder="Example: Wipro, Infosys, TCS"
)


if st.button("Search"):

    if not company.strip():
        st.warning("Please enter a company name.")

    else:
        ticker = get_ticker(company)

        st.session_state.selected_stock = ticker

        st.success(
            f"Selected stock: {ticker}"
        )


# =============================================
# PREDICTION
# =============================================

if st.session_state.selected_stock:

    st.write(
        f"### Selected Stock: "
        f"`{st.session_state.selected_stock}`"
    )

    days = st.slider(
        "Days to Predict",
        min_value=1,
        max_value=30,
        value=7
    )


    if st.button("Predict"):

        with st.spinner("Fetching stock data..."):

            try:

                data = yf.download(
    st.session_state.selected_stock,
    period="2y",
    auto_adjust=True,
    progress=False,
    threads=False
)

            except Exception as e:

                st.error(
                    f"Error fetching stock data: {e}"
                )

                st.stop()


        # =============================================
        # CHECK DATA
        # =============================================

        if data.empty:
            st.error(
                f"Could not fetch historical data for "
                f"{st.session_state.selected_stock}."
            )

            st.info(
                "Yahoo Finance did not return stock data. "
                "Please try again after a short wait."
            )

            st.stop()
        # =============================================
        # GET CLOSE PRICE
        # =============================================

        try:

            close = data["Close"]

            # Handle MultiIndex returned by some
            # versions of yfinance

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna().values.reshape(-1, 1)

        except Exception as e:

            st.error(
                f"Error processing stock data: {e}"
            )

            st.stop()


        # =============================================
        # CHECK HISTORICAL DATA
        # =============================================

        if len(close) < 60:

            st.error(
                f"Not enough historical data. "
                f"Only {len(close)} rows found."
            )

            st.stop()


        # =============================================
        # SCALE DATA
        # =============================================

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(close)


        # =============================================
        # LSTM PREDICTION
        # =============================================

        with st.spinner(
            "Training LSTM model and predicting..."
        ):

            try:

                predictions = predict_with_lstm(
                    scaled,
                    days,
                    scaler,
                    lookback=30
                )

            except Exception as e:

                st.error(
                    f"Model prediction failed: {e}"
                )

                st.stop()


        # =============================================
        # CREATE PREDICTION DATAFRAME
        # =============================================

        last_date = pd.to_datetime(
            data.index[-1]
        )

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(predictions),
            freq="B"
        )

        prediction_df = pd.DataFrame(
            {
                "Predicted Price": predictions.flatten()
            },
            index=future_dates
        )


        # =============================================
        # DISPLAY CHART
        # =============================================

        st.subheader("📊 Predicted Stock Price")

        st.line_chart(
            prediction_df
        )


        # =============================================
        # DISPLAY RESULTS
        # =============================================

        last_price = float(close[-1][0])

        predicted_price = float(
            predictions[-1][0]
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Last Price",
                f"₹{last_price:.2f}"
            )

        with col2:

            st.metric(
                f"Predicted ({days} days)",
                f"₹{predicted_price:.2f}"
            )


        # =============================================
        # PREDICTION TABLE
        # =============================================

        st.subheader("📅 Future Predictions")

        display_df = prediction_df.copy()

        display_df["Predicted Price"] = (
            display_df["Predicted Price"]
            .round(2)
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )
