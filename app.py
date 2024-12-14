import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime

# Load the pre-trained model
model = load_model(r'C:\Users\rayya\OneDrive\Desktop\Coding\Stock Predictor App\Stock Predictions Model.keras')

# Streamlit App Header
st.header('Stock Market Predictor Application')

# User Inputs for Stock Symbol and Date Range
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.date_input('Start Date', datetime.date(2012, 1, 1))
end_date = st.date_input('End Date', datetime.date(2024, 7, 15))

# Download stock data
try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error("Invalid stock symbol or no data available. Please try again.")
    else:
        st.subheader('Stock Data')
        st.write(data)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Data Splitting and Preprocessing
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

# Combine last 100 days from train and test for proper scaling
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

# Moving Averages Calculation
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

# Visualization: Price vs. Moving Averages
st.subheader('Price vs. MA50')
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], 'g', label='Close Price')
plt.plot(ma_50_days, 'r', label='MA50')
plt.legend()
plt.title("Close Price vs. 50-Day Moving Average")
st.pyplot(fig1)

st.subheader('Price vs. MA50 vs. MA100')
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], 'g', label='Close Price')
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.legend()
plt.title("Close Price vs. 50-Day and 100-Day Moving Averages")
st.pyplot(fig2)

st.subheader('Price vs. MA50 vs. MA100 vs. MA200')
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], 'g', label='Close Price')
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(ma_200_days, 'm', label='MA200')
plt.legend()
plt.title("Close Price vs. Multiple Moving Averages")
st.pyplot(fig3)

# Prepare Data for Model Prediction
x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
predictions = model.predict(x_test)

# Scale predictions back to original values
scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

# Visualization: Original vs. Predicted Prices
st.subheader('Original vs. Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(data.index[-len(predictions):], predictions, 'r', label='Predicted Price')
plt.plot(data.index[-len(y_test):], y_test, 'g', label='Original Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Original vs. Predicted Price')
plt.legend()
st.pyplot(fig4)

# Model Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
st.subheader('Model Performance Metrics')
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Option to Download Predictions
output = pd.DataFrame({
    'Date': data.index[-len(predictions):],
    'Original Price': y_test.flatten(),
    'Predicted Price': predictions.flatten()
})
st.download_button(
    label='Download Predictions as CSV',
    data=output.to_csv(index=False),
    file_name='stock_predictions.csv',
    mime='text/csv'
)
