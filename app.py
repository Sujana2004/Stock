import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the model
try:
    model = load_model('C:/Users/korup/OneDrive/Desktop/Stock/Stock Predictions Model.keras')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.header('Stock Market Predictor')

# User input for stock symbol and date range
stock = st.text_input('Enter Stock Symbol:', 'GOOG')
start = st.date_input('Start Date', pd.to_datetime('2012-01-01'))
end = st.date_input('End Date', pd.to_datetime('2023-12-31'))

# Fetch stock data
try:
    data = yf.download(stock, start=start, end=end)
    st.subheader('Stock Data')
    st.write(data)
except Exception as e:
    st.error(f"Error fetching data for {stock}: {e}")

# Prepare training and testing datasets
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']]) 
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Calculate moving averages and plot them
def plot_moving_averages(data):
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    # Price vs MA50
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close Price', line=dict(color='green')))
    fig1.update_layout(title='Price vs MA50', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig1, use_container_width=True)

    # Price vs MA50 vs MA100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close Price', line=dict(color='green')))
    fig2.update_layout(title='Price vs MA50 vs MA100', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig2, use_container_width=True)

    # Price vs MA100 vs MA200
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close Price', line=dict(color='green')))
    fig3.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig3, use_container_width=True)

plot_moving_averages(data)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
y_predict = model.predict(x)
# Inverse transform to get actual prices
y_predict_inverse = scaler.inverse_transform(y_predict.reshape(-1, 1))

# Inverse transform predictions to get actual prices
y_predict_inverse = scaler.inverse_transform(y_predict.reshape(-1, 1)).flatten()
y_inverse = scaler.inverse_transform(y.reshape(-1, 1)).flatten()  # Assuming y was scaled similarly


# Original Price vs Predicted Price (Test Data)
import plotly.graph_objects as go

# Assuming y_predict_inverse and y_inverse are already calculated
# Create a Plotly figure
fig = go.Figure()

# Add predicted prices to the figure
fig.add_trace(go.Scatter(
    x=list(range(len(y_predict_inverse))),  # Using index as x-axis
    y=y_predict_inverse,
    mode='lines',
    name='Predicted Price',
    line=dict(color='red')
))

# Add original prices to the figure
fig.add_trace(go.Scatter(
    x=list(range(len(y_inverse))),  # Using index as x-axis
    y=y_inverse,
    mode='lines',
    name='Original Price',
    line=dict(color='green')
))

# Update layout with titles and labels
fig.update_layout(
    title='Predicted vs Original Prices(test_data)',
    xaxis_title='Time',
    yaxis_title='Price',
    legend_title='Price Type'
)

# Show the figure in Streamlit or in your browser
st.plotly_chart(fig)  # If you're using Streamlit


# Original Price vs Predicted Price for Next 100 Days
fig5 = go.Figure()

last_date = data.index[-1]
predicted_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=300)

# Use actual dates for the original prices
fig5.add_trace(go.Scatter(
    x=data.index[-len(y_inverse):],  # Use original prices' dates
    y=y_inverse,  # Inverse transformed original prices
    mode='lines',
    name='Original Price',
    line=dict(color='blue')
))

# Add predicted prices for the next 300 days
fig5.add_trace(go.Scatter(
    x=predicted_dates,
    y=y_predict_inverse.flatten(),  # Predicted prices for next 300 days
    mode='lines',
    name='Predicted Price',
    line=dict(color='red')
))

fig5.update_layout(
    title='Original Price vs Predicted Price (Next 300 Days)',
    xaxis_title='Date',
    yaxis_title='Price'
)

st.plotly_chart(fig5, use_container_width=True)

# To run this code use streamlit run app.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, y_predict)
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_predict)
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_inverse - y_predict_inverse) / y_inverse)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Calculate MAPE while avoiding division by zero
# Filter out zero actual values
non_zero_indices = y_inverse != 0  # Get indices where actual values are not zero

if np.any(non_zero_indices):  # Check if there are any non-zero values
    mape = np.mean(np.abs((y_inverse[non_zero_indices] - y_predict_inverse[non_zero_indices]) / y_inverse[non_zero_indices])) * 100
else:
    mape = float('inf')  # Set MAPE to infinity if all actual values are zero

# Display MAPE in Streamlit
st.subheader('MAPE Calculation')
st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# AAPL
#TSLA
#BAJAJFINSV.NS
#AMZN
#HDB
#TCS.NS
