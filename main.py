import os
import requests
import pandas as pd
import numpy as np
import pickle as plk
import streamlit as st
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def fetch_weather_data():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        data = response.json()
        weather_data = []
        for item in data['list']:
            weather_data.append({
                'timestamp': item['dt'],
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'wind_speed': item['wind']['speed']
            })
        return pd.DataFrame(weather_data)
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, epochs=50, batch_size=32):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Streamlit app
st.title("Weather Forecasting with LSTM")
CITY = st.text_input("Enter the name of the city :")
BASE_URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
if CITY != "":
    # Fetch weather data
    df = fetch_weather_data()

    if not df.empty:
        st.subheader("Raw Weather Data")
        st.write(df)

        scaled_data, scaler = preprocess_data(df)

        seq_length = 24  # Assuming hourly data, this represents 1 day
        X, y = create_sequences(scaled_data, seq_length)

        # Split data into train and test sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train the model
        model = train_model(X_train, y_train)

        with open("trained_LSTM_model.pkl", "wb") as model_file:
            plk.dump(model, model_file)

        # Make predictions
        predictions = make_predictions(model, X_test, scaler)

        st.subheader("Predicted Weather Data")
        predictions_df = pd.DataFrame(predictions, columns=['Temperature', 'Humidity', 'Wind Speed'])
        st.write(predictions_df)

        # Predict the next 24 hours
        last_sequence = scaled_data[-seq_length:]
        next_24_hours = []

        for _ in range(24):
            next_prediction = model.predict(last_sequence.reshape(1, seq_length, 3))
            next_24_hours.append(next_prediction[0])
            last_sequence = np.vstack((last_sequence[1:], next_prediction))

        next_24_hours = np.array(next_24_hours)
        next_24_hours = scaler.inverse_transform(next_24_hours)

        st.subheader("Predictions for the Next 24 Hours")
        next_24_hours_df = pd.DataFrame(next_24_hours, columns=['Temperature', 'Humidity', 'Wind Speed'])
        next_24_hours_df.index = [f"Hour {i+1}" for i in range(24)]
        st.write(next_24_hours_df)
    else:
        st.error("No weather data fetched.")
