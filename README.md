# Micro_Climate_LSTM_Prediction

## Project Overview

This project aims to develop a weather forecasting system using Long Short-Term Memory (LSTM) networks to predict weather conditions such as temperature, humidity, and wind speed. The application fetches historical weather data from the OpenWeather API and uses it to train the LSTM model. The results are displayed in an interactive Streamlit application.

### Features

- Fetches historical weather data from the OpenWeather API.
- Preprocesses the data for use with an LSTM model.
- Trains an LSTM model to predict weather conditions.
- Displays raw weather data, predicted data, and future predictions.
- Interactive and user-friendly interface using Streamlit.

## Requirements

### Software Requirements

- Python 3.7+
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas
- Numpy
- Requests
- Python-dotenv

### Hardware Requirements

- Standard computer with internet access for fetching weather data.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Sachinn-p/Micro_Climate_LSTM_Prediction.git
    cd Micro_Climate_LSTM_Prediction
    ```

2. **Create and activate a virtual environment:**

    On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    On Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project root directory and add your OpenWeather API key:
    ```plaintext
    OPENWEATHER_API_KEY=your_openweather_api_key
    ```

## Usage

1. **Run the Streamlit application:**

    ```bash
    streamlit run main.py
    ```

2. **Interact with the application:**
    - The application will display raw weather data fetched from the OpenWeather API.
    - It will preprocess the data and train the LSTM model.
    - The application will show the predicted weather data and the next 24-hour predictions.

## Project Structure

```plaintext
weather-forecasting/
├── venv
├── main.py
├── requirements.txt
├── .env
└── README.md
```
## Screenshot:

![image](https://github.com/user-attachments/assets/b87d5210-6a5d-4549-8d89-f9bacc25d356)


