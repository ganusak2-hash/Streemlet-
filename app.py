     import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.title("📈 Система прогнозування попиту")

# === 1. Автоматичне завантаження CSV з GitHub ===
url = "https://raw.githubusercontent.com/ТвійЮзер/demand_forecast/main/sales_example.csv"  # 🔹 Заміни своїм URL
data = pd.read_csv(url)

st.write("### Дані з GitHub:")
st.dataframe(data.head())

# === 2. Вибір товару ===
products = data['product'].unique()
selected_product = st.selectbox("Оберіть товар", products)
product_data = data[data['product'] == selected_product].copy()

# === 3. Вибір моделі ===
model_type = st.radio("Оберіть модель прогнозування:", ["Linear Regression", "ARIMA"])

# === 4. Прогноз ===
n_days = st.slider("Кількість днів для прогнозу:", 3, 14, 7)
train = product_data.copy()
train['date'] = pd.to_datetime(train['date'])
train['t'] = np.arange(len(train))

if model_type == "Linear Regression":
    X = train[['t']]
    y = train['sales']
    model = LinearRegression().fit(X, y)
    future_t = np.arange(len(train), len(train) + n_days)
    forecast = model.predict(future_t.reshape(-1, 1))

elif model_type == "ARIMA":
    model = ARIMA(train['sales'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_days)

# === 5. Графік ===
future_dates = pd.date_range(train['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
plt.figure(figsize=(10, 5))
plt.plot(train['date'], train['sales'], label='Реальні дані', marker='o')
plt.plot(future_dates, forecast, label='Прогноз', marker='x')
plt.xlabel("Дата")
plt.ylabel("Продажі")
plt.legend()
st.pyplot(plt)

# === 6. Метрики точності ===
if len(train) > 5:
    train_part = train[:-3]
    test_part = train[-3:]
    if model_type == "Linear Regression":
        X_train, X_test = train_part[['t']], test_part[['t']]
        y_train, y_test = train_part['sales'], test_part['sales']
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model = ARIMA(train_part['sales'], order=(1, 1, 1)).fit()
        preds = model.forecast(steps=3)

    mse = mean_squared_error(test_part['sales'], preds)
    mae = mean_absolute_error(test_part['sales'], preds)

    st.write("### Метрики точності на історичних даних:")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
   

