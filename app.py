import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import requests
from io import StringIO

st.set_page_config(page_title="Прогнозування попиту", layout="wide")
st.title("Система прогнозування попиту")

# ===== 1. Зчитування CSV з GitHub (raw URL) =====
RAW_URL = "https://raw.githubusercontent.com/ganusak2-hash/demand_forecast/main/sales_example.csv"

try:
    content = requests.get(RAW_URL).content
    data = pd.read_csv(StringIO(content.decode('utf-8')))
    st.success("CSV успішно завантажено з GitHub!")
except Exception:
    st.error("Не вдалося завантажити CSV з GitHub.")
    st.stop()

# ===== 2. Вибір колонок =====
date_col = "date"
product_col = "product"
sales_col = "sales"

# ===== 3. Вибір товару =====
products = data[product_col].unique()
selected_product = st.selectbox("Оберіть товар:", products)
product_data = data[data[product_col] == selected_product].copy()

# ===== 4. Підготовка даних =====
product_data[date_col] = pd.to_datetime(product_data[date_col])
product_data = product_data.sort_values(by=date_col)
product_data['t'] = np.arange(len(product_data))

# ===== 5. Вибір моделі =====
model_type = st.radio("Оберіть модель прогнозування:", ["Linear Regression", "ARIMA"])
n_days = st.slider("Кількість днів для прогнозу:", 3, 14, 7)

forecast = []

if model_type == "Linear Regression":
    X = product_data[['t']]
    y = product_data[sales_col]
    model = LinearRegression().fit(X, y)
    future_t = np.arange(len(product_data), len(product_data) + n_days).reshape(-1, 1)
    forecast = model.predict(future_t)

elif model_type == "ARIMA":
    model = ARIMA(product_data[sales_col], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_days)

# ===== 6. Графік прогнозу =====
future_dates = pd.date_range(product_data[date_col].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
plt.figure(figsize=(10, 5))
plt.plot(product_data[date_col], product_data[sales_col], label="Реальні дані", marker='o')
plt.plot(future_dates, forecast, label="Прогноз", marker='x')
plt.xlabel("Дата")
plt.ylabel("Продажі")
plt.title(f"Прогноз продажів: {selected_product}")
plt.legend()
st.pyplot(plt)

# ===== 7. Метрики точності =====
if len(product_data) > 5:
    train_part = product_data[:-3]
    test_part = product_data[-3:]
    if model_type == "Linear Regression":
        X_train, X_test = train_part[['t']], test_part[['t']]
        y_train, y_test = train_part[sales_col], test_part[sales_col]
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model = ARIMA(train_part[sales_col], order=(1, 1, 1)).fit()
        preds = model.forecast(steps=3)

    mse = mean_squared_error(test_part[sales_col], preds)
    mae = mean_absolute_error(test_part[sales_col], preds)

    st.subheader("Метрики точності на історичних даних")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
