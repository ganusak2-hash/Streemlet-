import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.title("Система прогнозування попиту")

url = "https://raw.githubusercontent.com/USERNAME/REPO/main/sales_example.csv"

try:
    data = pd.read_csv(url, encoding="utf-8")
except Exception:
    st.warning("Не вдалося завантажити з GitHub, використовується локальний файл.")
    data = pd.read_csv("sales_example.csv", encoding="utf-8")

st.subheader("Історичні дані продажів")
st.dataframe(data)

X = data.index.values.reshape(-1, 1)
y = data['Sales'].values

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

mse = mean_squared_error(y, pred)
mae = mean_absolute_error(y, pred)

st.subheader("Результати прогнозування")
st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")

fig, ax = plt.subplots()
ax.plot(data["Month"], y, marker='o', label="Фактичні продажі")
ax.plot(data["Month"], pred, linestyle='--', label="Прогноз")
ax.legend()
st.pyplot(fig)
