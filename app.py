import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Заголовок
st.title("Система прогнозування попиту")

# Посилання на CSV з GitHub (замінити на свій файл)
url = "https://raw.githubusercontent.com/USERNAME/REPO/main/sales_example.csv"

# Зчитування CSV
data = pd.read_csv(url, encoding="utf-8")

# Відображення даних
st.subheader("📊 Історичні дані продажів")
st.dataframe(data)

# Підготовка даних
X = data.index.values.reshape(-1, 1)
y = data['Sales'].values

# Модель лінійної регресії
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Метрики точності
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)

st.subheader("📈 Результати прогнозування")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

# Побудова графіка
fig, ax = plt.subplots()
ax.plot(data["Month"], y, label="Фактичні продажі", marker='o')
ax.plot(data["Month"], predictions, label="Прогноз", linestyle='--')
ax.legend()
st.pyplot(fig)
