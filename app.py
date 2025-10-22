import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Прогнозування попиту", layout="wide")
st.title("📊 Система прогнозування попиту")

# --- 1. Завантаження CSV ---
uploaded_file = st.file_uploader("Завантаж CSV з історичними продажами", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    st.success("Файл завантажено!")
    st.dataframe(df.head())

    # --- 2. Вибір товару ---
    product_list = df['product'].unique()
    selected_product = st.selectbox("Оберіть товар для прогнозу", product_list)

    product_df = df[df['product'] == selected_product].sort_values('date')
    product_df = product_df.set_index('date')

    st.subheader(f"Історія продажів товару: {selected_product}")
    st.line_chart(product_df['sales'])

    # --- 3. Вибір моделі ---
    model_choice = st.radio("Виберіть модель прогнозу", ("ARIMA", "Linear Regression"))

    # --- 4. Прогноз ---
    forecast_horizon = st.number_input("Кількість днів для прогнозу", min_value=1, value=7)

    if st.button("Зробити прогноз"):
        if model_choice == "Linear Regression":
            # Підготовка даних
            product_df = product_df.reset_index()
            product_df['day'] = (product_df['date'] - product_df['date'].min()).dt.days
            X = product_df[['day']]
            y = product_df['sales']

            model = LinearRegression()
            model.fit(X, y)

            future_days = np.array(range(X['day'].max() + 1, X['day'].max() + 1 + forecast_horizon)).reshape(-1,1)
            forecast = model.predict(future_days)

            # Метрики на історичних даних
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

        else:  # ARIMA
            model = ARIMA(product_df['sales'], order=(1,1,1))  # простий приклад
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_horizon)

            # Метрики на історичних даних
            y_pred = model_fit.fittedvalues
            y_true = product_df['sales'][1:]  # через диференціювання
            mse = mean_squared_error(y_true, y_pred[1:])
            mae = mean_absolute_error(y_true, y_pred[1:])

        # --- 5. Візуалізація прогнозу ---
        st.subheader("Графік прогнозу")
        plt.figure(figsize=(10,5))
        plt.plot(product_df['date'], product_df['sales'], label='Історія')
        future_dates = pd.date_range(product_df['date'].max() + pd.Timedelta(days=1), periods=forecast_horizon)
        plt.plot(future_dates, forecast, color='red', marker='o', label='Прогноз')
        plt.xlabel("Дата")
        plt.ylabel("Продажі")
        plt.legend()
        st.pyplot(plt)

        # --- 6. Метрики точності ---
        st.subheader("Метрики точності на історичних даних")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

