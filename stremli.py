# file: app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title("Система прогнозування попиту")

# Завантаження CSV
uploaded_file = st.file_uploader("Завантажте CSV з історичними продажами", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Перші 5 рядків даних:")
    st.dataframe(data.head())

    # Припустимо, що дані мають стовпці: 'Дата', 'Товар', 'Продажі'
    data['Дата'] = pd.to_datetime(data['Дата'])
    
    # Вибір товару
    product_list = data['Товар'].unique()
    product = st.selectbox("Оберіть товар для прогнозу", product_list)
    
    product_data = data[data['Товар'] == product].sort_values('Дата')
    ts = product_data[['Дата', 'Продажі']].set_index('Дата')
    
    # Вибір моделі
    model_option = st.radio("Оберіть модель прогнозу", ["ARIMA", "LinearRegression"])
    
    # Кількість періодів для прогнозу
    periods = st.number_input("Кількість днів для прогнозу", min_value=1, max_value=365, value=30)
    
    if st.button("Зробити прогноз"):
        forecast = None
        if model_option == "ARIMA":
            # Підбір ARIMA(p,d,q) для прикладу (1,1,1)
            model = ARIMA(ts['Продажі'], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
        else:
            # Linear Regression на основі часу
            ts_lr = ts.reset_index()
            ts_lr['t'] = np.arange(len(ts_lr))
            lr = LinearRegression()
            lr.fit(ts_lr[['t']], ts_lr['Продажі'])
            future_t = np.arange(len(ts_lr), len(ts_lr)+periods).reshape(-1,1)
            forecast = lr.predict(future_t)
            forecast = pd.Series(forecast, index=pd.date_range(start=ts.index[-1]+pd.Timedelta(days=1), periods=periods))
        
        # Побудова графіку
        plt.figure(figsize=(10,5))
        plt.plot(ts.index, ts['Продажі'], label="Історія")
        plt.plot(forecast.index, forecast, label="Прогноз", color='red')
        plt.xlabel("Дата")
        plt.ylabel("Продажі")
        plt.title(f"Прогноз продажів для {product}")
        plt.legend()
        st.pyplot(plt)
        
        # Метрики точності на тренувальних даних (для ARIMA можна подивитися на in-sample)
        if model_option == "ARIMA":
            predictions = model_fit.predict(start=0, end=len(ts)-1)
        else:
            predictions = lr.predict(ts_lr[['t']])
        
        mse = mean_squared_error(ts['Продажі'], predictions)
        mae = mean_absolute_error(ts['Продажі'], predictions)
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
      
