import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG DASHBOARD ---
st.set_page_config(page_title="Prediksi Sampah EPA", layout="wide")
st.title("🗑️ Dashboard Prediksi Timbulan Sampah")
st.markdown("Dashboard ini membandingkan 4 model Machine Learning untuk proyeksi sampah hingga 2030.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Menggunakan file yang sudah dibersihkan sebelumnya
    df = pd.read_csv('epa_material_data.csv') 
    return df

df = load_data()

# --- SIDEBAR: PENGATURAN ---
st.sidebar.header("Konfigurasi Model")
selected_models = st.sidebar.multiselect(
    "Pilih Model untuk Ditampilkan:",
    ['ARIMA', 'Prophet', 'XGBoost', 'LSTM'],
    default=['ARIMA', 'Prophet']
)

year_to_predict = st.sidebar.slider("Prediksi hingga Tahun:", 2019, 2040, 2030)

# --- PROSES MODELING (LOGIKA SEDERHANA) ---
st.subheader("Grafik Perbandingan Proyeksi")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Asli')

# Logika plotting model berdasarkan checkbox di sidebar
# (Catatan: Di sini Anda masukkan fungsi prediksi yang sudah kita buat sebelumnya)

if 'Prophet' in selected_models:
    # Contoh implementasi cepat Prophet
    df_p = df[['Year', 'Total_Generation']].rename(columns={'Year': 'ds', 'Total_Generation': 'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    m = Prophet().fit(df_p)
    future = m.make_future_dataframe(periods=year_to_predict-2018, freq='Y')
    forecast = m.predict(future)
    ax.plot(forecast['ds'].dt.year, forecast['yhat'], label='Prophet', color='green')

# ... tambahkan logika untuk model lain ...

ax.legend()
st.pyplot(fig)

# --- TABEL HASIL ---
st.subheader("Estimasi Angka Prediksi")
# Tampilkan dataframe hasil angka prediksi di sini
st.dataframe(df.tail(10))