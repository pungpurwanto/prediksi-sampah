import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import Algoritma (Tanpa Tensorflow/LSTM)
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor

# Konfigurasi Halaman
st.set_page_config(page_title="Forecasting Sampah EPA", layout="wide", page_icon="🗑️")

st.title("🗑️ Dashboard Prediksi Timbulan Sampah")
st.markdown("Analisis perbandingan model **Prophet, Auto-ARIMA, dan XGBoost** untuk proyeksi sampah hingga 2030+.")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        return None
    
    target_file = files[0]
    # Mencoba berbagai cara membaca file EPA
    for skip in [2, 3, 4, 5]:
        try:
            df = pd.read_csv(target_file, skiprows=skip, encoding='ISO-8859-1', sep=None, engine='python')
            # Ambil Kolom Tahun (index 0) dan Total Generation (index 8)
            data = df.iloc[:, [0, 8]].copy()
            data.columns = ['Year', 'Total_Generation']
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Total_Generation'] = pd.to_numeric(data['Total_Generation'], errors='coerce')
            data = data.dropna()
            data = data[(data['Year'] >= 1960) & (data['Year'] <= 2018)]
            if not data.empty:
                return data.sort_values('Year').reset_index(drop=True)
        except:
            continue
    return None

df = load_data()

if df is not None:
    # --- SIDEBAR PENGATURAN ---
    st.sidebar.header("⚙️ Konfigurasi Model")
    selected_algos = st.sidebar.multiselect(
        "Pilih Algoritma:",
        ['Prophet', 'Auto-ARIMA', 'XGBoost'],
        default=['Prophet', 'Auto-ARIMA', 'XGBoost']
    )
    
    last_year = int(df['Year'].max())
    target_year = st.sidebar.slider("Prediksi Sampai Tahun:", last_year + 1, 2040, 2030)
    n_years = target_year - last_year
    years_future = np.arange(last_year + 1, target_year + 1)

    # --- PROSES MODELING ---
    forecasts = {}
    
    # 1. Prophet
    if 'Prophet' in selected_algos:
        with st.spinner('Menghitung Prophet...'):
            m_p = Prophet(yearly_seasonality=True).fit(df.rename(columns={'Year':'ds', 'Total_Generation':'y'}))
            fut_p = m_p.make_future_dataframe(periods=n_years, freq='Y')
            res_p = m_p.predict(fut_p)
            forecasts['Prophet'] = res_p.iloc[-n_years:]['yhat'].values

    # 2. Auto-ARIMA
    if 'Auto-ARIMA' in selected_algos:
        with st.spinner('Menghitung Auto-ARIMA...'):
            m_a = pm.auto_arima(df['Total_Generation'], seasonal=False, suppress_warnings=True)
            forecasts['Auto-ARIMA'] = m_a.predict(n_periods=n_years)

    # 3. XGBoost
    if 'XGBoost' in selected_algos:
        with st.spinner('Menghitung XGBoost...'):
            m_x = XGBRegressor(n_estimators=100, learning_rate=0.1).fit(df[['Year']], df['Total_Generation'])
            forecasts['XGBoost'] = m_x.predict(pd.DataFrame({'Year': years_future}))

    # --- VISUALISASI ---
    st.subheader("📈 Grafik Perbandingan Prediksi")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Data Historis
    ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis (EPA)', s=40, alpha=0.6)
    
    # Plot Garis Prediksi
    colors = {'Prophet': '#2ca02c', 'Auto-ARIMA': '#1f77b4', 'XGBoost': '#ff7f0e'}
    for name, vals in forecasts.items():
        ax.plot(years_future, vals, label=f"Prediksi {name}", linewidth=3, marker='o', markersize=4, color=colors.get(name))
    
    ax.axvline(x=last_year, color='red', linestyle='--', alpha=0.5, label='Batas Data Historis')
    ax.set_title("Proyeksi Tren Sampah Masa Depan", fontsize=14)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Total Sampah (Ribu Ton)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

    # --- TABEL HASIL ---
    st.divider()
    st.subheader(f"📋 Estimasi Angka Tahun {target_year}")
    
    if forecasts:
        cols = st.columns(len(forecasts))
        for i, (name, vals) in enumerate(forecasts.items()):
            # Menggunakan .iloc[-1] agar selalu mengambil baris terakhir 
            # baik itu list, numpy array, maupun pandas series
            try:
                if hasattr(vals, "iloc"):
                    last_val = vals.iloc[-1]
                else:
                    last_val = vals[-1]
                
                cols[i].metric(name, f"{last_val:,.0f} k-tons")
            except Exception as e:
                cols[i].error("Error format")
    else:
        st.warning("Silakan pilih algoritma di sidebar untuk melihat hasil estimasi.")
