import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
import os

# --- CONFIG DASHBOARD ---
st.set_page_config(page_title="Prediksi Sampah EPA", layout="wide")
st.title("🗑️ Dashboard Prediksi Timbulan Sampah (EPA)")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Mencari file CSV apapun di folder root jika nama file berubah-ubah
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        st.error("❌ File CSV tidak ditemukan di repositori GitHub Anda!")
        return None
    
    # Membaca file pertama yang ditemukan, lewati 3 baris header seperti diskusi sebelumnya
    target_file = files[0]
    df = pd.read_csv(target_file, skiprows=3)
    
    # Pembersihan dasar
    df_clean = df.iloc[:, [0, 8]].copy()
    df_clean.columns = ['Year', 'Total_Generation']
    df_clean = df_clean.dropna()
    df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
    df_clean['Total_Generation'] = pd.to_numeric(df_clean['Total_Generation'], errors='coerce')
    df_clean = df_clean.dropna().sort_values('Year')
    return df_clean

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Konfigurasi")
    models_to_run = st.sidebar.multiselect(
        "Pilih Model:",
        ['ARIMA', 'Prophet', 'XGBoost'],
        default=['Prophet']
    )
    future_year = st.sidebar.slider("Prediksi hingga Tahun:", 2020, 2040, 2030)
    n_periods = future_year - 2018

    # --- TAMPILAN UTAMA ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Visualisasi Proyeksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis', s=20)
        
        years_pred = np.arange(2019, future_year + 1)
        results = {}

        # 1. PROPHET
        if 'Prophet' in models_to_run:
            df_p = df.rename(columns={'Year': 'ds', 'Total_Generation': 'y'})
            df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
            m = Prophet().fit(df_p)
            future = m.make_future_dataframe(periods=n_periods, freq='Y')
            forecast = m.predict(future)
            ax.plot(forecast['ds'].dt.year, forecast['yhat'], label='Prophet', color='green')
            results['Prophet'] = forecast['yhat'].iloc[-1]

        # 2. ARIMA
        if 'ARIMA' in models_to_run:
            m_arima = pm.auto_arima(df['Total_Generation'], seasonal=False)
            f_arima = m_arima.predict(n_periods=n_periods)
            ax.plot(years_pred, f_arima, label='ARIMA', color='blue', linestyle='--')
            results['ARIMA'] = f_arima.iloc[-1]

        # 3. XGBOOST
        if 'XGBoost' in models_to_run:
            m_xgb = XGBRegressor().fit(df[['Year']], df['Total_Generation'])
            f_xgb = m_xgb.predict(pd.DataFrame({'Year': years_pred}))
            ax.plot(years_pred, f_xgb, label='XGBoost', color='orange', linestyle=':')
            results['XGBoost'] = f_xgb[-1]

        ax.legend()
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ribu Ton")
        st.pyplot(fig)

    with col2:
        st.subheader("Hasil Akhir")
        if results:
            st.write(f"Estimasi Tahun {future_year}:")
            for m, val in results.items():
                st.metric(m, f"{val:,.0f} ton")
        else:
            st.info("Pilih model di sidebar.")

    st.divider()
    st.subheader("Data Mentah (Cleaned)")
    st.dataframe(df, use_container_width=True)
