import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Sampah EPA", layout="wide", page_icon="🗑️")

st.title("🗑️ Dashboard Analisis & Prediksi Timbulan Sampah (EPA)")
st.markdown("""
Dashboard ini membandingkan berbagai model Machine Learning untuk memproyeksikan total timbulan sampah berdasarkan data historis EPA (1960-2018).
""")

# --- FUNGSI LOAD & CLEAN DATA ---
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        st.error("❌ File CSV tidak ditemukan!")
        return None
    
    target_file = files[0]
    
    try:
        # Gunakan sep=None dan engine='python' agar pandas mendeteksi pemisah secara otomatis
        # on_bad_lines='skip' akan melewati baris yang jumlah kolomnya tidak konsisten
        df = pd.read_csv(
            target_file, 
            skiprows=3, 
            encoding='ISO-8859-1', 
            sep=None, 
            engine='python', 
            on_bad_lines='skip'
        )
        
        # Membersihkan nama kolom dari spasi tambahan
        df.columns = [str(c).strip() for c in df.columns]
        
        # Berdasarkan file EPA: Kolom pertama biasanya Tahun, 
        # dan kolom 'Total' (generasi) biasanya berada di indeks ke-8
        # Kita coba ambil berdasarkan posisi indeks untuk menghindari error nama kolom
        df_clean = df.iloc[:, [0, 8]].copy()
        df_clean.columns = ['Year', 'Total_Generation']
        
        # Konversi data ke numerik
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean['Total_Generation'] = pd.to_numeric(df_clean['Total_Generation'], errors='coerce')
        
        # Hapus baris yang memiliki NaN setelah konversi
        df_clean = df_clean.dropna().sort_values('Year')
        
        # Pastikan tipe data tahun adalah integer
        df_clean['Year'] = df_clean['Year'].astype(int)
        
        # Filter data agar masuk akal (Tahun > 1900)
        df_clean = df_clean[df_clean['Year'] > 1900]
        
        return df_clean

    except Exception as e:
        st.error(f"❌ Gagal memproses data: {e}")
        return None

# Load dataset
df = load_data()

if df is not None:
    # --- SIDEBAR KONFIGURASI ---
    st.sidebar.header("⚙️ Pengaturan Model")
    selected_models = st.sidebar.multiselect(
        "Pilih Model Prediksi:",
        ['Prophet', 'Auto-ARIMA', 'XGBoost', 'Linear Regression'],
        default=['Prophet', 'Linear Regression']
    )
    
    target_year = st.sidebar.slider("Prediksi hingga Tahun:", 2020, 2040, 2030)
    n_years = target_year - 2018
    years_future = np.arange(2019, target_year + 1)

    # --- LAYOUT KOLOM ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📈 Grafik Proyeksi Timbulan Sampah")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Data Historis
        ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis (EPA)', s=30, alpha=0.7)
        
        final_preds = {}

        # 1. MODEL PROPHET
        if 'Prophet' in selected_models:
            with st.spinner('Menghitung Prophet...'):
                df_p = df.rename(columns={'Year': 'ds', 'Total_Generation': 'y'})
                df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
                m_prophet = Prophet(yearly_seasonality=True)
                m_prophet.fit(df_p)
                future_p = m_prophet.make_future_dataframe(periods=n_years, freq='Y')
                forecast_p = m_prophet.predict(future_p)
                ax.plot(forecast_p['ds'].dt.year, forecast_p['yhat'], label='Prophet', linewidth=2, color='green')
                final_preds['Prophet'] = forecast_p['yhat'].iloc[-1]

        # 2. MODEL AUTO-ARIMA
        if 'Auto-ARIMA' in selected_models:
            with st.spinner('Menghitung ARIMA...'):
                m_arima = pm.auto_arima(df['Total_Generation'], seasonal=False, suppress_warnings=True)
                forecast_arima = m_arima.predict(n_periods=n_years)
                ax.plot(years_future, forecast_arima, label='Auto-ARIMA', linestyle='--', linewidth=2, color='blue')
                final_preds['Auto-ARIMA'] = forecast_arima.iloc[-1]

        # 3. MODEL XGBOOST
        if 'XGBoost' in selected_models:
            with st.spinner('Menghitung XGBoost...'):
                m_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
                m_xgb.fit(df[['Year']], df['Total_Generation'])
                forecast_xgb = m_xgb.predict(pd.DataFrame({'Year': years_future}))
                ax.plot(years_future, forecast_xgb, label='XGBoost', linestyle=':', linewidth=2, color='orange')
                final_preds['XGBoost'] = forecast_xgb[-1]

        # 4. LINEAR REGRESSION
        if 'Linear Regression' in selected_models:
            m_lr = LinearRegression()
            m_lr.fit(df[['Year']], df['Total_Generation'])
            forecast_lr = m_lr.predict(years_future.reshape(-1, 1))
            ax.plot(years_future, forecast_lr, label='Linear Regression', linestyle='-.', linewidth=2, color='red')
            final_preds['Linear Regression'] = forecast_lr[-1]

        # Pengaturan Grafik
        ax.axvline(x=2018, color='grey', linestyle='--', alpha=0.5)
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total Sampah (Ribu Ton)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("📌 Estimasi Tahun " + str(target_year))
        if final_preds:
            for model_name, value in final_preds.items():
                st.metric(label=model_name, value=f"{value:,.0f} k-tons")
            
            st.info("Keterangan: Satuan dalam ribu ton.")
        else:
            st.warning("Pilih setidaknya satu model di sidebar.")

    # --- TABEL DATA ---
    st.divider()
    with st.expander("👁️ Lihat Data Historis"):
        st.dataframe(df.style.format({"Year": "{:.0f}", "Total_Generation": "{:,.2f}"}), use_container_width=True)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Data Source: US EPA Material Generation and Recycling Data.")
