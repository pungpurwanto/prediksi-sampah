import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, r2_score

# Import Algoritma
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor

# Konfigurasi Halaman
st.set_page_config(page_title="Forecasting Sampah EPA", layout="wide", page_icon="🗑️")

st.title("🗑️ Dashboard Prediksi Timbulan Sampah")
st.markdown("Analisis perbandingan model untuk proyeksi sampah berdasarkan data EPA (1960-2018).")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files: return None
    target_file = files[0]
    for skip in [2, 3, 4, 5]:
        try:
            df = pd.read_csv(target_file, skiprows=skip, encoding='ISO-8859-1', sep=None, engine='python')
            data = df.iloc[:, [0, 8]].copy()
            data.columns = ['Year', 'Total_Generation']
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Total_Generation'] = pd.to_numeric(data['Total_Generation'], errors='coerce')
            data = data.dropna()
            data = data[(data['Year'] >= 1960) & (data['Year'] <= 2018)]
            if not data.empty:
                return data.sort_values('Year').reset_index(drop=True)
        except: continue
    return None

df = load_data()

if df is not None:
    # --- SIDEBAR ---
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

    forecasts = {}
    metrics = {}

    # --- PROSES MODELING & EVALUASI ---
    # 1. Prophet
    if 'Prophet' in selected_algos:
        with st.spinner('Calculating Prophet...'):
            m_p = Prophet(yearly_seasonality=True).fit(df.rename(columns={'Year':'ds', 'Total_Generation':'y'}))
            # Evaluasi pada data historis
            hist_p = m_p.predict(df.rename(columns={'Year':'ds'}))['yhat']
            metrics['Prophet'] = (mean_absolute_error(df['Total_Generation'], hist_p), r2_score(df['Total_Generation'], hist_p))
            # Forecast
            fut_p = m_p.make_future_dataframe(periods=n_years, freq='Y')
            forecasts['Prophet'] = m_p.predict(fut_p).iloc[-n_years:]['yhat']

    # 2. Auto-ARIMA
    if 'Auto-ARIMA' in selected_algos:
        with st.spinner('Calculating ARIMA...'):
            m_a = pm.auto_arima(df['Total_Generation'], seasonal=False)
            hist_a = m_a.predict_in_sample()
            metrics['Auto-ARIMA'] = (mean_absolute_error(df['Total_Generation'], hist_a), r2_score(df['Total_Generation'], hist_a))
            forecasts['Auto-ARIMA'] = m_a.predict(n_periods=n_years)

    # 3. XGBoost
    if 'XGBoost' in selected_algos:
        with st.spinner('Calculating XGBoost...'):
            m_x = XGBRegressor().fit(df[['Year']], df['Total_Generation'])
            hist_x = m_x.predict(df[['Year']])
            metrics['XGBoost'] = (mean_absolute_error(df['Total_Generation'], hist_x), r2_score(df['Total_Generation'], hist_x))
            forecasts['XGBoost'] = m_x.predict(pd.DataFrame({'Year': years_future}))

    # --- VISUALISASI ---
    st.subheader("📈 Grafik Perbandingan Prediksi")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis', s=40, alpha=0.5)
    
    colors = {'Prophet': '#2ca02c', 'Auto-ARIMA': '#1f77b4', 'XGBoost': '#ff7f0e'}
    for name, vals in forecasts.items():
        ax.plot(years_future, vals, label=f"Prediksi {name}", linewidth=3, marker='o', markersize=4, color=colors[name])
    
    ax.axvline(x=last_year, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # --- TABEL HASIL & METRIK ---
    st.divider()
    st.subheader(f"📊 Hasil Estimasi Tahun {target_year} & Akurasi Model")
    
    if forecasts:
        cols = st.columns(len(forecasts))
        for i, (name, vals) in enumerate(forecasts.items()):
            # Ambil nilai terakhir dengan iloc untuk keamanan index
            last_val = vals.iloc[-1] if hasattr(vals, "iloc") else vals[-1]
            mae, r2 = metrics[name]
            
            with cols[i]:
                st.metric(name, f"{last_val:,.0f} k-tons")
                st.caption(f"**MAE:** {mae:,.2f}")
                st.caption(f"**R² Score:** {r2:.4f}")
    else:
        st.info("Pilih model di sidebar.")

else:
    st.error("Gagal memuat data. Periksa file CSV Anda.")
