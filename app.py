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
st.markdown("Dashboard ini memproyeksikan total timbulan sampah berdasarkan data historis EPA (1960-2018).")

# --- FUNGSI LOAD & CLEAN DATA ---
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        st.error("❌ File CSV tidak ditemukan di repositori!")
        return None
    
    target_file = files[0]
    
    try:
        # 1. Gunakan sep=None dan engine='python' agar Pandas menebak sendiri pemisahnya (koma/titik koma)
        df = pd.read_csv(target_file, skiprows=3, encoding='ISO-8859-1', sep=None, engine='python', on_bad_lines='skip')
        
        # 2. Bersihkan nama kolom dari spasi atau karakter aneh
        df.columns = [str(c).strip() for c in df.columns]
        
        # 3. Validasi jumlah kolom untuk mencegah error 'out-of-bounds'
        if df.shape[1] < 2:
            st.error(f"❌ File terdeteksi hanya memiliki {df.shape[1]} kolom. Pastikan pemisah CSV benar.")
            return None

        # 4. Cari kolom Tahun (biasanya kolom pertama)
        # Cari kolom Total Generation (Kita cari kolom yang mengandung kata 'Total' dan 'Generation')
        col_tahun = df.columns[0]
        col_target = None
        
        # Mencari kolom yang kemungkinan besar adalah 'Total Generation'
        potential_cols = [c for c in df.columns if 'Total' in c and 'Generation' in c]
        if potential_cols:
            col_target = potential_cols[0]
        elif df.shape[1] >= 9: # Jika tidak ketemu namanya, ambil index ke-8 (kolom ke-9)
            col_target = df.columns[8]
        else:
            col_target = df.columns[-1] # Ambil kolom terakhir sebagai cadangan

        df_clean = df[[col_tahun, col_target]].copy()
        df_clean.columns = ['Year', 'Total_Generation']
        
        # 5. Konversi ke numerik
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean['Total_Generation'] = pd.to_numeric(df_clean['Total_Generation'], errors='coerce')
        
        # 6. Hapus baris yang tidak valid
        df_clean = df_clean.dropna()
        df_clean = df_clean[df_clean['Year'] > 1900].sort_values('Year')
        df_clean['Year'] = df_clean['Year'].astype(int)
        
        return df_clean
    except Exception as e:
        st.error(f"❌ Error Detail: {e}")
        return None

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Pengaturan")
    selected_models = st.sidebar.multiselect(
        "Pilih Model:",
        ['Prophet', 'Auto-ARIMA', 'XGBoost', 'Linear Regression'],
        default=['Prophet', 'Linear Regression']
    )
    target_year = st.sidebar.slider("Prediksi hingga:", 2020, 2040, 2030)
    
    # Persiapan variabel waktu
    last_year = int(df['Year'].max())
    n_years = target_year - last_year
    years_future = np.arange(last_year + 1, target_year + 1)

    # --- VISUALISASI ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📈 Grafik Proyeksi")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis', s=25)
        
        final_values = {}

        # 1. PROPHET
        if 'Prophet' in selected_models:
            m_p = Prophet().fit(df.rename(columns={'Year':'ds', 'Total_Generation':'y'}))
            fut_p = m_p.make_future_dataframe(periods=n_years, freq='Y')
            fc_p = m_p.predict(fut_p)
            ax.plot(fc_p['ds'].dt.year, fc_p['yhat'], label='Prophet', color='green')
            final_values['Prophet'] = fc_p['yhat'].iloc[-1]

        # 2. LINEAR REGRESSION
        if 'Linear Regression' in selected_models:
            m_lr = LinearRegression().fit(df[['Year']], df['Total_Generation'])
            fc_lr = m_lr.predict(years_future.reshape(-1, 1))
            ax.plot(years_future, fc_lr, label='Linear Regression', color='red', linestyle='--')
            final_values['Linear Regression'] = fc_lr[-1]

        # 3. XGBOOST
        if 'XGBoost' in selected_models:
            m_xgb = XGBRegressor().fit(df[['Year']], df['Total_Generation'])
            fc_xgb = m_xgb.predict(pd.DataFrame({'Year': years_future}))
            ax.plot(years_future, fc_xgb, label='XGBoost', color='orange', linestyle=':')
            final_values['XGBoost'] = fc_xgb[-1]

        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader(f"Estimasi {target_year}")
        for m, v in final_values.items():
            st.metric(m, f"{v:,.0f} k-tons")

    st.divider()
    st.subheader("📋 Data Historis")
    st.dataframe(df, use_container_width=True)
