import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import os

# --- CONFIG ---
st.set_page_config(page_title="Prediksi Sampah EPA", layout="wide")
st.title("🗑️ Dashboard Prediksi Timbulan Sampah (EPA)")

@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        st.error("❌ File CSV tidak ditemukan!")
        return None
    
    target_file = files[0]
    
    # Mencoba berbagai kombinasi pembacaan
    encodings = ['ISO-8859-1', 'utf-8', 'cp1252']
    separators = [',', ';']
    
    for enc in encodings:
        for sep in separators:
            try:
                # Membaca file
                df = pd.read_csv(target_file, sep=sep, encoding=enc, on_bad_lines='skip')
                
                # Cari baris di mana kata 'Year' muncul (mencari header yang sebenarnya)
                # Seringkali header asli ada di baris ke-3 atau ke-4
                for i in range(len(df)):
                    if 'Year' in str(df.iloc[i, 0]):
                        df = pd.read_csv(target_file, sep=sep, encoding=enc, skiprows=i+1)
                        break
                
                # Bersihkan nama kolom
                df.columns = [str(c).strip() for c in df.columns]
                
                # Ambil kolom Year dan Total Generation
                # Berdasarkan struktur EPA: Kolom 0 adalah Tahun, Kolom 8 adalah Total Generation
                data = df.iloc[:, [0, 8]].copy()
                data.columns = ['Year', 'Total_Generation']
                
                # Konversi ke angka
                data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
                data['Total_Generation'] = pd.to_numeric(data['Total_Generation'], errors='coerce')
                
                # Hapus baris kosong
                data = data.dropna()
                
                # Filter hanya tahun yang valid (1960-2018)
                data = data[(data['Year'] >= 1960) & (data['Year'] <= 2020)]
                
                if not data.empty:
                    return data.sort_values('Year')
            except:
                continue
                
    st.error("❌ Gagal memproses struktur file CSV. Pastikan file tidak rusak.")
    return None

df = load_data()

# Proteksi jika df kosong
if df is not None and not df.empty:
    st.sidebar.header("⚙️ Pengaturan")
    selected_models = st.sidebar.multiselect(
        "Pilih Model:",
        ['Prophet', 'Auto-ARIMA', 'XGBoost', 'Linear Regression'],
        default=['Prophet', 'Linear Regression']
    )
    
    # Ambil tahun terakhir dengan aman
    current_last_year = int(df['Year'].max())
    target_year = st.sidebar.slider("Prediksi hingga:", current_last_year + 1, 2040, 2030)
    
    n_years = target_year - current_last_year
    years_future = np.arange(current_last_year + 1, target_year + 1)

    # --- VISUALISASI ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📈 Grafik Proyeksi")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis', s=25)
        
        final_values = {}

        # Logic model tetap sama seperti sebelumnya
        if 'Prophet' in selected_models:
            m_p = Prophet().fit(df.rename(columns={'Year':'ds', 'Total_Generation':'y'}))
            fut_p = m_p.make_future_dataframe(periods=n_years, freq='Y')
            fc_p = m_p.predict(fut_p)
            ax.plot(fc_p['ds'].dt.year, fc_p['yhat'], label='Prophet', color='green')
            final_values['Prophet'] = fc_p['yhat'].iloc[-1]

        if 'Linear Regression' in selected_models:
            m_lr = LinearRegression().fit(df[['Year']], df['Total_Generation'])
            fc_lr = m_lr.predict(years_future.reshape(-1, 1))
            ax.plot(years_future, fc_lr, label='Linear Regression', color='red', linestyle='--')
            final_values['Linear Regression'] = fc_lr[-1]

        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader(f"Estimasi {target_year}")
        for m, v in final_values.items():
            st.metric(m, f"{v:,.0f} k-tons")

    st.divider()
    st.subheader("📋 Data Historis Terdeteksi")
    st.write(df)
else:
    st.warning("⚠️ Data tidak ditemukan. Pastikan file CSV Anda berisi kolom 'Year' dan data numerik yang benar.")
