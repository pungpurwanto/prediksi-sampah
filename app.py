import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="Prediksi Sampah EPA", layout="wide")
st.title("🗑️ Dashboard Prediksi Timbulan Sampah (EPA)")

@st.cache_data
def load_data():
    # 1. Cari file CSV
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        return None
    
    target_file = files[0]
    
    # 2. Coba baca file dengan melompati baris header laporan EPA yang biasanya ada di atas
    # Kita coba skiprows 2, 3, dan 4
    for skip in [2, 3, 4, 5]:
        try:
            # Gunakan encoding ISO dan engine python untuk fleksibilitas
            df = pd.read_csv(target_file, skiprows=skip, encoding='ISO-8859-1', sep=None, engine='python')
            
            # Ambil kolom ke-1 (Tahun) dan kolom ke-9 (Total Generation)
            # iloc[:, 0] adalah kolom 'Year'
            # iloc[:, 8] adalah kolom 'Total Generation'
            data = df.iloc[:, [0, 8]].copy()
            data.columns = ['Year', 'Total_Generation']
            
            # Konversi ke angka
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Total_Generation'] = pd.to_numeric(data['Total_Generation'], errors='coerce')
            
            # Hapus baris yang kosong (NaN)
            data = data.dropna()
            
            # Validasi: Apakah ada data tahun antara 1960-2020?
            valid_data = data[(data['Year'] >= 1960) & (data['Year'] <= 2020)]
            
            if not valid_data.empty:
                return valid_data.sort_values('Year')
        except:
            continue
    return None

df = load_data()

if df is not None and not df.empty:
    st.sidebar.success("✅ Data Berhasil Dimuat!")
    
    # Ambil info tahun
    last_year = int(df['Year'].max())
    target_year = st.sidebar.slider("Prediksi hingga Tahun:", last_year + 1, 2040, 2030)
    
    # --- MODELING ---
    # Prophet (Paling stabil untuk data tahunan EPA)
    df_p = df.rename(columns={'Year':'ds', 'Total_Generation':'y'})
    df_p['ds'] = pd.to_datetime(df_p['ds'], format='%Y')
    
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=target_year - last_year, freq='Y')
    forecast = m.predict(future)
    
    # --- VISUALISASI ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['Year'], df['Total_Generation'], color='black', label='Data Historis')
    ax.plot(forecast['ds'].dt.year, forecast['yhat'], color='green', label='Prediksi Prophet', linewidth=2)
    
    ax.set_title(f"Proyeksi Timbulan Sampah hingga {target_year}")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Total Sampah (Ribu Ton)")
    ax.legend()
    st.pyplot(fig)
    
    # --- TABEL PREDIKSI ---
    st.subheader(f"📊 Tabel Hasil Prediksi (Target {target_year})")
    pred_val = forecast[['ds', 'yhat']].tail(1)
    st.write(f"Estimasi timbulan sampah pada tahun {target_year} adalah **{pred_val['yhat'].values[0]:,.0f} ribu ton**.")
    
    with st.expander("Lihat Data Mentah Terdeteksi"):
        st.dataframe(df)

else:
    st.error("❌ Data masih tidak terbaca.")
    st.markdown("""
    **Cara Perbaikan Manual:**
    1. Buka file CSV Anda di Notepad/Excel.
    2. Pastikan angka tahun (1960, 1961...) ada di kolom paling kiri (**Kolom A**).
    3. Pastikan angka total sampah ada di **Kolom I** (kolom ke-9).
    4. Hapus baris teks judul di bagian paling atas file sehingga baris 1 atau 2 langsung berisi angka data.
    """)
