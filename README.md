# Analisis dan Prediksi Timbulan Sampah (EPA Dataset 1960-2018)

Proyek ini bertujuan untuk melakukan analisis tren historis dan prediksi masa depan terhadap timbulan sampah (*waste generation*) menggunakan berbagai algoritma Machine Learning dan Deep Learning. Dashboard interaktif dibangun menggunakan **Streamlit** untuk memvisualisasikan perbandingan antar model secara real-time.

## 🚀 Fitur Utama
- **Analisis Historis:** Visualisasi data EPA dari tahun 1960 hingga 2018.
- **Multi-Model Forecasting:** Perbandingan 4 model prediksi utama:
  - **Auto-ARIMA:** Model statistik klasik untuk data deret waktu.
  - **Facebook Prophet:** Model yang andal untuk menangkap tren dan musiman.
  - **XGBoost:** Algoritma berbasis pohon keputusan untuk akurasi tinggi pada data tabel.
  - **LSTM (Long Short-Term Memory):** Pendekatan Deep Learning untuk pola jangka panjang.
- **Dashboard Interaktif:** Pengguna dapat menentukan rentang tahun prediksi (hingga 2030+).

## 📊 Dataset
Dataset yang digunakan berasal dari **Environmental Protection Agency (EPA)** yang mencakup data generasi material, daur ulang, dan pengomposan (dalam ribu ton).
- **Sumber:** [Kaggle - EPA Material Generation & Recycling](https://www.kaggle.com/)
- **Rentang Data:** 1960 - 2018.

## 🛠️ Teknologi yang Digunakan
- **Bahasa:** Python 3.9+
- **Library Utama:**
  - `streamlit` (Dashboard interface)
  - `prophet` (Time-series forecasting)
  - `pmdarima` (Auto-ARIMA)
  - `xgboost` (Regression)
  - `tensorflow/keras` (LSTM)
  - `pandas`, `matplotlib`, `scikit-learn` (Data processing & Viz)

## 💻 Cara Menjalankan Secara Lokal

1. **Clone Repository**
   ```bash
   git clone [https://github.com/username-anda/nama-repo.git](https://github.com/username-anda/nama-repo.git)
   cd nama-repo
