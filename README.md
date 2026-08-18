# Sistem Prediksi Permintaan Produk Farmasi dan Alat Kesehatan

Project ini digunakan untuk memprediksi permintaan produk berdasarkan data historis transaksi penjualan periode Januari 2025 sampai Desember 2025.

## Judul Project

Perbandingan Metode Single Moving Average dan Decision Tree Regressor untuk Prediksi Permintaan Produk Farmasi dan Alat Kesehatan Berbasis Streamlit

## Metode yang Digunakan

1. Single Moving Average
2. Decision Tree Regressor

## Tujuan

Membantu memperkirakan permintaan produk pada periode bulan berikutnya agar dapat mengurangi risiko kekurangan stok dan kelebihan stok.

## Dataset

Dataset yang digunakan merupakan data transaksi penjualan produk periode Januari 2025 sampai Desember 2025.

## Alur Program

1. Membaca dataset transaksi.
2. Membersihkan data.
3. Melakukan konversi satuan.
4. Membentuk data penjualan bulanan.
5. Membuat fitur time series seperti Lag dan Rolling Mean.
6. Melatih model Single Moving Average dan Decision Tree Regressor.
7. Mengevaluasi model menggunakan MAE, RMSE, MAPE, MAPE Aktual Minimal 10, dan WMAPE.
8. Memilih model terbaik berdasarkan WMAPE.
9. Membuat prediksi permintaan bulan berikutnya.
10. Menampilkan hasil prediksi dan evaluasi melalui aplikasi Streamlit.

## Struktur Folder

- `app/` : aplikasi Streamlit
- `data/` : dataset mentah dan file konversi satuan
- `src/` : source code preprocessing, training, evaluasi, dan prediksi
- `model/` : file model dan hasil prediksi
- `output/` : hasil preprocessing, evaluasi, dan prediksi
- `notebook/` : eksplorasi awal data

## Cara Menjalankan Program

Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```
