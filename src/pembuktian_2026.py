import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.append(str(BASE_DIR / "src"))

from preprocessing import clean_data, load_conversion, apply_unit_conversion

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

def load_actual_2026():
    candidates = [
        BASE_DIR / "dataset 2026 Jan-Juni.xlsx",
        BASE_DIR / "PENJUALAN NPP JAN - JUNI 2026.xlsx",
        DATA_DIR / "dataset 2026 Jan-Juni.xlsx",
        DATA_DIR / "PENJUALAN NPP JAN - JUNI 2026.xlsx",
    ]
    
    raw_2026 = None
    for file_path in candidates:
        if file_path.exists():
            try:
                raw_2026 = pd.read_excel(file_path)
                print(f"  -> Berhasil membaca data aktual dari: {file_path.name}")
                break
            except PermissionError:
                continue
    
    if raw_2026 is None:
        raise PermissionError(
            "Semua file dataset 2026 sedang terbuka di Microsoft Excel atau dikunci oleh OneDrive. "
            "Silakan tutup terlebih dahulu file Excel dataset 2026 lalu jalankan ulang."
        )
    
    clean_2026 = clean_data(raw_2026)
    konversi = load_conversion()
    conv_2026 = apply_unit_conversion(clean_2026, konversi)
    return conv_2026

def bounded_percentage_error(actual, pred):
    if actual == 0:
        return 0.0 if pred == 0 else 100.0
    err = abs(actual - pred) / actual * 100.0
    return min(100.0, err)

def evaluate_january_2026(conv_2026, pred_df):
    # Filter aktual untuk Januari 2026
    jan_actual = conv_2026[conv_2026["Periode"] == "2026-01"].groupby(
        ["Kode Item", "Satuan Standar"], as_index=False
    )["Jumlah Standar"].sum()
    jan_actual.rename(columns={"Jumlah Standar": "Aktual_Januari_2026"}, inplace=True)
    
    eval_df = pred_df.merge(jan_actual, on=["Kode Item", "Satuan Standar"], how="left")
    eval_df["Aktual_Januari_2026"] = eval_df["Aktual_Januari_2026"].fillna(0).astype(int)
    
    eval_df["Selisih_Decision_Tree"] = eval_df["Prediksi_Decision_Tree"] - eval_df["Aktual_Januari_2026"]
    eval_df["Selisih_SMA"] = eval_df["Prediksi_SMA"] - eval_df["Aktual_Januari_2026"]
    eval_df["Abs_Error_Decision_Tree"] = eval_df["Selisih_Decision_Tree"].abs()
    eval_df["Abs_Error_SMA"] = eval_df["Selisih_SMA"].abs()
    
    eval_df["Bounded_Error_Decision_Tree"] = eval_df.apply(lambda r: bounded_percentage_error(r["Aktual_Januari_2026"], r["Prediksi_Decision_Tree"]), axis=1)
    eval_df["Bounded_Error_SMA"] = eval_df.apply(lambda r: bounded_percentage_error(r["Aktual_Januari_2026"], r["Prediksi_SMA"]), axis=1)
    
    out_path = OUTPUT_DIR / "pembuktian_januari_2026.csv"
    eval_df.sort_values("Aktual_Januari_2026", ascending=False).to_csv(out_path, index=False)
    
    total_actual = eval_df["Aktual_Januari_2026"].sum()
    dt_mae = eval_df["Abs_Error_Decision_Tree"].mean()
    dt_rmse = np.sqrt((eval_df["Abs_Error_Decision_Tree"] ** 2).mean())
    dt_wmape = (eval_df["Abs_Error_Decision_Tree"].sum() / total_actual * 100) if total_actual > 0 else 0
    
    sma_mae = eval_df["Abs_Error_SMA"].mean()
    sma_rmse = np.sqrt((eval_df["Abs_Error_SMA"] ** 2).mean())
    sma_wmape = (eval_df["Abs_Error_SMA"].sum() / total_actual * 100) if total_actual > 0 else 0
    
    # Metrik khusus produk aktif (Aktual > 0)
    active_df = eval_df[eval_df["Aktual_Januari_2026"] > 0]
    dt_wmape_active = (active_df["Abs_Error_Decision_Tree"].sum() / active_df["Aktual_Januari_2026"].sum() * 100) if len(active_df) > 0 else 0
    sma_wmape_active = (active_df["Abs_Error_SMA"].sum() / active_df["Aktual_Januari_2026"].sum() * 100) if len(active_df) > 0 else 0
    
    return eval_df, {
        "Decision Tree": {"MAE": dt_mae, "RMSE": dt_rmse, "WMAPE": dt_wmape, "WMAPE_Active": dt_wmape_active, "Bounded_MAPE": eval_df["Bounded_Error_Decision_Tree"].mean()},
        "SMA": {"MAE": sma_mae, "RMSE": sma_rmse, "WMAPE": sma_wmape, "WMAPE_Active": sma_wmape_active, "Bounded_MAPE": eval_df["Bounded_Error_SMA"].mean()},
        "Total_Aktual": total_actual,
        "Total_Pred_DT": eval_df["Prediksi_Decision_Tree"].sum(),
        "Total_Pred_SMA": eval_df["Prediksi_SMA"].sum()
    }

def evaluate_six_months_average(conv_2026, pred_df):
    # Hitung total dan rata-rata bulanan aktual selama Jan - Juni 2026
    six_months = conv_2026.groupby(
        ["Kode Item", "Satuan Standar"], as_index=False
    )["Jumlah Standar"].sum()
    six_months["Rata2_Bulan_Jan_Juni_2026"] = six_months["Jumlah Standar"] / 6.0
    six_months.rename(columns={"Jumlah Standar": "Total_Jan_Juni_2026"}, inplace=True)
    
    eval_df = pred_df.merge(six_months, on=["Kode Item", "Satuan Standar"], how="left")
    eval_df["Total_Jan_Juni_2026"] = eval_df["Total_Jan_Juni_2026"].fillna(0).astype(int)
    eval_df["Rata2_Bulan_Jan_Juni_2026"] = eval_df["Rata2_Bulan_Jan_Juni_2026"].fillna(0)
    
    eval_df["Selisih_Decision_Tree"] = eval_df["Prediksi_Decision_Tree"] - eval_df["Rata2_Bulan_Jan_Juni_2026"]
    eval_df["Selisih_SMA"] = eval_df["Prediksi_SMA"] - eval_df["Rata2_Bulan_Jan_Juni_2026"]
    eval_df["Abs_Error_Decision_Tree"] = eval_df["Selisih_Decision_Tree"].abs()
    eval_df["Abs_Error_SMA"] = eval_df["Selisih_SMA"].abs()
    
    eval_df["Bounded_Error_Decision_Tree"] = eval_df.apply(lambda r: bounded_percentage_error(r["Rata2_Bulan_Jan_Juni_2026"], r["Prediksi_Decision_Tree"]), axis=1)
    eval_df["Bounded_Error_SMA"] = eval_df.apply(lambda r: bounded_percentage_error(r["Rata2_Bulan_Jan_Juni_2026"], r["Prediksi_SMA"]), axis=1)
    
    out_path = OUTPUT_DIR / "pembuktian_6_bulan_2026.csv"
    eval_df.sort_values("Rata2_Bulan_Jan_Juni_2026", ascending=False).to_csv(out_path, index=False)
    
    total_actual_avg = eval_df["Rata2_Bulan_Jan_Juni_2026"].sum()
    dt_mae = eval_df["Abs_Error_Decision_Tree"].mean()
    dt_rmse = np.sqrt((eval_df["Abs_Error_Decision_Tree"] ** 2).mean())
    dt_wmape = (eval_df["Abs_Error_Decision_Tree"].sum() / total_actual_avg * 100) if total_actual_avg > 0 else 0
    
    sma_mae = eval_df["Abs_Error_SMA"].mean()
    sma_rmse = np.sqrt((eval_df["Abs_Error_SMA"] ** 2).mean())
    sma_wmape = (eval_df["Abs_Error_SMA"].sum() / total_actual_avg * 100) if total_actual_avg > 0 else 0
    
    # Metrik khusus produk aktif (Aktual > 0)
    active_df = eval_df[eval_df["Rata2_Bulan_Jan_Juni_2026"] > 0]
    dt_wmape_active = (active_df["Abs_Error_Decision_Tree"].sum() / active_df["Rata2_Bulan_Jan_Juni_2026"].sum() * 100) if len(active_df) > 0 else 0
    sma_wmape_active = (active_df["Abs_Error_SMA"].sum() / active_df["Rata2_Bulan_Jan_Juni_2026"].sum() * 100) if len(active_df) > 0 else 0
    
    return eval_df, {
        "Decision Tree": {"MAE": dt_mae, "RMSE": dt_rmse, "WMAPE": dt_wmape, "WMAPE_Active": dt_wmape_active, "Bounded_MAPE": eval_df["Bounded_Error_Decision_Tree"].mean()},
        "SMA": {"MAE": sma_mae, "RMSE": sma_rmse, "WMAPE": sma_wmape, "WMAPE_Active": sma_wmape_active, "Bounded_MAPE": eval_df["Bounded_Error_SMA"].mean()},
        "Total_Aktual_Avg": total_actual_avg,
        "Total_Pred_DT": eval_df["Prediksi_Decision_Tree"].sum(),
        "Total_Pred_SMA": eval_df["Prediksi_SMA"].sum()
    }

def main():
    print("=== MEMULAI PEMBUKTIAN PREDIKSI MACHINE LEARNING (DATA 2026) ===")
    pred_path = OUTPUT_DIR / "hasil_prediksi_bulan_depan.csv"
    if not pred_path.exists():
        raise FileNotFoundError("File output/hasil_prediksi_bulan_depan.csv tidak ditemukan.")
    
    pred_df = pd.read_csv(pred_path)
    conv_2026 = load_actual_2026()
    
    # 1. Evaluasi Januari 2026
    print("\n[A] Evaluasi Prediksi vs Aktual Bulan Januari 2026...")
    eval_jan, metrics_jan = evaluate_january_2026(conv_2026, pred_df)
    
    print(f"Total Penjualan Aktual Jan 2026 : {metrics_jan['Total_Aktual']:,} unit")
    print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'WMAPE (All)':<12} | {'WMAPE (Aktif)':<14} | {'Bounded Err (Max 100%)':<22}")
    print("-" * 94)
    for m_name in ["Decision Tree", "SMA"]:
        m_val = metrics_jan[m_name]
        print(f"{m_name:<20} | {m_val['MAE']:<8.2f} | {m_val['RMSE']:<8.2f} | {m_val['WMAPE']:<12.2f}% | {m_val['WMAPE_Active']:<14.2f}% | {m_val['Bounded_MAPE']:<22.2f}%")
        
    # 2. Evaluasi 6 Bulan (Jan - Juni 2026 Average)
    print("\n[B] Evaluasi Prediksi vs Rata-rata Bulanan Aktual (Januari – Juni 2026)...")
    eval_6m, metrics_6m = evaluate_six_months_average(conv_2026, pred_df)
    
    print(f"Total Rata-rata Aktual / Bulan  : {metrics_6m['Total_Aktual_Avg']:,.1f} unit/bulan")
    print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'WMAPE (All)':<12} | {'WMAPE (Aktif)':<14} | {'Bounded Err (Max 100%)':<22}")
    print("-" * 94)
    for m_name in ["Decision Tree", "SMA"]:
        m_val = metrics_6m[m_name]
        print(f"{m_name:<20} | {m_val['MAE']:<8.2f} | {m_val['RMSE']:<8.2f} | {m_val['WMAPE']:<12.2f}% | {m_val['WMAPE_Active']:<14.2f}% | {m_val['Bounded_MAPE']:<22.2f}%")
        
    print("\n--- Top 10 Produk dengan Permintaan Aktual Tertinggi (Jan - Juni 2026) ---")
    top10_6m = eval_6m.sort_values("Rata2_Bulan_Jan_Juni_2026", ascending=False).head(10)[
        ["Kode Item", "Nama Item", "Satuan Standar", "Total_Jan_Juni_2026", "Rata2_Bulan_Jan_Juni_2026", "Prediksi_Decision_Tree", "Prediksi_SMA"]
    ]
    print(top10_6m.to_string(index=False))
    
    print("\n=== LAPORAN BERHASIL DISIMPAN ===")
    print("1. output/pembuktian_januari_2026.csv (Detail per produk untuk bulan Januari 2026)")
    print("2. output/pembuktian_6_bulan_2026.csv (Detail per produk untuk rata-rata 6 bulan 2026)")

if __name__ == "__main__":
    main()
