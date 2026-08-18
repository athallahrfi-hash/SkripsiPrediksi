from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

PREDICTION_PATH = OUTPUT_DIR / "prediksi_vs_aktual.csv"


def calculate_percentage_error(actual, prediction):
    if actual == 0:
        return np.nan

    return abs((actual - prediction) / actual) * 100


def load_prediction_result():
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            "File output/prediksi_vs_aktual.csv belum ditemukan. Jalankan dulu: python src/training.py"
        )

    df = pd.read_csv(PREDICTION_PATH)

    return df


def create_error_analysis(df):
    df = df.copy()

    df["Error_Persen_SMA"] = df.apply(
        lambda row: calculate_percentage_error(
            row["Aktual"],
            row["Prediksi_SMA"]
        ),
        axis=1
    )

    df["Error_Persen_Decision_Tree"] = df.apply(
        lambda row: calculate_percentage_error(
            row["Aktual"],
            row["Prediksi_Decision_Tree"]
        ),
        axis=1
    )

    df["Selisih_SMA"] = df["Prediksi_SMA"] - df["Aktual"]
    df["Selisih_Decision_Tree"] = df["Prediksi_Decision_Tree"] - df["Aktual"]

    return df


def create_top_error_report(df):
    top_error_sma = df.sort_values(
        "Error_Persen_SMA",
        ascending=False
    ).head(10)

    top_error_dt = df.sort_values(
        "Error_Persen_Decision_Tree",
        ascending=False
    ).head(10)

    return top_error_sma, top_error_dt


def create_summary_report(df):
    summary = pd.DataFrame([
        {
            "Model": "Single Moving Average",
            "Rata-rata Aktual": df["Aktual"].mean(),
            "Rata-rata Prediksi": df["Prediksi_SMA"].mean(),
            "Rata-rata Error Absolut": df["Error_SMA"].mean(),
            "Rata-rata Error Persen": df["Error_Persen_SMA"].mean(),
            "Median Error Persen": df["Error_Persen_SMA"].median(),
            "Error Persen Tertinggi": df["Error_Persen_SMA"].max(),
        },
        {
            "Model": "Decision Tree Regression",
            "Rata-rata Aktual": df["Aktual"].mean(),
            "Rata-rata Prediksi": df["Prediksi_Decision_Tree"].mean(),
            "Rata-rata Error Absolut": df["Error_Decision_Tree"].mean(),
            "Rata-rata Error Persen": df["Error_Persen_Decision_Tree"].mean(),
            "Median Error Persen": df["Error_Persen_Decision_Tree"].median(),
            "Error Persen Tertinggi": df["Error_Persen_Decision_Tree"].max(),
        }
    ])

    return summary


def create_product_error_report(df):
    product_report = df.groupby(
        ["Kode Item", "Nama Item", "Satuan Standar"],
        as_index=False
    ).agg({
        "Aktual": "sum",
        "Prediksi_SMA": "sum",
        "Prediksi_Decision_Tree": "sum",
        "Error_SMA": "mean",
        "Error_Decision_Tree": "mean",
        "Error_Persen_SMA": "mean",
        "Error_Persen_Decision_Tree": "mean",
    })

    product_report = product_report.sort_values(
        "Error_Persen_SMA",
        ascending=False
    )

    return product_report


def main():
    print("Membaca hasil prediksi vs aktual...")

    df = load_prediction_result()

    print("Membuat analisis error...")

    error_analysis = create_error_analysis(df)

    top_error_sma, top_error_dt = create_top_error_report(error_analysis)
    summary_report = create_summary_report(error_analysis)
    product_error_report = create_product_error_report(error_analysis)

    error_analysis.to_csv(
        OUTPUT_DIR / "analisis_error_detail.csv",
        index=False
    )

    top_error_sma.to_csv(
        OUTPUT_DIR / "top_error_sma.csv",
        index=False
    )

    top_error_dt.to_csv(
        OUTPUT_DIR / "top_error_decision_tree.csv",
        index=False
    )

    summary_report.to_csv(
        OUTPUT_DIR / "ringkasan_error_model.csv",
        index=False
    )

    product_error_report.to_csv(
        OUTPUT_DIR / "error_per_produk.csv",
        index=False
    )

    print("\n=== ANALISIS ERROR SELESAI ===")

    print("\n=== RINGKASAN ERROR MODEL ===")
    print(summary_report)

    print("\n=== TOP 5 ERROR SINGLE MOVING AVERAGE ===")
    print(
        top_error_sma[
            [
                "Periode",
                "Kode Item",
                "Nama Item",
                "Aktual",
                "Prediksi_SMA",
                "Error_Persen_SMA",
            ]
        ].head()
    )

    print("\n=== TOP 5 ERROR DECISION TREE ===")
    print(
        top_error_dt[
            [
                "Periode",
                "Kode Item",
                "Nama Item",
                "Aktual",
                "Prediksi_Decision_Tree",
                "Error_Persen_Decision_Tree",
            ]
        ].head()
    )

    print("\nFile hasil disimpan:")
    print("- output/analisis_error_detail.csv")
    print("- output/top_error_sma.csv")
    print("- output/top_error_decision_tree.csv")
    print("- output/ringkasan_error_model.csv")
    print("- output/error_per_produk.csv")


if __name__ == "__main__":
    main()
