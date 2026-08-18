from pathlib import Path
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

MONTHLY_SALES_PATH = OUTPUT_DIR / "penjualan_bulanan.csv"
ELIGIBLE_PRODUCTS_PATH = OUTPUT_DIR / "produk_layak_prediksi.csv"
EVALUATION_PATH = OUTPUT_DIR / "hasil_evaluasi.csv"
MODEL_PATH = MODEL_DIR / "decision_tree_model.joblib"

PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "hasil_prediksi_bulan_depan.csv"
PREDICTION_MODEL_PATH = MODEL_DIR / "hasil_prediksi.csv"


NUMERIC_FEATURES = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Rolling_Mean_3",
    "Rolling_Max_3",
    "Rolling_Min_3",
    "Rolling_Mean_6",
    "Rolling_Mean_12",
    "Tahun",
    "Bulan",
]

CATEGORICAL_FEATURES = [
    "Kode Item",
]

PRODUCT_KEY_COLUMNS = [
    "Kode Item",
    "Nama Item",
    "Satuan Standar",
]


def load_required_files():
    if not MONTHLY_SALES_PATH.exists():
        raise FileNotFoundError(
            "File output/penjualan_bulanan.csv belum ditemukan. "
            "Jalankan dulu: python src/preprocessing.py"
        )

    if not ELIGIBLE_PRODUCTS_PATH.exists():
        raise FileNotFoundError(
            "File output/produk_layak_prediksi.csv belum ditemukan. "
            "Jalankan dulu: python src/training.py"
        )

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            "File output/hasil_evaluasi.csv belum ditemukan. "
            "Jalankan dulu: python src/training.py"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "File model/decision_tree_model.joblib belum ditemukan. "
            "Jalankan dulu: python src/training.py"
        )

    monthly_sales = pd.read_csv(MONTHLY_SALES_PATH)
    eligible_products = pd.read_csv(ELIGIBLE_PRODUCTS_PATH)
    evaluation = pd.read_csv(EVALUATION_PATH)
    model = joblib.load(MODEL_PATH)

    return monthly_sales, eligible_products, evaluation, model


def prepare_monthly_sales(monthly_sales):
    monthly_sales = monthly_sales.copy()

    monthly_sales["Tanggal Periode"] = pd.to_datetime(
        monthly_sales["Tanggal Periode"],
        errors="coerce"
    )

    monthly_sales["Kode Item"] = monthly_sales["Kode Item"].astype(str).str.strip().str.upper()
    monthly_sales["Nama Item"] = monthly_sales["Nama Item"].astype(str).str.strip()
    monthly_sales["Satuan Standar"] = monthly_sales["Satuan Standar"].astype(str).str.strip().str.upper()

    monthly_sales["Jumlah Standar"] = pd.to_numeric(
        monthly_sales["Jumlah Standar"],
        errors="coerce"
    ).fillna(0)

    monthly_sales = monthly_sales.dropna(subset=["Tanggal Periode"])

    monthly_sales = monthly_sales.sort_values(
        ["Kode Item", "Tanggal Periode"]
    ).reset_index(drop=True)

    return monthly_sales


def prepare_eligible_products(eligible_products):
    eligible_products = eligible_products.copy()

    eligible_products["Kode Item"] = (
        eligible_products["Kode Item"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    eligible_products["Nama Item"] = (
        eligible_products["Nama Item"]
        .astype(str)
        .str.strip()
    )

    eligible_products["Satuan Standar"] = (
        eligible_products["Satuan Standar"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return eligible_products


def get_best_model_name(evaluation):
    best_model = evaluation.sort_values("WMAPE").iloc[0]["Model"]
    return best_model


def create_next_month_features(monthly_sales, eligible_products):
    latest_period = monthly_sales["Tanggal Periode"].max()
    next_period = latest_period + pd.DateOffset(months=1)

    prediction_rows = []

    monthly_sales = monthly_sales.merge(
        eligible_products[PRODUCT_KEY_COLUMNS],
        on=PRODUCT_KEY_COLUMNS,
        how="inner"
    )

    for product_key, group in monthly_sales.groupby(PRODUCT_KEY_COLUMNS):
        kode_item, nama_item, satuan_standar = product_key

        group = group.sort_values("Tanggal Periode")

        if len(group) < 3:
            continue

        last_3_months = group.tail(3)
        last_6_months = group.tail(6)
        last_12_months = group.tail(12)

        lag_1 = group.iloc[-1]["Jumlah Standar"]
        lag_2 = group.iloc[-2]["Jumlah Standar"]
        lag_3 = group.iloc[-3]["Jumlah Standar"]
        lag_6 = group.iloc[-6]["Jumlah Standar"] if len(group) >= 6 else 0
        lag_12 = group.iloc[-12]["Jumlah Standar"] if len(group) >= 12 else 0

        rolling_mean_3 = last_3_months["Jumlah Standar"].mean()
        rolling_max_3 = last_3_months["Jumlah Standar"].max()
        rolling_min_3 = last_3_months["Jumlah Standar"].min()
        rolling_mean_6 = last_6_months["Jumlah Standar"].mean()
        rolling_mean_12 = last_12_months["Jumlah Standar"].mean()

        prediction_rows.append({
            "Periode Prediksi": next_period.strftime("%Y-%m"),
            "Kode Item": kode_item,
            "Nama Item": nama_item,
            "Satuan Standar": satuan_standar,
            "Lag_1": lag_1,
            "Lag_2": lag_2,
            "Lag_3": lag_3,
            "Lag_6": lag_6,
            "Lag_12": lag_12,
            "Rolling_Mean_3": rolling_mean_3,
            "Rolling_Max_3": rolling_max_3,
            "Rolling_Min_3": rolling_min_3,
            "Rolling_Mean_6": rolling_mean_6,
            "Rolling_Mean_12": rolling_mean_12,
            "Tahun": next_period.year,
            "Bulan": next_period.month,
        })

    prediction_df = pd.DataFrame(prediction_rows)

    return prediction_df


def add_recommendation(df):
    df = df.copy()

    q50 = df["Prediksi_Model_Terbaik"].quantile(0.50)
    q75 = df["Prediksi_Model_Terbaik"].quantile(0.75)

    def priority_label(value):
        if value >= q75:
            return "Prioritas Tinggi"
        if value >= q50:
            return "Prioritas Sedang"
        if value > 0:
            return "Prioritas Rendah"
        return "Tidak Perlu Restock"

    df["Prioritas Restock"] = df["Prediksi_Model_Terbaik"].apply(priority_label)

    df["Keterangan"] = df["Prioritas Restock"].map({
        "Prioritas Tinggi": "Produk memiliki estimasi permintaan tinggi pada bulan prediksi.",
        "Prioritas Sedang": "Produk memiliki estimasi permintaan sedang pada bulan prediksi.",
        "Prioritas Rendah": "Produk memiliki estimasi permintaan rendah pada bulan prediksi.",
        "Tidak Perlu Restock": "Produk tidak menunjukkan estimasi permintaan pada bulan prediksi.",
    })

    return df


def main():
    print("Membaca file hasil preprocessing, training, dan model...")

    monthly_sales, eligible_products, evaluation, model = load_required_files()

    monthly_sales = prepare_monthly_sales(monthly_sales)
    eligible_products = prepare_eligible_products(eligible_products)

    print("Membuat fitur untuk prediksi bulan depan...")

    prediction_df = create_next_month_features(monthly_sales, eligible_products)

    if len(prediction_df) == 0:
        raise ValueError("Tidak ada data yang bisa diprediksi.")

    print("Membuat prediksi menggunakan Single Moving Average dan Decision Tree...")

    X_prediction = prediction_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    prediction_df["Prediksi_SMA"] = prediction_df["Rolling_Mean_3"]
    prediction_df["Prediksi_Decision_Tree"] = model.predict(X_prediction)

    prediction_df["Prediksi_SMA"] = prediction_df["Prediksi_SMA"].clip(lower=0)
    prediction_df["Prediksi_Decision_Tree"] = prediction_df["Prediksi_Decision_Tree"].clip(lower=0)

    best_model = get_best_model_name(evaluation)

    if best_model == "Decision Tree Regression":
        prediction_df["Model_Terbaik"] = "Decision Tree Regression"
        prediction_df["Prediksi_Model_Terbaik"] = prediction_df["Prediksi_Decision_Tree"]
    else:
        prediction_df["Model_Terbaik"] = "Single Moving Average"
        prediction_df["Prediksi_Model_Terbaik"] = prediction_df["Prediksi_SMA"]

    prediction_df["Prediksi_Model_Terbaik"] = np.ceil(
        prediction_df["Prediksi_Model_Terbaik"]
    ).astype(int)

    prediction_df["Prediksi_SMA"] = np.ceil(
        prediction_df["Prediksi_SMA"]
    ).astype(int)

    prediction_df["Prediksi_Decision_Tree"] = np.ceil(
        prediction_df["Prediksi_Decision_Tree"]
    ).astype(int)

    prediction_df = add_recommendation(prediction_df)

    prediction_df = prediction_df.sort_values(
        ["Prediksi_Model_Terbaik", "Kode Item"],
        ascending=[False, True],
        kind="stable"
    ).reset_index(drop=True)

    prediction_df.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    prediction_df.to_csv(PREDICTION_MODEL_PATH, index=False)

    print("\n=== PREDIKSI BULAN DEPAN SELESAI ===")
    print(f"Periode prediksi       : {prediction_df['Periode Prediksi'].iloc[0]}")
    print(f"Jumlah produk prediksi : {len(prediction_df)}")
    print(f"Model terbaik dipakai  : {best_model}")

    print("\n=== TOP 10 PREDIKSI PERMINTAAN TERTINGGI ===")
    print(
        prediction_df[
            [
                "Kode Item",
                "Nama Item",
                "Satuan Standar",
                "Prediksi_Model_Terbaik",
                "Prioritas Restock",
            ]
        ].head(10)
    )

    print("\nFile hasil disimpan:")
    print("- output/hasil_prediksi_bulan_depan.csv")
    print("- model/hasil_prediksi.csv")


if __name__ == "__main__":
    main()
