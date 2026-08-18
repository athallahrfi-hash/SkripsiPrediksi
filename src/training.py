from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

DATA_ML_PATH = OUTPUT_DIR / "data_ml.csv"
MODEL_PATH = MODEL_DIR / "decision_tree_model.joblib"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


TARGET_COLUMN = "Jumlah Standar"

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

MIN_NON_ZERO_MONTHS = 3
MIN_TOTAL_DEMAND = 10


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        return 0

    return np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100


def calculate_mape_filtered(y_true, y_pred, min_actual=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true >= min_actual

    if mask.sum() == 0:
        return 0

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def calculate_wmape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    total_actual = np.sum(np.abs(y_true))

    if total_actual == 0:
        return 0

    return np.sum(np.abs(y_true - y_pred)) / total_actual * 100


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    mape_filtered = calculate_mape_filtered(y_true, y_pred, min_actual=10)
    wmape = calculate_wmape(y_true, y_pred)

    return mae, rmse, mape, mape_filtered, wmape


def load_ml_data():
    if not DATA_ML_PATH.exists():
        raise FileNotFoundError(
            "File output/data_ml.csv belum ditemukan. Jalankan dulu: python src/preprocessing.py"
        )

    df = pd.read_csv(DATA_ML_PATH)

    df["Tanggal Periode"] = pd.to_datetime(df["Tanggal Periode"], errors="coerce")
    df = df.dropna(subset=["Tanggal Periode"])

    df = df.sort_values(["Tanggal Periode", "Kode Item"]).reset_index(drop=True)

    return df


def create_product_activity_report(df):
    product_report = df.groupby(
        PRODUCT_KEY_COLUMNS,
        as_index=False
    ).agg(
        Jumlah_Bulan_Data=("Periode", "nunique"),
        Jumlah_Bulan_Terjual=(TARGET_COLUMN, lambda x: (x > 0).sum()),
        Total_Permintaan=(TARGET_COLUMN, "sum"),
        Rata_Rata_Permintaan=(TARGET_COLUMN, "mean"),
        Maksimum_Permintaan=(TARGET_COLUMN, "max"),
    )

    product_report["Status_Kelayakan"] = np.where(
        (product_report["Jumlah_Bulan_Terjual"] >= MIN_NON_ZERO_MONTHS)
        & (product_report["Total_Permintaan"] >= MIN_TOTAL_DEMAND),
        "Layak Diprediksi",
        "Data Terlalu Sedikit"
    )

    return product_report


def filter_predictable_products(df):
    product_report = create_product_activity_report(df)

    eligible_products = product_report[
        product_report["Status_Kelayakan"] == "Layak Diprediksi"
    ][PRODUCT_KEY_COLUMNS]

    filtered_df = df.merge(
        eligible_products,
        on=PRODUCT_KEY_COLUMNS,
        how="inner"
    )

    product_report.to_csv(
        OUTPUT_DIR / "laporan_kelayakan_produk.csv",
        index=False
    )

    eligible_products.to_csv(
        OUTPUT_DIR / "produk_layak_prediksi.csv",
        index=False
    )

    return filtered_df, product_report


def split_train_test_by_time(df, test_ratio=0.2):
    unique_periods = sorted(df["Tanggal Periode"].unique())

    total_periods = len(unique_periods)
    test_period_count = max(1, int(total_periods * test_ratio))

    test_periods = unique_periods[-test_period_count:]

    train_df = df[~df["Tanggal Periode"].isin(test_periods)].copy()
    test_df = df[df["Tanggal Periode"].isin(test_periods)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        split_index = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def build_decision_tree_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=4,
        min_samples_leaf=3
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():
    print("Membaca data machine learning...")

    df = load_ml_data()

    print("Memfilter produk yang layak diprediksi...")

    df_filtered, product_report = filter_predictable_products(df)

    print(f"Jumlah data sebelum filter : {len(df)} baris")
    print(f"Jumlah data setelah filter : {len(df_filtered)} baris")
    print(
        "Jumlah produk layak       : "
        f"{len(product_report[product_report['Status_Kelayakan'] == 'Layak Diprediksi'])}"
    )
    print(
        "Jumlah produk tidak layak : "
        f"{len(product_report[product_report['Status_Kelayakan'] == 'Data Terlalu Sedikit'])}"
    )

    df = df_filtered

    if len(df) == 0:
        raise ValueError(
            "Tidak ada produk yang memenuhi kriteria layak prediksi. "
            "Turunkan nilai MIN_NON_ZERO_MONTHS atau MIN_TOTAL_DEMAND."
        )

    print("Membagi data training dan testing berdasarkan waktu...")

    train_df, test_df = split_train_test_by_time(df)

    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    print("Melatih model Decision Tree Regression...")

    decision_tree_model = build_decision_tree_model()
    decision_tree_model.fit(X_train, y_train)

    print("Membuat prediksi...")

    pred_decision_tree = decision_tree_model.predict(X_test)

    # Baseline Single Moving Average menggunakan Rolling_Mean_3
    pred_sma = X_test["Rolling_Mean_3"].values

    mae_dt, rmse_dt, mape_dt, mape_filtered_dt, wmape_dt = calculate_metrics(
        y_test,
        pred_decision_tree
    )

    mae_sma, rmse_sma, mape_sma, mape_filtered_sma, wmape_sma = calculate_metrics(
        y_test,
        pred_sma
    )

    hasil_evaluasi = pd.DataFrame([
        {
            "Model": "Single Moving Average",
            "MAE": mae_sma,
            "RMSE": rmse_sma,
            "MAPE": mape_sma,
            "MAPE_Aktual_Min_10": mape_filtered_sma,
            "WMAPE": wmape_sma,
        },
        {
            "Model": "Decision Tree Regression",
            "MAE": mae_dt,
            "RMSE": rmse_dt,
            "MAPE": mape_dt,
            "MAPE_Aktual_Min_10": mape_filtered_dt,
            "WMAPE": wmape_dt,
        },
    ])

    prediksi_vs_aktual = test_df[
        [
            "Periode",
            "Kode Item",
            "Nama Item",
            "Satuan Standar",
            TARGET_COLUMN,
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
        ]
    ].copy()

    prediksi_vs_aktual = prediksi_vs_aktual.rename(
        columns={
            TARGET_COLUMN: "Aktual"
        }
    )

    prediksi_vs_aktual["Prediksi_SMA"] = pred_sma
    prediksi_vs_aktual["Prediksi_Decision_Tree"] = pred_decision_tree

    prediksi_vs_aktual["Error_SMA"] = (
        prediksi_vs_aktual["Aktual"] - prediksi_vs_aktual["Prediksi_SMA"]
    ).abs()

    prediksi_vs_aktual["Error_Decision_Tree"] = (
        prediksi_vs_aktual["Aktual"] - prediksi_vs_aktual["Prediksi_Decision_Tree"]
    ).abs()

    prediksi_vs_aktual["Error_Persen_SMA"] = prediksi_vs_aktual.apply(
        lambda row: abs((row["Aktual"] - row["Prediksi_SMA"]) / row["Aktual"]) * 100
        if row["Aktual"] != 0 else np.nan,
        axis=1
    )

    prediksi_vs_aktual["Error_Persen_Decision_Tree"] = prediksi_vs_aktual.apply(
        lambda row: abs((row["Aktual"] - row["Prediksi_Decision_Tree"]) / row["Aktual"]) * 100
        if row["Aktual"] != 0 else np.nan,
        axis=1
    )

    hasil_evaluasi.to_csv(OUTPUT_DIR / "hasil_evaluasi.csv", index=False)
    prediksi_vs_aktual.to_csv(OUTPUT_DIR / "prediksi_vs_aktual.csv", index=False)

    joblib.dump(decision_tree_model, MODEL_PATH)

    print("\n=== TRAINING DAN EVALUASI SELESAI ===")
    print(f"Jumlah data ML       : {len(df)} baris")
    print(f"Data training        : {len(train_df)} baris")
    print(f"Data testing         : {len(test_df)} baris")

    print("\n=== HASIL EVALUASI ===")
    print(hasil_evaluasi)

    best_model = hasil_evaluasi.sort_values("WMAPE").iloc[0]["Model"]
    print(f"\nModel terbaik berdasarkan WMAPE: {best_model}")

    print("\nCatatan:")
    print("- MAPE biasa bisa sangat besar jika nilai aktual sangat kecil.")
    print("- Untuk laporan skripsi, gunakan MAE, RMSE, MAPE_Aktual_Min_10, dan WMAPE.")
    print("- WMAPE lebih stabil untuk dataset yang memiliki permintaan kecil atau tidak merata.")
    print("- Produk dengan histori terlalu sedikit dipisahkan dari evaluasi model.")

    print("\nFile hasil disimpan:")
    print("- output/hasil_evaluasi.csv")
    print("- output/prediksi_vs_aktual.csv")
    print("- output/laporan_kelayakan_produk.csv")
    print("- output/produk_layak_prediksi.csv")
    print("- model/decision_tree_model.joblib")


if __name__ == "__main__":
    main()
