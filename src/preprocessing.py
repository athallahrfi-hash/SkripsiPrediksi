from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RAW_DATA_PATH = DATA_DIR / "Dataset_Lengkap_2023-2025.xlsx"
CONVERSION_PATH = DATA_DIR / "konversi_satuan.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


REQUIRED_COLUMNS = [
    "Tanggal",
    "Kode Item",
    "Nama Item",
    "Jumlah",
    "Satuan",
    "Total Item",
]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Excel dataset with clear handling for PermissionError.
    If the file is locked (e.g., open in Excel or syncing), this function will raise a
    descriptive error prompting the user to close the file or ensure it is not in use.
    """
    # Resolve fallback path if the expected file is not present
    if not file_path.exists():
        fallback_path = BASE_DIR / file_path.name
        if fallback_path.exists():
            file_path = fallback_path
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {file_path}")

    try:
        xl = pd.ExcelFile(file_path)
    except PermissionError as e:
        raise PermissionError(
            "Tidak dapat membaca file dataset karena PermissionError. "
            "Pastikan file tidak terbuka di aplikasi lain (misalnya Excel) atau sedang disinkronisasi oleh OneDrive, "
            "lalu coba lagi."
        ) from e

    sheet = "Item Detail" if "Item Detail" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di dataset: {missing_columns}"
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    df = df.copy()

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Kode Item"] = normalize_text(df["Kode Item"]).str.upper()
    df["Nama Item"] = normalize_text(df["Nama Item"])
    df["Satuan"] = normalize_text(df["Satuan"]).str.upper()

    df["Jumlah"] = pd.to_numeric(df["Jumlah"], errors="coerce").fillna(0)
    df["Total Item"] = pd.to_numeric(df["Total Item"], errors="coerce").fillna(0)

    df = df[df["Tanggal"].notna()]
    df = df[df["Kode Item"] != ""]
    df = df[df["Nama Item"] != ""]
    df = df[df["Jumlah"] > 0]

    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    df["Periode"] = df["Tanggal"].dt.to_period("M").astype(str)

    return df


def load_conversion(file_path: Path = CONVERSION_PATH) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(
            columns=[
                "Kode Item",
                "Nama Item",
                "Satuan Asal",
                "Satuan Standar",
                "Isi Per Satuan",
            ]
        )

    konversi = pd.read_csv(file_path)

    konversi["Kode Item"] = normalize_text(konversi["Kode Item"]).str.upper()
    konversi["Nama Item"] = normalize_text(konversi["Nama Item"])
    konversi["Satuan Asal"] = normalize_text(konversi["Satuan Asal"]).str.upper()
    konversi["Satuan Standar"] = normalize_text(konversi["Satuan Standar"]).str.upper()
    konversi["Isi Per Satuan"] = pd.to_numeric(
        konversi["Isi Per Satuan"],
        errors="coerce"
    ).fillna(1)

    return konversi


def apply_unit_conversion(
    df: pd.DataFrame,
    konversi: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()

    if not konversi.empty:
        konversi = konversi.drop_duplicates(
            subset=["Kode Item", "Satuan Asal"]
        )

    df_konversi = df.merge(
        konversi[
            [
                "Kode Item",
                "Satuan Asal",
                "Satuan Standar",
                "Isi Per Satuan",
            ]
        ],
        left_on=["Kode Item", "Satuan"],
        right_on=["Kode Item", "Satuan Asal"],
        how="left"
    )

    df_konversi["Isi Per Satuan"] = df_konversi["Isi Per Satuan"].fillna(1)

    df_konversi["Satuan Standar"] = df_konversi["Satuan Standar"].fillna(
        df_konversi["Satuan"]
    )

    df_konversi["Jumlah Standar"] = (
        df_konversi["Jumlah"] * df_konversi["Isi Per Satuan"]
    )

    return df_konversi

def create_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    monthly_sales = df.groupby(
        ["Periode", "Kode Item", "Nama Item", "Satuan Standar"],
        as_index=False
    ).agg({
        "Jumlah Standar": "sum",
        "Total Item": "sum",
    })

    monthly_sales["Tanggal Periode"] = pd.to_datetime(
        monthly_sales["Periode"] + "-01",
        errors="coerce"
    )

    min_period = monthly_sales["Tanggal Periode"].min()
    max_period = monthly_sales["Tanggal Periode"].max()

    all_periods = pd.date_range(
        start=min_period,
        end=max_period,
        freq="MS"
    )

    product_list = monthly_sales[
        ["Kode Item", "Nama Item", "Satuan Standar"]
    ].drop_duplicates()

    period_list = pd.DataFrame({
        "Tanggal Periode": all_periods
    })

    product_list["key"] = 1
    period_list["key"] = 1

    full_grid = product_list.merge(period_list, on="key").drop(columns=["key"])

    monthly_sales_full = full_grid.merge(
        monthly_sales.drop(columns=["Periode"]),
        on=["Kode Item", "Nama Item", "Satuan Standar", "Tanggal Periode"],
        how="left"
    )

    monthly_sales_full["Jumlah Standar"] = monthly_sales_full["Jumlah Standar"].fillna(0)
    monthly_sales_full["Total Item"] = monthly_sales_full["Total Item"].fillna(0)

    monthly_sales_full["Periode"] = (
        monthly_sales_full["Tanggal Periode"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_sales_full = monthly_sales_full[
        [
            "Periode",
            "Tanggal Periode",
            "Kode Item",
            "Nama Item",
            "Satuan Standar",
            "Jumlah Standar",
            "Total Item",
        ]
    ]

    monthly_sales_full = monthly_sales_full.sort_values(
        ["Kode Item", "Tanggal Periode"]
    ).reset_index(drop=True)

    return monthly_sales_full


def create_ml_dataset(monthly_sales: pd.DataFrame) -> pd.DataFrame:
    ml_data = monthly_sales.copy()

    if "Tanggal Periode" not in ml_data.columns:
        ml_data["Tanggal Periode"] = pd.to_datetime(
            ml_data["Periode"] + "-01",
            errors="coerce"
        )

    ml_data = ml_data.sort_values(
        ["Kode Item", "Tanggal Periode"]
    ).reset_index(drop=True)

    ml_data["Lag_1"] = ml_data.groupby("Kode Item")["Jumlah Standar"].shift(1)
    ml_data["Lag_2"] = ml_data.groupby("Kode Item")["Jumlah Standar"].shift(2)
    ml_data["Lag_3"] = ml_data.groupby("Kode Item")["Jumlah Standar"].shift(3)
    ml_data["Lag_6"] = ml_data.groupby("Kode Item")["Jumlah Standar"].shift(6).fillna(0)
    ml_data["Lag_12"] = ml_data.groupby("Kode Item")["Jumlah Standar"].shift(12).fillna(0)

    ml_data["Rolling_Mean_3"] = ml_data.groupby("Kode Item")["Jumlah Standar"].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )

    ml_data["Rolling_Max_3"] = ml_data.groupby("Kode Item")["Jumlah Standar"].transform(
        lambda x: x.shift(1).rolling(window=3).max()
    )

    ml_data["Rolling_Min_3"] = ml_data.groupby("Kode Item")["Jumlah Standar"].transform(
        lambda x: x.shift(1).rolling(window=3).min()
    )

    ml_data["Rolling_Mean_6"] = ml_data.groupby("Kode Item")["Jumlah Standar"].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    )

    ml_data["Rolling_Mean_12"] = ml_data.groupby("Kode Item")["Jumlah Standar"].transform(
        lambda x: x.shift(1).rolling(window=12, min_periods=1).mean()
    )

    ml_data["Tahun"] = ml_data["Tanggal Periode"].dt.year
    ml_data["Bulan"] = ml_data["Tanggal Periode"].dt.month

    ml_data = ml_data.dropna(
        subset=[
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Rolling_Mean_3",
            "Rolling_Max_3",
            "Rolling_Min_3",
        ]
    ).reset_index(drop=True)

    return ml_data


def create_unit_check_report(df: pd.DataFrame) -> pd.DataFrame:
    satuan_report = df.groupby(
        ["Kode Item", "Nama Item", "Satuan"],
        as_index=False
    ).agg({
        "Jumlah": "sum",
        "Total Item": "sum",
    })

    satuan_report = satuan_report.sort_values(
        ["Kode Item", "Nama Item", "Satuan"]
    )

    return satuan_report


def create_multi_unit_report(satuan_report: pd.DataFrame) -> pd.DataFrame:
    jumlah_satuan = satuan_report.groupby(
        ["Kode Item", "Nama Item"],
        as_index=False
    )["Satuan"].nunique()

    produk_multi_satuan = jumlah_satuan[jumlah_satuan["Satuan"] > 1]

    report = satuan_report.merge(
        produk_multi_satuan[["Kode Item", "Nama Item"]],
        on=["Kode Item", "Nama Item"],
        how="inner"
    )

    return report.sort_values(["Kode Item", "Nama Item", "Satuan"])


def main():
    print("Membaca dataset...")

    raw_data = load_data()
    clean = clean_data(raw_data)

    print("Membaca file konversi satuan...")

    konversi = load_conversion()
    converted = apply_unit_conversion(clean, konversi)

    print("Membuat data penjualan bulanan...")

    monthly_sales = create_monthly_sales(converted)

    print("Membuat data untuk machine learning...")

    ml_data = create_ml_dataset(monthly_sales)

    print("Membuat laporan pengecekan satuan...")

    satuan_report = create_unit_check_report(clean)
    multi_unit_report = create_multi_unit_report(satuan_report)

    clean.to_csv(OUTPUT_DIR / "data_bersih.csv", index=False)
    converted.to_csv(OUTPUT_DIR / "data_terkonversi.csv", index=False)
    monthly_sales.to_csv(OUTPUT_DIR / "penjualan_bulanan.csv", index=False)
    ml_data.to_csv(OUTPUT_DIR / "data_ml.csv", index=False)
    satuan_report.to_csv(OUTPUT_DIR / "cek_satuan_produk.csv", index=False)
    multi_unit_report.to_csv(OUTPUT_DIR / "produk_multi_satuan.csv", index=False)

    print("\n=== PREPROCESSING SELESAI ===")
    print(f"Data mentah               : {len(raw_data)} baris")
    print(f"Data bersih               : {len(clean)} baris")
    print(f"Data penjualan bulanan    : {len(monthly_sales)} baris")
    print(f"Data siap machine learning: {len(ml_data)} baris")
    print(f"Produk multi satuan       : {len(multi_unit_report)} baris")
    print("\nFile hasil disimpan di folder output/")


if __name__ == "__main__":
    main()
