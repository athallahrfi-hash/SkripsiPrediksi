import subprocess
import sys


commands = [
    ["python", "src/preprocessing.py"],
    ["python", "src/training.py"],
    ["python", "src/evaluation.py"],
    ["python", "src/predict.py"],
    ["python", "src/pembuktian_2026.py"],
]


def run_command(command):
    print("\nMenjalankan:", " ".join(command))
    result = subprocess.run(command)

    if result.returncode != 0:
        print("Proses gagal:", " ".join(command))
        sys.exit(result.returncode)


def main():
    print("=== MENJALANKAN PIPELINE PREDIKSI ===")

    for command in commands:
        run_command(command)

    print("\n=== PIPELINE SELESAI ===")
    print("Semua file output berhasil diperbarui.")
    print("Untuk membuka aplikasi, jalankan:")
    print("streamlit run app/app.py")


if __name__ == "__main__":
    main()
