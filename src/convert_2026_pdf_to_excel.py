import pdfplumber
import re
import pandas as pd
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def clean_nama_pelanggan(kp, raw, known_mapping):
    if kp in known_mapping:
        return known_mapping[kp]
    # Split by common address keywords
    parts = re.split(
        r'\s+(?:jl\.?|ji\.?|jln\.?|jalan|gedung|perumahan|menara|wisma|blk\.?|pagar|kompleks|desa|kelurahan|kecamatan|ruko|tower|sukamahi|rt\.?|rw\.?|kartika tower|dinas kesehatan jl)(?:\b|\s|$)',
        raw,
        flags=re.IGNORECASE
    )
    cleaned = parts[0].strip().rstrip(',')
    return cleaned if cleaned else raw.strip()

def parse_pdf(file_path, known_mapping):
    pdf = pdfplumber.open(file_path)
    records = []
    current_trx = None
    current_items = []
    current_item = None

    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if (line.startswith('LAPORAN PENJUALAN DETAIL') or 
                line.startswith('PT. NUSA PHARMA') or 
                line.startswith('JL. Koja') or 
                line.startswith('Kota Bekasi') or 
                line.startswith('021-82757730') or 
                line.startswith('3008') or 
                line.startswith('IDAK') or
                line.startswith('CDOB') or
                line.startswith('022000047') or
                line.startswith('No Transaksi Tanggal') or 
                line.startswith('No. Kd. Item Nama Item') or 
                re.match(r'^\d{2}/\d{2}/\d{4}\s+\d{2}[.:]\d{2}\s+ADMIN', line)):
                continue

            # Check transaction header
            m_trx = re.match(r'^(\d{4}/JL/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(PL-\d+)\s+(.*)', line)
            if m_trx:
                kp = m_trx.group(3)
                nama = clean_nama_pelanggan(kp, m_trx.group(4), known_mapping)
                current_trx = {
                    'No Transaksi': m_trx.group(1),
                    'Tanggal': pd.to_datetime(m_trx.group(2), format='%d/%m/%Y'),
                    'Kode Pel.': kp,
                    'Nama Pelanggan': nama
                }
                current_items = []
                current_item = None
                continue

            # Check footer
            m_foot = re.match(r'^Pot\.\s*:\s*([\d\.,]+)\s+Pajak\s*:\s*([\d\.,]+)\s+Biaya\s*:\s*([\d\.,]+)\s+Total Akhir\s*:\s*([\d\.,]+)', line)
            if m_foot and current_trx:
                pajak = float(m_foot.group(2).replace('.', '').replace(',', '.'))
                tot_akhir = float(m_foot.group(4).replace('.', '').replace(',', '.'))
                subtotal = sum(it['Total Item'] for it in current_items)
                for it in current_items:
                    it['Subtotal'] = subtotal
                    it['Pajak'] = pajak
                    it['Total Akhir'] = tot_akhir
                    records.append(it)
                current_trx = None
                current_items = []
                current_item = None
                continue

            # Check item start with [a-zA-Z\-]+ for Satuan
            m_item = re.match(r'^(\d+)\s+([^\s]+)\s+(.+)\s+([\d\.,]+)([a-zA-Z\-]+)\s+([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)$', line)
            if m_item and current_trx:
                jml = float(m_item.group(4).replace('.', '').replace(',', '.'))
                hrg = float(m_item.group(6).replace('.', '').replace(',', '.'))
                pot = float(m_item.group(7).replace('.', '').replace(',', '.'))
                tot = float(m_item.group(8).replace('.', '').replace(',', '.'))
                current_item = {
                    'No Transaksi': current_trx['No Transaksi'],
                    'Tanggal': current_trx['Tanggal'],
                    'Kode Pel.': current_trx['Kode Pel.'],
                    'Nama Pelanggan': current_trx['Nama Pelanggan'],
                    'Kode Item': m_item.group(2),
                    'Nama Item': m_item.group(3).strip(),
                    'Jumlah': int(jml) if jml.is_integer() else int(round(jml)),
                    'Satuan': m_item.group(5).upper(),
                    'Harga': hrg,
                    'Potongan %': pot,
                    'Total Item': tot
                }
                current_items.append(current_item)
                continue

            # Check subtotal line
            m_sub = re.match(r'^([\d\.,]+)\s+([\d\.,]+)$', line)
            if m_sub and current_trx and current_items:
                continue

            # Continuation line
            if current_item:
                current_item['Nama Item'] += ' ' + line.strip()

    return pd.DataFrame(records)

def main():
    template_path = BASE_DIR / 'data' / 'Dataset Rekap.xlsx'
    if not template_path.exists():
        template_path = BASE_DIR / 'Dataset Rekap.xlsx'
    
    df_rekap = pd.read_excel(template_path)
    known_mapping = dict(zip(df_rekap['Kode Pel.'], df_rekap['Nama Pelanggan']))

    pdf_path = BASE_DIR / 'PENJUALAN NPP JAN - JUNI 2026.pdf'
    print(f"Membaca {pdf_path.name}...")
    df_all = parse_pdf(pdf_path, known_mapping)
    
    n_trx = df_all['No Transaksi'].nunique() if not df_all.empty else 0
    print(f"  -> Periode Jan - Juni 2026: {len(df_all)} baris item, {n_trx} transaksi")

    # Ensure columns match Dataset Rekap.xlsx exactly
    cols = [
        'No Transaksi', 'Tanggal', 'Kode Pel.', 'Nama Pelanggan',
        'Kode Item', 'Nama Item', 'Jumlah', 'Satuan', 'Harga',
        'Potongan %', 'Total Item', 'Subtotal', 'Pajak', 'Total Akhir'
    ]
    df_all = df_all[cols]

    # Sort by Tanggal and No Transaksi
    df_all = df_all.sort_values(['Tanggal', 'No Transaksi', 'Kode Item']).reset_index(drop=True)

    out_xlsx = BASE_DIR / 'dataset 2026 Jan-Juni.xlsx'
    out_alt = BASE_DIR / 'PENJUALAN NPP JAN - JUNI 2026.xlsx'

    print(f"\nMenyimpan ke {out_xlsx.name} dan {out_alt.name}...")
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='Item Detail', index=False)

    shutil.copy(out_xlsx, out_alt)

    print(f"Selesai! Berhasil membuat dataset 2026 dengan total {len(df_all)} baris.")

if __name__ == '__main__':
    main()
