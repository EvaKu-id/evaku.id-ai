from collections.abc import Sequence

from evaku_ai.adapters.release import LATEST_KEY, ReleaseFile

BYTES_PER_MB = 1_000_000

FEATURE_NOTES = {
    "village_id": "Kode BPS 10 digit sebagai teks (contoh `1114050004`); baca sebagai string",
    "name": "Nama desa",
    "district_id": "Kode kecamatan",
    "area_km2": "Luas desa (UTM 47N)",
    "basin_share": "Porsi luas desa yang berada di daerah aliran Sungai Tamiang (0-1)",
    "distance_to_stream_m": (
        "Jarak titik tengah desa ke alur sungai terdekat di `geo/streams.parquet`; "
        "alur 5 km2 termasuk anak sungai, bukan hanya sungai utama"
    ),
    "population": "Penduduk, WorldPop 2026 (pecahan)",
    "children_under5": "Balita 0-4 tahun",
    "elderly": "Lansia 60 tahun ke atas",
    "hand_median": "Median HAND warga desa dalam meter di atas alur sungai (berbobot penduduk)",
    "population_hand_le_1m": "Jumlah warga yang tinggal 1 m atau kurang di atas sungai",
    "population_hand_le_3m": "Jumlah warga yang tinggal 3 m atau kurang di atas sungai",
}


def render_readme(
    version: str,
    files: Sequence[ReleaseFile],
    licenses: Sequence[str],
    feature_columns: Sequence[str],
) -> str:
    rows = "\n".join(f"| `{f.path}` | {f.size / BYTES_PER_MB:.2f} |" for f in files)
    columns = "\n".join(f"| `{name}` | {FEATURE_NOTES.get(name, '')} |" for name in feature_columns)
    credits = "\n".join(f"- {text}" for text in licenses)
    return f"""# Data statis EvaKu {version}

Rilis data statis Kabupaten Aceh Tamiang untuk pipeline, API, ML, dan web. Rilis ini tidak pernah
ditimpa; versi baru selalu memakai nomor baru. Penunjuk versi terbaru ada di `{LATEST_KEY}`.

## Isi

| File | Ukuran (MB) |
| --- | --- |
{rows}

`release.json` (di folder yang sama) memuat ukuran dan SHA-256 semua file di atas. `manifest.json`
dan `scenario.npz` adalah bundle untuk API model (larik mesin akses dan koordinat ruas untuk
rute). Semua file di `geo/` memakai `village_id` kode BPS sebagai kunci desa. File bergeometri
berformat GeoParquet (EPSG:4326), kecuali `hand.tif` (HAND 30 m, EPSG:32647, nodata -9999).

## `geo/village_static_features.csv`

Fitur statis, satu baris per desa. Nilainya tidak berubah antarhari.

| Kolom | Arti |
| --- | --- |
{columns}

## Catatan

- HAND berasal dari DSM (Copernicus GLO-30): kanopi, atap, dan jembatan ikut terukur, dan sel HAND
  negatif dipotong ke 0, jadi desa di dekat sungai bisa terbaca lebih rendah dari kenyataan.
- Debit sungai tidak ada di rilis ini; hanya ada satu titik GloFAS untuk Sungai Tamiang.

## Lisensi dan atribusi (wajib saat dipakai atau dipublikasikan)

{credits}

## Membaca dari R2

```python
import json

import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)
latest = json.loads(s3.get_object(Bucket="evaku-private", Key="{LATEST_KEY}")["Body"].read())
prefix = latest["prefix"]
body = s3.get_object(Bucket="evaku-private", Key=prefix + "release.json")["Body"]
release = json.loads(body.read())
```
"""
