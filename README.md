# 🥬 Smart Farming Pakcoy — Streamlit App

Adaptasi dari sistem *Smart Farming Bayam Brazil* (Kelompok 3), disesuaikan penuh dengan
struktur `Dataset_Pakcoy.xlsx` (kolom: Day, DAP, Time, Soil Moisture (%), Temperature (°C),
Soil Condition, Ground Truth Maturity Level (%), Ground Truth Criteria).

## Cara Menjalankan

1. Pastikan file `Dataset_Pakcoy.xlsx` berada satu folder dengan `app.py`
   (atau upload ulang lewat sidebar saat aplikasi berjalan).
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
4. Buka browser ke alamat yang muncul (biasanya `http://localhost:8501`).

## Struktur Halaman

- **🏠 Dashboard** — ringkasan KPI, distribusi kondisi tanah & tahap pertumbuhan.
- **📊 Eksplorasi Data** — filter interaktif, tabel berwarna, histogram & scatter plot.
- **🤖 Model AI (CNN)** — dua model klasifikasi:
  - *Kondisi Tanah* (Dry/Optimal/Wet) dari kelembapan & suhu.
  - *Tahap Pertumbuhan / Kesiapan Panen* dari DAP & tingkat kematangan
    (menggantikan model citra daun pada versi Bayam Brazil karena dataset ini
    berbasis sensor, bukan foto).
  - Termasuk kurva training, confusion matrix, dan form prediksi manual.
- **📡 Monitoring Realtime** — simulasi sensor IoT berjalan (LSCM/regresi linear + CNN +
  logika otomatisasi pompa air).
- **ℹ️ Tentang** — penjelasan adaptasi dataset & perubahan dari versi sebelumnya.

## Catatan Penting

- Kelas **Dry** dan **Wet** pada `Soil Condition` jumlahnya sangat sedikit dibanding
  **Optimal** (data tidak seimbang). Model sudah memakai *class weighting*, tapi
  performa pada kedua kelas minoritas tetap perlu dicermati — idealnya tambahkan
  lebih banyak data sensor riil untuk kondisi kering/basah.
- Model dilatih ulang dan disimpan di *session state* Streamlit (`st.cache_resource`),
  jadi hanya dilatih sekali per sesi kecuali dataset diganti.
- Untuk deployment (mis. Streamlit Community Cloud), pastikan `tensorflow` versi CPU
  tercantum di `requirements.txt` agar build tidak melebihi batas resource.
