🏦 Credit Risk Classification & Default Prediction

Proyek Machine Learning berbasis Python ini bertujuan untuk melakukan penilaian risiko kredit (Credit Risk Assessment) dan memprediksi status kemungkinan gagal bayar (default) nasabah lembaga keuangan/bank berdasarkan kriteria demografi serta profil finansial.

🚀 Fitur & Komponen Utama

Aturan Risiko Kredit Sintetis (Feature Engineering):

Menghitung skor risiko berdasarkan kriteria rasio cicilan (Debt Service Ratio), total beban bulanan vs pendapatan, jumlah tanggungan, dan durasi kerja.

Preprocessing Pipeline:

StandardScaler: Mensejajarkan skala variabel numerik (penghasilan, pengeluaran, pinjaman, rasio cicilan).

OneHotEncoder: Mengonversi variabel kategorikal (Pendidikan, Jenis Pekerjaan) menjadi format vektor numerik.

Machine Learning Model:

Menggunakan algoritma Random Forest Classifier (n_estimators=100, max_depth=10) untuk akurasi klasifikasi tinggi.

Visualisasi Data Interaktif:

Grafik 1: Distribusi status gagal bayar nasabah (0: Lancar, 1: Berisiko).

Grafik 2: Heatmap Confusion Matrix untuk mengukur performa prediksi model.

Grafik 3: Feature Importance Chart untuk melihat faktor/kolom yang paling berpengaruh terhadap prediksi.

Simulasi Prediksi Nasabah Baru:

Memprediksi status kelayakan kredit serta menghitung persentase probabilitas risiko nasabah baru secara otomatis.

📁 Struktur Repositori

Credit Risk Classification/
├── data nasabah.xlsx             # Dataset utama profil finansial nasabah
├── Classification code.py        # Skrip Python utama (Model, Evaluasi, & Visualisasi)
├── notebook/
│   └── modeling_classification.ipynb  # File Jupyter Notebook interaktif
├── requirements.txt              # Daftar dependensi library Python
└── README.md                     # Dokumentasi proyek


🛠️ Prasyarat & Instalasi

1. Library yang Dibutuhkan

Pastikan Python 3.x sudah terpasang. Library utama yang digunakan:

pandas & openpyxl (Pengolahan data Excel)

numpy (Komputasi numerik)

scikit-learn (Pemodelan & Preprocessing Machine Learning)

matplotlib & seaborn (Visualisasi Grafik)

2. Cara Menginstal Dependensi

Buka Terminal / Command Prompt di VS Code, lalu jalankan perintah:

pip install -r requirements.txt


💻 Cara Menjalankan Program

Jalankan Skrip Python (.py)

python "Classification code.py"


Jalankan Lewat Jupyter Notebook (.ipynb)

Buka folder notebook/ di VS Code.

Buka file modeling_classification.ipynb.

Klik Run All atau jalankan cell satu per satu menggunakan tombol Shift + Enter.

📊 Contoh Output Evaluasi & Visualisasi

Akurasi Model: ~90% - 95%

Output Prediksi Simulasi:

=======================
PREDIKSI NASABAH BARU
=======================
Hasil Prediksi     : Gagal Bayar (Berisiko)
Probabilitas Risiko: 82.40%


👨‍💻 Penggunaan & Pengembangan

Proyek ini sangat cocok dijadikan sebagai acuan sistem Credit Scoring otomatis pada industri perbankan, financial technology (fintech), maupun koperasi simpan pinjam.
