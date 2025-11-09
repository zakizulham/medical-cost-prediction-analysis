# Analisis Prediksi Biaya Medis (Medical Cost Prediction Analysis)

[![Status](https://img.shields.io/badge/Status-Finished-brightgreen.svg)](https://github.com/zakizulham/medical-cost-prediction-analysis/graphs/commit-activity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Used-red.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Used-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Used-darkgreen.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Used-900c3f.svg)](https://shap.readthedocs.io/en/latest/)

Repositori ini mendokumentasikan analisis komprehensif terhadap dataset "Medical Cost Personal Datasets" dari Kaggle. Tujuan dari proyek ini adalah untuk menerapkan metodologi CRISP-ML(Q) terstruktur untuk mengeksplorasi, memodelkan, dan mengevaluasi berbagai pendekatan machine learning dari perspektif aktuaria dan ilmu data.

Fokus utamanya adalah pada pemodelan prediktif untuk biaya asuransi (`charges`) dan interpretasi dari faktor-faktor pendorong biaya tersebut.

## Metodologi

Proyek ini secara ketat mengikuti alur kerja **CRISP-ML(Q) (Cross-Industry Standard Process for Machine Learning with Quality Assurance)**. Analisis dibagi menjadi beberapa fase yang logis, di mana setiap *notebook* mewakili satu atau lebih dari fase-fase tersebut.

## Dataset

* **Sumber:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
* **Target Utama:** `charges` (Biaya medis individu yang ditagih oleh asuransi kesehatan).
* **Fitur:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`.

## Struktur Repositori

Struktur direktori dirancang untuk mencerminkan alur kerja CRISP-ML(Q) dan memisahkan kode sumber, data, dan artefak model.

```
medical-cost-prediction-analysis/
├── .gitignore
├── LICENSE
├── README.md
│
├── data/
│   ├── raw/            <-- (Data asli tidak dimodifikasi, diabaikan oleh Git)
│   └── prepared/       <-- (Data bersih hasil Notebook 01, diabaikan oleh Git)
│
├── models/             <-- (Model yang dilatih, diabaikan oleh Git)
│
├── notebooks/
│   ├── 00_Business_Data_Understanding.ipynb
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Modeling_Cost_Regression_XAI.ipynb
│   ├── 03_Modeling_High_Risk_Classification.ipynb
│   └── 04_Modeling_Risk_Pool_Clustering.ipynb
│
└── requirements.txt
```

## Alur Kerja Analisis (Per Notebook)

Setiap *notebook* dibangun di atas *notebook* sebelumnya, mengikuti alur kerja yang logis.

### 00_Business_Data_Understanding.ipynb
* **Fase CRISP:** 1. Business Understanding, 2. Data Understanding.
* **Tujuan:** Mendefinisikan pertanyaan bisnis (memprediksi biaya, mengidentifikasi risiko, menemukan segmen) dan melakukan Exploratory Data Analysis (EDA) lengkap pada data mentah. Mengidentifikasi `smoker` sebagai prediktor utama dan `charges` yang sangat miring (*skewed*).

### 01_Data_Preparation.ipynb
* **Fase CRISP:** 3. Data Preparation.
* **Tujuan:** Mem-prototipe dan memvalidasi "resep" *preprocessing* (One-Hot Encoding, Scaling). *Notebook* ini berfungsi sebagai laboratorium untuk merancang `ColumnTransformer` yang akan digunakan dalam *pipeline* pemodelan.
* **Output:** `data/prepared/insurance_features.csv` (Hanya untuk inspeksi manual, tidak digunakan dalam pemodelan).

### 02_Modeling_Cost_Regression_XAI.ipynb
* **Fase CRISP:** 4. Modeling, 5. Evaluation, 6. Deployment (Interpretasi).
* **Tujuan:** Memprediksi `charges` (Regresi).
* **Model:** Membandingkan Generalized Linear Model (GLM) dengan XGBoost Regressor.
* **Temuan:** XGBoost (R² 0.85) **secara signifikan mengungguli** GLM (R² 0.26), membuktikan bahwa model non-linear diperlukan untuk menangkap interaksi kompleks (misal: `smoker` + `bmi`).
* **Interpretasi:** Menerapkan **SHAP** (menggunakan `KernelExplainer` setelah *debugging* `TreeExplainer`) untuk mengonfirmasi bahwa `smoker_yes`, `age`, dan `bmi` adalah pendorong biaya utama.
* **Output:** `models/cost_regressor.joblib`.

### 03_Modeling_High_Risk_Classification.ipynb
* **Fase CRISP:** 4. Modeling, 5. Evaluation.
* **Tujuan:** Mengubah masalah menjadi Klasifikasi Biner (memprediksi nasabah `is_high_cost`).
* **Model:** Logistic Regression vs. Random Forest Classifier.
* **Temuan:** Kedua model (Linear dan Non-Linear) menghasilkan performa yang **identik** (ROC-AUC 0.8433). Analisis *Feature Importance* mengonfirmasi ini disebabkan oleh satu fitur biner (`smoker_yes`) yang mendominasi 54% dari keputusan model.
* **Output:** `models/high_risk_classifier.joblib`.

### 04_Modeling_Risk_Pool_Clustering.ipynb
* **Fase CRISP:** 4. Modeling (Unsupervised), 5. Evaluation.
* **Tujuan:** Mengidentifikasi segmen nasabah (risk pools) alami tanpa menggunakan data `charges`.
* **Temuan:** Menunjukkan **kegagalan** K-Means standar (yang hanya mengelompokkan berdasarkan `age`). Kemudian mendemonstrasikan **keberhasilan** pendekatan kedua dengan *feature weighting* (memberi bobot 10x pada `smoker`) yang berhasil memisahkan populasi menjadi 4 *risk pools* yang jelas (2 cluster Perokok, 2 cluster Non-Perokok), yang divalidasi dengan `charges` asli.

## Replikasi

Untuk mereplikasi analisis ini secara lokal:

1.  Clone repositori ini: `git clone https://github.com/zakizulham/medical-cost-prediction-analysis.git`
2.  Buat dan aktifkan *virtual environment*: `python -m venv venv && source venv/bin/activate`
3.  Install dependensi: `pip install -r requirements.txt`
4.  Unduh dataset dari [link Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance/data) dan letakkan `insurance.csv` di dalam folder `data/raw/`.
5.  Jalankan *notebook* secara berurutan (00 sampai 04).

---

Untuk folder data dan models yang diabaikan di git. **Sangat REKOMENDASIKAN** untuk jalankan ulang notebooks secara berurutan. Namun, jika sewaktu-waktu data menghilang dan tidak memungkinkan untuk menghasilkan file `*.joblib`. Silahkan akses [Google Drive](https://drive.google.com/drive/folders/1IU6HBHLu3xDX4fWCFWZzzldRC4H5oEtm?usp=sharing) berikut ini untuk pembelajaran.

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
