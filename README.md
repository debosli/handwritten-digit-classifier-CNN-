# Handwritten Digit Classifier (CNN)

Aplikasi klasifikasi digit tulisan tangan menggunakan Convolutional Neural Network (CNN) dengan dataset MNIST. Pengguna dapat menggambar digit langsung di canvas dan prediksi muncul secara otomatis.

---

## Demo

Gambar digit 0-9 di canvas hitam, model akan langsung memprediksi angka beserta distribusi probabilitas untuk semua kelas.

## Live Demo

Akses aplikasi secara online di:
https://huggingface.co/spaces/simanisjawa/handwritten-digit-classifier

---

## Fitur

- Canvas interaktif untuk menggambar digit secara langsung
- Prediksi otomatis tanpa perlu menekan tombol
- Menampilkan top 3 kandidat digit beserta persentase kepercayaan
- Visualisasi distribusi probabilitas untuk semua digit 0-9

---

## Arsitektur Model

Model CNN dibangun dengan arsitektur Sequential:

```
Input (28x28x1)
    - Conv2D (32 filter, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Conv2D (64 filter, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Flatten
    - Dense (128, ReLU)
    - Dense (10, Softmax)
```

**Total parameter:** 225.034

---

## Hasil

| Metrik | Nilai |
|--------|-------|
| Training Accuracy | 99.51% |
| Validation Accuracy | 99.05% |
| Test Loss | 0.0294 |
| Epochs | 5 |
| Dataset | MNIST (70.000 gambar) |

---

## Instalasi dan Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/debosli/handwritten-digit-classifier-CNN-.git
cd handwritten-digit-classifier-CNN-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

---

## Struktur File

```
├── app.py                  # Aplikasi Streamlit
├── mnist_cnn.keras         # Model yang sudah dilatih
├── mnist_cnn.ipynb         # Notebook training
├── requirements.txt        # Dependencies
└── README.md
```

---

## Dependencies

- Python 3.11
- TensorFlow / Keras
- Streamlit
- streamlit-drawable-canvas
- NumPy

---

## Author

**Delino Vicky** - 38250009  
Universitas Bunda Mulia - Program Studi Kecerdasan Buatan
