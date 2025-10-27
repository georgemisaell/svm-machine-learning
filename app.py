import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np


def preprocess_data(df):
    """
    Fungsi untuk membersihkan dan mengubah dataset employee.
    """

    # 1. Pisahkan fitur (X) dan target (y)
    X = df.drop('LeaveOrNot', axis=1)
    y = df['LeaveOrNot']

    # 2. Ubah Target (y) 
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 3. Identifikasi kolom numerik dan kategorikal
    numeric_cols = ['JoiningYear', 'PaymentTier',
                    'Age', 'ExperienceInCurrentDomain']
    categorical_cols = ['Education', 'City', 'Gender', 'EverBenched']

    # 4. Ubah Kolom Kategorikal menjadi Angka (One-Hot Encoding)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 5. Scaling Kolom Numerik
    scaler = StandardScaler()

    # Hanya scale kolom numerik
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler  # Kita kembalikan scaler untuk dipakai nanti jika ada input baru


def run_svm_analysis(X_train, y_train, X_test, y_test):
    """
    Melatih 3 kernel SVM dan mengembalikan hasil evaluasinya.
    """
    # Siapkan daftar kernel
    kernels = {
        'Linear': SVC(kernel='linear', probability=True),
        'Polynomial': SVC(kernel='poly', degree=3, probability=True),
        'RBF': SVC(kernel='rbf', probability=True)
    }

    results = {} 

    progress_bar = st.progress(0)
    total_kernels = len(kernels)

    for i, (name, model) in enumerate(kernels.items()):
        st.write(f"--- Melatih Model {name} ---")

        # 1. Latih model
        model.fit(X_train, y_train)

        # 2. Prediksi data test
        y_pred = model.predict(X_test)

        # 3. Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred)

        # 4. Simpan hasil
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'confusion_matrix': matrix,
            'report_string': classification_report(y_test, y_pred, zero_division=0)
        }

        # Update progress bar
        progress_bar.progress((i + 1) / total_kernels)

    return results

# --- Mulai UI/UX Streamlit ---


st.set_page_config(layout="wide")
st.title("Analisis SVM untuk Dataset Employee 🤖")
st.write("Aplikasi ini melatih 3 kernel SVM (Linear, Polinomial, RBF) dan membandingkan performanya.")

# 1. Baca Dataset
DATA_FILE = "employee.csv"
try:
    data = pd.read_csv(DATA_FILE)
    st.success(f"Dataset '{DATA_FILE}' berhasil di-load!")
    st.dataframe(data.head())  # Tampilkan 5 baris pertama
except FileNotFoundError:
    st.error(f"Error: File '{DATA_FILE}' tidak ditemukan.")
    st.write(
        f"Pastikan file `{DATA_FILE}` berada di folder yang sama dengan file `app.py` Anda.")
    st.stop()  # Hentikan eksekusi jika file tidak ada
except Exception as e:
    st.error(f"Error membaca file: {e}")
    st.stop()  # Hentikan eksekusi jika error lain


# Tombol untuk memulai analisis
if st.button("🚀 Mulai Analisis SVM"):

    with st.spinner("Sedang melakukan pra-pemrosesan data..."):
        # 2. Pra-pemrosesan Data
        try:
            X, y, scaler = preprocess_data(data.copy())
            st.write("Data setelah Pra-pemrosesan (Fitur X):")
            st.dataframe(X.head())
        except KeyError as e:
            st.error(f"Error saat preprocessing: Kolom {e} tidak ditemukan.")
            st.write(
                "Pastikan nama kolom di dataset Anda (misal: 'LeaveOrNot') sesuai dengan yang ada di kode `preprocess_data`.")
            st.stop()
        except Exception as e:
            st.error(f"Error saat preprocessing: {e}")
            st.stop()

    with st.spinner("Membagi data dan melatih model... Ini mungkin butuh waktu..."):
        # 3. Split Data        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 4. Latih dan Evaluasi Model
        svm_results = run_svm_analysis(X_train, y_train, X_test, y_test)

    st.success("Analisis Selesai! 🎉")

    # 5. Tampilkan Hasil
    st.header("Hasil Evaluasi Model")

    # Buat tabel perbandingan
    comparison_data = []
    for name, metrics in svm_results.items():
        comparison_data.append({
            "Kernel": name,
            "Akurasi": metrics['accuracy'],
            "Presisi": metrics['precision'],
            "Recall": metrics['recall'],
            "F1-Score": metrics['f1-score']
        })

    comparison_df = pd.DataFrame(comparison_data).set_index("Kernel")

    st.subheader("Tabel Perbandingan Metrik")
    # Memberi highlight pada nilai tertinggi di setiap kolom
    st.dataframe(comparison_df.style.highlight_max(
        axis=0, color='lightgreen', props='color:black; font-weight:bold;'))

    st.subheader("Detail Laporan Klasifikasi & Confusion Matrix")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1. Linear Kernel")
        st.text(svm_results['Linear']['report_string'])
        st.write("**Confusion Matrix:**")
        st.dataframe(svm_results['Linear']['confusion_matrix'])

    with col2:
        st.markdown("### 2. Polynomial Kernel")
        st.text(svm_results['Polynomial']['report_string'])
        st.write("**Confusion Matrix:**")
        st.dataframe(svm_results['Polynomial']['confusion_matrix'])

    with col3:
        st.markdown("### 3. RBF Kernel")
        st.text(svm_results['RBF']['report_string'])
        st.write("**Confusion Matrix:**")
        st.dataframe(svm_results['RBF']['confusion_matrix'])
