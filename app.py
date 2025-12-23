import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===== LOAD MODEL =====
model = load_model("student_pass_model.h5")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# ===== UI =====
st.set_page_config(page_title="Student Prediction App")

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.subheader("📥 Input Data Akademik")

studytime = st.number_input(
    "📚 Waktu Belajar per Minggu (jam)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

absences = st.number_input(
    "❌ Jumlah Absensi",
    min_value=0,
    max_value=100,
    value=3,
    step=1
)

G1 = st.number_input(
    "📝 Nilai G1 (Semester 1)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

G2 = st.number_input(
    "📝 Nilai G2 (Semester 2)",
    min_value=0.0,
    max_value=10.0,
    value=5.0
    step=0.5
)

import pandas as pd

# ===== PREDICTION =====
if st.button("🔮 Prediksi Kelulusan"):

    # 1. Buat dataframe kosong sesuai fitur saat training
    input_df = pd.DataFrame(
        data=np.zeros((1, len(feature_cols))),
        columns=feature_cols
    )

    # 2. Isi fitur numerik yang kamu input
    if "studytime" in input_df.columns:
        input_df["studytime"] = studytime

    if "absences" in input_df.columns:
        input_df["absences"] = absences

    if "G1" in input_df.columns:
        input_df["G1"] = G1

    if "G2" in input_df.columns:
        input_df["G2"] = G2

    # 3. Scaling
    input_scaled = scaler.transform(input_df)

    # 4. Prediksi
    prediction = model.predict(input_scaled)[0][0]

    if prediction >= 0.5:
        st.success("✅ Mahasiswa DIPREDIKSI LULUS")
    else:
        st.error("❌ Mahasiswa DIPREDIKSI TIDAK LULUS")
