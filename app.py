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
    max_value=20.0,
    value=10.0
)

G2 = st.number_input(
    "📝 Nilai G2 (Semester 2)",
    min_value=0.0,
    max_value=20.0,
    value=11.0
)

# ===== PREDICTION =====
if st.button("🔮 Prediksi Kelulusan"):
    input_data = np.array([[studytime, absences, G1, G2]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    if prediction >= 0.5:
        st.success("✅ Mahasiswa DIPREDIKSI LULUS")
    else:
        st.error("❌ Mahasiswa DIPREDIKSI TIDAK LULUS")
