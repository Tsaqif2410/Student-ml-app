import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===== LOAD MODEL =====
model = load_model("student_pass_model.h5")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

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
