import streamlit as st
import numpy as np
from deepface import DeepFace
import joblib
import os
import tempfile
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Face Recognition Faskes", page_icon="🏥")
st.title("🏥 Portal Pendaftaran Berbasis Wajah")

# Load Model
MODEL_PATH = '/content/drive/MyDrive/SEC_ARDI/model_svm_arcface_terbaik.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error("❌ Model tidak ditemukan. Pastikan path benar.")
else:
    st.success("✅ Model berhasil dimuat!")

    # Input Kamera / Upload
    st.write("Silakan ambil foto wajah Anda.")
    opsi = st.radio("Metode Input:", ("Kamera", "Upload"))
    
    img_buffer = None
    if opsi == "Kamera":
        img_buffer = st.camera_input("Ambil Foto")
    else:
        img_buffer = st.file_uploader("Unggah Foto", type=['jpg', 'jpeg', 'png'])

    # Prediksi
    if img_buffer is not None:
        image = Image.open(img_buffer)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
            
        with st.spinner("Menganalisis wajah..."):
            try:
                rep = DeepFace.represent(
                    img_path=tmp_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=True
                )
                
                if len(rep) > 0:
                    embedding = np.array(rep[0]["embedding"]).reshape(1, -1)
                    prediksi = model.predict(embedding)[0]
                    probabilitas = np.max(model.predict_proba(embedding)) * 100
                    
                    st.success(f"**Teridentifikasi sebagai:** {prediksi}")
                    st.info(f"**Keyakinan:** {probabilitas:.1f}%")
                else:
                    st.warning("⚠️ Wajah tidak dapat diekstrak.")
            except ValueError:
                st.error("❌ Wajah tidak terdeteksi. Pastikan pencahayaan cukup.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        
        os.remove(tmp_path)
