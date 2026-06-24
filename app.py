import streamlit as st
import numpy as np
from deepface import DeepFace
import joblib
import os
import tempfile
from PIL import Image
import datetime

# ==========================================
# KONFIGURASI HALAMAN & MODEL
# ==========================================
st.set_page_config(page_title="Sistem Faskes Terpadu", page_icon="🏥", layout="wide")
MODEL_PATH = 'model_svm_arcface_terbaik.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# ==========================================
# INISIALISASI DATABASE SIMULASI (SESSION STATE)
# ==========================================
def init_db():
    if 'tb_faskes' not in st.session_state:
        st.session_state.tb_faskes = {
            "RS Medika": {"kamar_kosong": 15, "pelayanan": ["Poli Umum", "UGD"]},
            "Puskesmas Sehat": {"kamar_kosong": 5, "pelayanan": ["Poli Umum", "KIA"]}
        }
    if 'tb_admin' not in st.session_state:
        st.session_state.tb_admin = {
            "puskesmas1": {"password": "12345", "faskes": "Puskesmas Sehat"},
            "rsmedika1": {"password": "54321", "faskes": "RS Medika"}
        }
    if 'tb_dokter' not in st.session_state:
        st.session_state.tb_dokter = {}
    if 'tb_antrian' not in st.session_state:
        st.session_state.tb_antrian = []
    if 'tb_rekam_medis' not in st.session_state:
        st.session_state.tb_rekam_medis = []
    if 'tb_rujukan' not in st.session_state:
        st.session_state.tb_rujukan = {}
        
    # State untuk form pengunjung
    if 'scanned_pasien' not in st.session_state:
        st.session_state.scanned_pasien = None
        
    # State untuk login
    if 'admin_user' not in st.session_state: st.session_state.admin_user = None
    if 'dokter_user' not in st.session_state: st.session_state.dokter_user = None

init_db()

# ==========================================
# FUNGSI-FUNGSI PENDUKUNG
# ==========================================
def extract_face(img_buffer):
    if model is None:
        return None, "❌ Model tidak ditemukan. Pastikan path benar."
    
    image = Image.open(img_buffer)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name
        
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
            pesan = f"✅ Teridentifikasi: **{prediksi}** (Keyakinan: {probabilitas:.1f}%)"
            return prediksi, pesan
        return None, "❌ Wajah tidak dapat diekstrak."
    except ValueError:
        return None, "❌ Wajah tidak terdeteksi. Pastikan pencahayaan cukup."
    except Exception as e:
        return None, f"❌ Terjadi kesalahan: {e}"
    finally:
        os.remove(tmp_path)

def display_info_board():
    st.markdown("### 📊 STATUS KETERSEDIAAN")
    for nama, data in st.session_state.tb_faskes.items():
        st.write(f"- **{nama}** | Kamar Kosong: **{data['kamar_kosong']}** | Pelayanan: {', '.join(data['pelayanan'])}")
        
    st.markdown("### 👥 ANTRIAN AKTIF SAAT INI")
    aktif = [a for a in st.session_state.tb_antrian if a['status'] == 'Menunggu']
    if not aktif:
        st.info("Belum ada antrian.")
    else:
        for a in aktif:
            st.write(f"🕒 [{a['waktu']}] **{a['id']}** - {a['pasien']} (Faskes: {a['faskes']}) - Poli: {a['poli']}")

# ==========================================
# NAVIGASI (SIDEBAR)
# ==========================================
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Peran Anda:", ["🏠 Home", "👨‍👩‍👧‍👦 Pengunjung", "🏢 Admin Faskes", "🩺 Dokter / Perawat"])

# ==========================================
# HALAMAN HOME
# ==========================================
if menu == "🏠 Home":
    st.title("🏥 Sistem Informasi Faskes Terpadu")
    st.markdown("Selamat datang di Portal Sistem Fasilitas Kesehatan Terpadu (Database Terpusat). Silakan pilih peran Anda melalui navigasi di sebelah kiri untuk melanjutkan.")
    
    st.info("💡 Sistem ini menggunakan pengenalan wajah untuk pendaftaran pasien dan sistem manajemen antrian yang tersinkronisasi antar fasilitas kesehatan.")

# ==========================================
# HALAMAN PENGUNJUNG
# ==========================================
elif menu == "👨‍👩‍👧‍👦 Pengunjung":
    st.title("👨‍👩‍👧‍👦 Portal Pendaftaran Pasien")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 1. Registrasi Layanan")
        tipe_pasien = st.radio("Tipe Pasien", ["Pasien Baru", "Rujukan"], horizontal=True)
        
        valid_rujukan = False
        id_ruj_dipakai = None
        faskes_tujuan_default = 0
        
        if tipe_pasien == "Rujukan":
            id_rujukan = st.text_input("Masukkan ID Rujukan (Misal: RUJ-001)")
            if st.button("Cek Database Rujukan"):
                if id_rujukan in st.session_state.tb_rujukan:
                    data_ruj = st.session_state.tb_rujukan[id_rujukan]
                    if not data_ruj["status_dipakai"]:
                        st.success(f"✅ Rujukan ditemukan! Pasien: {data_ruj['pasien']} | Tujuan: {data_ruj['faskes_tujuan']}")
                        st.session_state.scanned_pasien = data_ruj['pasien']
                    else:
                        st.error("❌ ID Rujukan sudah dipakai.")
                else:
                    st.error("❌ ID Rujukan tidak valid.")
                    
        st.markdown("---")
        st.markdown("**Identifikasi Pasien (Wajib)**")
        opsi_input = st.radio("Metode Input Wajah:", ("Kamera", "Upload"), horizontal=True)
        
        img_buffer = None
        if opsi_input == "Kamera":
            img_buffer = st.camera_input("Ambil Foto untuk Pendaftaran")
        else:
            img_buffer = st.file_uploader("Unggah Foto Pasien", type=['jpg', 'jpeg', 'png'])
            
        if st.button("🔍 Identifikasi Wajah", use_container_width=True):
            if img_buffer is not None:
                with st.spinner("Menganalisis wajah..."):
                    nama_prediksi, pesan = extract_face(img_buffer)
                    if nama_prediksi:
                        st.session_state.scanned_pasien = nama_prediksi
                        st.success(pesan)
                    else:
                        st.error(pesan)
            else:
                st.warning("⚠️ Silakan ambil/unggah foto terlebih dahulu.")
                
        nama_pasien = st.text_input("Nama Lengkap Pasien (Terisi Otomatis)", value=st.session_state.scanned_pasien if st.session_state.scanned_pasien else "", disabled=True)
        
        faskes_list = list(st.session_state.tb_faskes.keys())
        if not faskes_list:
            st.warning("Belum ada data faskes.")
        else:
            faskes_pilihan = st.selectbox("Pilih Fasilitas Kesehatan", faskes_list)
            poli_pilihan = st.selectbox("Pilih Layanan / Poli", st.session_state.tb_faskes[faskes_pilihan]["pelayanan"])
            
            if st.button("Daftar & Ambil Antrian", type="primary", use_container_width=True):
                if not st.session_state.scanned_pasien:
                    st.error("❌ Mohon lakukan Identifikasi Wajah terlebih dahulu!")
                elif st.session_state.tb_faskes[faskes_pilihan]["kamar_kosong"] <= 0:
                    st.error(f"❌ Maaf, kamar di {faskes_pilihan} sedang penuh!")
                else:
                    sukses_daftar = True
                    if tipe_pasien == "Rujukan":
                        if id_rujukan not in st.session_state.tb_rujukan or st.session_state.tb_rujukan[id_rujukan]["status_dipakai"]:
                            st.error("❌ Pendaftaran gagal. ID Rujukan tidak valid atau belum diverifikasi.")
                            sukses_daftar = False
                        else:
                            st.session_state.tb_rujukan[id_rujukan]["status_dipakai"] = True
                            
                    if sukses_daftar:
                        st.session_state.tb_faskes[faskes_pilihan]["kamar_kosong"] -= 1
                        id_antrian = f"{faskes_pilihan[:3].upper()}-{poli_pilihan[:3].upper()}-{len(st.session_state.tb_antrian)+1:03d}"
                        waktu = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        st.session_state.tb_antrian.append({
                            "id": id_antrian, "pasien": st.session_state.scanned_pasien, 
                            "faskes": faskes_pilihan, "poli": poli_pilihan, 
                            "status": "Menunggu", "waktu": waktu
                        })
                        st.success(f"✅ Berhasil daftar! Nomor Antrian: **{id_antrian}**. Kamar tersisa di {faskes_pilihan}: {st.session_state.tb_faskes[faskes_pilihan]['kamar_kosong']}")
                        st.session_state.scanned_pasien = None # Reset session form
    
    with col2:
        st.markdown("### 2. Papan Informasi Realtime")
        display_info_board()
        if st.button("🔄 Segarkan Papan Informasi", use_container_width=True):
            st.rerun()

# ==========================================
# HALAMAN ADMIN
# ==========================================
elif menu == "🏢 Admin Faskes":
    st.title("🏢 Dasbor Admin Fasilitas Kesehatan")
    
    if st.session_state.admin_user is None:
        st.subheader("Login Admin Sistem")
        user = st.text_input("Username Admin")
        pwd = st.text_input("Password", type="password")
        if st.button("Login Secure"):
            if user in st.session_state.tb_admin and st.session_state.tb_admin[user]["password"] == pwd:
                st.session_state.admin_user = user
                st.success("✅ Login Berhasil!")
                st.rerun()
            else:
                st.error("❌ Username atau Password salah!")
    else:
        admin_data = st.session_state.tb_admin[st.session_state.admin_user]
        faskes_admin = admin_data["faskes"]
        
        st.info(f"👋 Selamat datang Admin dari **{faskes_admin}**")
        if st.button("Logout Admin", variant="secondary"):
            st.session_state.admin_user = None
            st.rerun()
            
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Update Database Layanan")
            jml_kamar = st.number_input("Set Ketersediaan Kamar Baru", min_value=0, value=st.session_state.tb_faskes[faskes_admin]["kamar_kosong"])
            layanan_baru = st.text_input("Tambah Layanan Baru (Pisahkan dengan koma)")
            if st.button("Simpan ke Database Faskes"):
                st.session_state.tb_faskes[faskes_admin]["kamar_kosong"] = int(jml_kamar)
                if layanan_baru.strip():
                    list_baru = [l.strip() for l in layanan_baru.split(",") if l.strip()]
                    st.session_state.tb_faskes[faskes_admin]["pelayanan"] = list(set(st.session_state.tb_faskes[faskes_admin]["pelayanan"] + list_baru))
                st.success("✅ Fasilitas berhasil diupdate!")
                
            st.markdown("---")
            st.markdown("### Panggil Antrian Selanjutnya")
            if st.button("🔊 Panggil Antrian Selanjutnya", type="primary"):
                dipanggil = False
                for a in st.session_state.tb_antrian:
                    if a["faskes"] == faskes_admin and a["status"] == "Menunggu":
                        a["status"] = "Sedang Dilayani"
                        st.success(f"🔊 Memanggil Pasien: **{a['pasien']}** ({a['id']}) - Poli: {a['poli']}")
                        dipanggil = True
                        break
                if not dipanggil:
                    st.warning("Tidak ada pasien menunggu di Faskes Anda.")
                    
        with col2:
            st.markdown("### Pendaftaran Akun Dokter Baru")
            dok_uname = st.text_input("Buat Username Dokter")
            dok_pwd = st.text_input("Buat Password Dokter", type="password")
            dok_nama = st.text_input("Nama Lengkap Dokter")
            if st.button("Daftarkan Dokter ke Sistem"):
                if not dok_uname or not dok_pwd or not dok_nama:
                    st.error("Lengkapi data dokter!")
                elif dok_uname in st.session_state.tb_dokter:
                    st.error("Username dokter sudah dipakai!")
                else:
                    st.session_state.tb_dokter[dok_uname] = {
                        "password": dok_pwd, "nama": dok_nama, "faskes": faskes_admin, "wajah_terdaftar": True
                    }
                    st.success(f"✅ Dokter {dok_nama} berhasil didaftarkan!")
            
            st.markdown("---")
            display_info_board()

# ==========================================
# HALAMAN DOKTER
# ==========================================
elif menu == "🩺 Dokter / Perawat":
    st.title("🩺 Ruang Praktik Elektronik Dokter")
    
    if st.session_state.dokter_user is None:
        st.subheader("Login Tenaga Medis")
        user = st.text_input("Username Dokter")
        pwd = st.text_input("Password", type="password")
        if st.button("Login Sistem Medis"):
            if user in st.session_state.tb_dokter and st.session_state.tb_dokter[user]["password"] == pwd:
                st.session_state.dokter_user = user
                st.success("✅ Login Berhasil!")
                st.rerun()
            else:
                st.error("❌ Akses Ditolak! Akun tidak ditemukan atau password salah.")
    else:
        dok_data = st.session_state.tb_dokter[st.session_state.dokter_user]
        faskes_dok = dok_data["faskes"]
        nama_dok = dok_data["nama"]
        
        st.info(f"👨‍⚕️ Selamat bertugas **dr. {nama_dok}** ({faskes_dok})")
        if st.button("Logout Dokter"):
            st.session_state.dokter_user = None
            st.rerun()
            
        st.markdown("---")
        
        # Cari pasien yang sedang dilayani di faskes ini
        pasien_aktif = [a['pasien'] for a in st.session_state.tb_antrian if a['status'] == 'Sedang Dilayani' and a['faskes'] == faskes_dok]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Panel Pemeriksaan Pasien")
            pasien_pilihan = st.selectbox("Pilih Pasien Aktif (Dalam Ruangan)", ["-- Pilih Pasien --"] + pasien_aktif)
            
            st.markdown("**Input Database Rekam Medis**")
            keluhan = st.text_area("Keluhan (Anamnesis)")
            diagnosis = st.text_area("Diagnosis Medis")
            resep = st.text_area("Resep / Tindakan")
            
            if st.button("💾 Simpan RM & Selesaikan Antrian", type="primary"):
                if pasien_pilihan == "-- Pilih Pasien --":
                    st.error("Pilih pasien terlebih dahulu!")
                else:
                    waktu = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.tb_rekam_medis.append({
                        "tanggal": waktu, "pasien": pasien_pilihan, "dokter": nama_dok, 
                        "faskes": faskes_dok, "anamnesis": keluhan, 
                        "diagnosis": diagnosis, "resep": resep
                    })
                    
                    # Update status antrian dan kembalikan kamar kosong
                    for a in st.session_state.tb_antrian:
                        if a["pasien"] == pasien_pilihan and a["status"] == "Sedang Dilayani" and a["faskes"] == faskes_dok:
                            a["status"] = "Selesai"
                            if faskes_dok in st.session_state.tb_faskes:
                                st.session_state.tb_faskes[faskes_dok]["kamar_kosong"] += 1
                            break
                            
                    st.success(f"✅ Rekam Medis {pasien_pilihan} disimpan ke Database. Antrian diselesaikan!")
                    st.rerun()

        with col2:
            st.markdown("### Tarik Riwayat Medis Pasien")
            if pasien_pilihan != "-- Pilih Pasien --":
                riwayat = [r for r in st.session_state.tb_rekam_medis if r['pasien'] == pasien_pilihan]
                if not riwayat:
                    st.info("Pasien belum memiliki riwayat medis di database.")
                else:
                    with st.expander("Buka Riwayat Rekam Medis", expanded=True):
                        for r in reversed(riwayat):
                            st.markdown(f"**[{r['tanggal']}] - dr. {r['dokter']} ({r['faskes']})**")
                            st.write(f"- **Keluhan:** {r['anamnesis']}")
                            st.write(f"- **Diagnosis:** {r['diagnosis']}")
                            st.write(f"- **Resep:** {r['resep']}")
                            st.divider()
            
            st.markdown("---")
            st.markdown("### Sistem Rujukan Terintegrasi")
            faskes_rujukan = st.selectbox("Faskes Rujukan Tujuan", [f for f in st.session_state.tb_faskes.keys() if f != faskes_dok])
            alasan_rujuk = st.text_input("Alasan Rujukan / Catatan Klinis")
            
            if st.button("Generate ID Rujukan"):
                if pasien_pilihan == "-- Pilih Pasien --":
                    st.error("Pilih pasien terlebih dahulu!")
                elif not alasan_rujuk:
                    st.error("Isi alasan rujukan!")
                else:
                    id_rujuk = f"RUJ-{len(st.session_state.tb_rujukan)+1:03d}"
                    st.session_state.tb_rujukan[id_rujuk] = {
                        "pasien": pasien_pilihan, "faskes_asal": faskes_dok, 
                        "faskes_tujuan": faskes_rujukan, "alasan": alasan_rujuk, 
                        "status_dipakai": False
                    }
                    st.success(f"📄 Surat Rujukan Dibuat! ID Rujukan: **{id_rujuk}**")
                    st.info(f"Berikan ID ini ke pasien untuk mendaftar di {faskes_rujukan}.")
