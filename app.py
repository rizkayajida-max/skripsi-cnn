import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model("model/model_uang.keras")

kelas = [
    "asli_rp100000",
    "asli_rp50000",
    "palsu_rp100000",
    "palsu_rp50000"
]

# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="Cuan Detector",
    page_icon="💵",
    layout="centered"
)

# ==========================
# HISTORY DETEKSI
# ==========================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# LOGIN SESSION
# ==========================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.image("assets/logo.png", width=150)

    st.title("🔐 Login Sistem")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.login = True
            st.rerun()

        else:

            st.error(
                "Username atau Password Salah"
            )

    st.stop()

# ==========================
# CSS CUSTOM
# ==========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1E88E5;
}

.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background-color:#E8F5E9;
    border:2px solid #4CAF50;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.conf-box{
    padding:15px;
    border-radius:15px;
    background-color:#E3F2FD;
    text-align:center;
    font-size:20px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown(
    "<div class='big-title'>💵 Cuan Detector</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Menggunakan Convolutional Neural Network (CNN)</div>",
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:

    st.image(
        "assets/logo.png",
        width=120
    )

    st.title("Cuan Detector")

    menu = st.radio(
        "Menu",
        [
            "Dashboard",
            "Deteksi Uang",
            "Tentang"
        ]
    )

    st.markdown("---")

    if st.button("Logout"):

        st.session_state.login = False
        st.rerun()

if menu == "Dashboard":

    st.title("📊 Dashboard Sistem")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Akurasi", "94.87%")

    with col2:
        st.metric("Epoch", "20")

    with col3:
        st.metric("Batch Size", "32")

    st.markdown("---")

    st.subheader("Arsitektur CNN")

    st.write("""
    • Conv2D 32 Filter + ReLU
    • MaxPooling2D
    • Conv2D 64 Filter + ReLU
    • MaxPooling2D
    • Conv2D 128 Filter + ReLU
    • MaxPooling2D
    • Flatten
    • Dense 128
    • Dropout 0.5
    • Dense 4 (Softmax)
    """)

    st.markdown("---")

    st.subheader("Grafik Accuracy")

    st.image(
        "hasil/grafik_accuracy.png"
    )

    st.subheader("Grafik Loss")

    st.image(
    "hasil/grafik_loss.png"
    )

    st.subheader("Confusion Matrix")

    st.image(
        "hasil/confusion_matrix.png"
    )

    st.markdown("---")

    st.subheader("Riwayat Deteksi")

    if len(st.session_state.history) > 0:

        st.table(
            st.session_state.history
        )
         # Tombol hapus riwayat
    if st.button("🗑️ Hapus Riwayat"):

        st.session_state.history = []

        st.rerun()
    else:

        st.info(
         "Belum ada riwayat deteksi."
        )

# ==========================
# KAMERA
# ==========================

if menu == "Deteksi Uang":

    st.title("📷 Deteksi Keaslian Uang")

    foto = st.camera_input(
        "Ambil Foto Uang"
    )

    

    if foto is not None:

     image = Image.open(foto)

     st.image(
        image,
        caption="Foto yang Diambil",
        use_container_width=True
      )

     # preprocessing
     img = image.resize((256,256))

     img_array = np.array(img)

     img_array = img_array / 255.0

     img_array = np.expand_dims(
        img_array,
        axis=0
        )

      # prediksi
     prediksi = model.predict(img_array)

     kelas_idx = np.argmax(prediksi)

     confidence = np.max(prediksi) * 100

     # Ambil probabilitas terbesar dan terbesar kedua
     sorted_pred = np.sort(prediksi[0])

     selisih = sorted_pred[-1] - sorted_pred[-2]

     st.subheader("Probabilitas Prediksi")

     for i, nama_kelas in enumerate(kelas):

        st.write(
        f"{nama_kelas} : {prediksi[0][i]*100:.2f}%"
        )

     # Threshold validasi
     if confidence < 98 or selisih < 0.30:

        st.error("⚠️ Nominal uang tidak valid atau tidak terdeteksi")

        st.info(
            "Sistem hanya dapat mendeteksi uang Rp50.000 dan Rp100.000."
        )

     else:

        hasil = kelas[kelas_idx]

        if "asli" in hasil:
            status = "ASLI ✅"
        else:
            status = "PALSU ❌"

        if "100000" in hasil:
            nominal = "Rp100.000"
        else:
            nominal = "Rp50.000"

        st.success(f"Status : {status}")

        st.markdown(
          f"""
          ### Nominal Terdeteksi

          {nominal}
          """
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='conf-box'>
            Tingkat Keyakinan<br>
            <b>{confidence:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(confidence/100))

        st.session_state.history.append(
          {
                "Nominal": nominal,
                "Status": status,
                "Confidence": f"{confidence:.2f}%"
          }
        )

if menu == "Tentang":

    st.title("ℹ️ Tentang Sistem")

    st.write("""
    Sistem ini dikembangkan untuk mendeteksi
    keaslian uang Rupiah dan mengklasifikasikan
    nominal uang menggunakan metode
    Convolutional Neural Network (CNN).
    """)

    st.markdown("---")

    st.write("""
    Pengembang:

    Rizka Yajida Amalia
    
    Nim : 2204030047

    Program Studi Teknik Informatika

    Universitas Islam Syekh Yusuf Tangerang

    Tahun 2026
    """)

# ==========================
# FOOTER
# ==========================
st.markdown(
    """
    <div class='footer'>
    Sistem Deteksi Keaslian Uang Rupiah Berbasis CNN
    </div>
    """,
    unsafe_allow_html=True
 )