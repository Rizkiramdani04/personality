import streamlit as st
import sklearn
import pickle
import numpy as np

#load model


st.set_page_config(page_title="Prediksi Kepribadian", layout="centered")

# Judul Halaman
st.title("🧠 Prediksi Kepribadian Berdasarkan Kebiasaan Sosial")

# Deskripsi Singkat
st.markdown("""
Selamat datang! Aplikasi ini akan memprediksi tipe kepribadian kamu berdasarkan beberapa kebiasaan sosial harian.  
Silakan isi data di bawah ini dan lihat hasil prediksi secara langsung! 😊
""")

# Load Model
model=pickle.load(open('personality.sav','rb'))

# Input User (tanpa form)
st.header("📋 Input Data")

col1, col2 = st.columns(2)
with col1:
    Stage_fear = st.selectbox('🎤 Takut Tampil di Depan Umum?', ['Ya', 'Tidak'])
    Time_spent_Alone = st.number_input('🛌 Waktu Sendiri (jam/hari)', min_value=0, step=1, format="%d")
    Social_event_attendance = st.number_input('👥 Menghadiri Acara Sosial (kali/bulan)', min_value=0, step=1, format="%d")
    Going_outside = st.number_input('🚶‍♂️ Frekuensi Keluar Rumah (kali/minggu)', min_value=0, step=1, format="%d")

with col2:
    Drained_after_socializing = st.selectbox('😓 Lelah Setelah Bersosialisasi?', ['Ya', 'Tidak'])
    Friends_circle_size = st.number_input('👨‍👩‍👧‍👦 Jumlah Teman Dekat', min_value=0, step=1, format="%d")
    Post_frequency = st.number_input('📱 Frekuensi Posting di Sosmed (post/minggu)', min_value=0, step=1, format="%d")

# Konversi data kategorikal ke numerik
Stage_fear_numerik = 1 if Stage_fear == 'Ya' else 0
Drained_after_socializing_numerik = 1 if Drained_after_socializing == 'Ya' else 0

# Tombol Prediksi
if st.button("🔍 Prediksi Sekarang"):
    input_data = np.array([[
        Time_spent_Alone,
        Social_event_attendance,
        Going_outside,
        Friends_circle_size,
        Post_frequency,
        Stage_fear_numerik,
        Drained_after_socializing_numerik
    ]])

    # Lakukan Prediksi
    prediksi = model.predict(input_data)
    hasil_label = "Introvert" if prediksi[0] == 0 else "Extrovert"
    st.success(f"🎯 **Hasil Prediksi Kepribadian Anda: {hasil_label}**")
    st.info("Catatan: Hasil ini bersifat prediktif dan bukan diagnosis resmi. Tetaplah menjadi dirimu yang terbaik! 💡")
