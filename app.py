import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords

# Download stopwords (hanya pertama kali)
nltk.download("stopwords")

# Load stopwords
stop_words_en = set(stopwords.words("english"))

# Load model dan TF-IDF
model = joblib.load("model/model_sentimen_twitter.pkl")
tfidf_vectorizer = joblib.load("model/vectorizer_tfidf_twitter.pkl")


# Fungsi preprocessing
def bersihkan_teks_twitter(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    tokens_bersih = [kata for kata in tokens if kata not in stop_words_en]
    return " ".join(tokens_bersih)


# ==============================================================================
# KONFIGURASI HALAMAN & STYLING CUSTOM
# ==============================================================================
st.set_page_config(
    page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered"
)

# Custom CSS untuk mempercantik card dan tombol
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DA1F2;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1991DA;
        border: none;
        color: white;
        transform: scale(1.02);
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Utama
st.title("Twitter Sentiment Analysis")
st.caption(
    "Aplikasi berbasis Machine Learning untuk mendeteksi sentimen tweet menggunakan TF-IDF dan Logistic Regression."
)
st.write("---")

# Layout Utama menggunakan Container agar rapi
with st.container():
    # Input pengguna
    tweet = st.text_area(
        "Masukkan teks tweet yang ingin dianalisis:",
        height=120,
        placeholder="Ketik tweet Anda di sini... (Contoh: I love this day!)",
    )

    # Tombol prediksi dengan spasi di bawahnya
    st.write("")
    tombol_analisis = st.button("Analisis Sentimen")
    st.write("")

# Logika Prediksi ketika tombol ditekan
if tombol_analisis:
    if tweet.strip() == "":
        st.warning("⚠️ Silakan masukkan teks tweet terlebih dahulu.")
    else:
        # Preprocessing
        tweet_bersih = bersihkan_teks_twitter(tweet)

        # AKAL-AKALAN FIX (Tetap dipertahankan jika file vectorizer kosong)
        if not hasattr(tfidf_vectorizer, "vocabulary_"):
            tfidf_vectorizer.fit([tweet_bersih])

        # TF-IDF & Prediksi
        tweet_tfidf = tfidf_vectorizer.transform([tweet_bersih])
        hasil = model.predict(tweet_tfidf)[0]

        # Hitung Probabilitas / Confidence Score
        probabilitas = model.predict_proba(tweet_tfidf)[0]
        confidence = max(probabilitas) * 100

        st.subheader("Hasil Analisis")

        # Membuat 2 Kolom untuk Hasil Sentimen dan Confidence Score
        col1, col2 = st.columns(2)

        with col1:
            # Tampilan visual berdasarkan label hasil prediksi
            if hasil.lower() == "positive":
                st.markdown(
                    '<div class="result-card" style="background-color: rgba(46, 204, 113, 0.15); border-left: 5px solid #2ecc71;">'
                    '<h3 style="color: #2ecc71; margin:0;">😊 Positive</h3>'
                    "<p style='margin-top:5px; margin-bottom:0; font-size:14px;'>Tweet ini mengandung pesan atau emosi positif.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif hasil.lower() == "negative":
                st.markdown(
                    '<div class="result-card" style="background-color: rgba(231, 76, 60, 0.15); border-left: 5px solid #e74c3c;">'
                    '<h3 style="color: #e74c3c; margin:0;">😠 Negative</h3>'
                    "<p style='margin-top:5px; margin-bottom:0; font-size:14px;'>Tweet ini mengandung pesan berunsur negatif atau kritikan.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card" style="background-color: rgba(52, 152, 219, 0.15); border-left: 5px solid #3498db;">'
                    '<h3 style="color: #3498db; margin:0;">😐 Neutral</h3>'
                    "<p style='margin-top:5px; margin-bottom:0; font-size:14px;'>Tweet ini bersifat netral atau berupa informasi umum.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        with col2:
            # Menggunakan st.metric bawaan streamlit agar terlihat profesional
            st.write("")  # memberikan sedikit jeda agar sejajar dengan card
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        # Fitur tambahan hasil teks bersih
        st.write("")
        with st.expander("🔍 Lihat Hasil Preprocessing Teks"):
            if tweet_bersih.strip() == "":
                st.info(
                    "Semua kata dalam tweet terhapus karena termasuk stopwords."
                )
            else:
                st.code(tweet_bersih, language="text")