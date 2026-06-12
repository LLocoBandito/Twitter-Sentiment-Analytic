import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (hanya pertama kali)
nltk.download('stopwords')

# Load stopwords
stop_words_en = set(stopwords.words('english'))

# Load model dan TF-IDF
model = joblib.load("model/model_sentimen_twitter.pkl")
tfidf_vectorizer = joblib.load("model/vectorizer_tfidf_twitter.pkl")    


# Fungsi preprocessing
def bersihkan_teks_twitter(text):
    if not isinstance(text, str):
        return ""

    # Case folding
    text = text.lower()

    # Hapus mention
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)

    # Hapus hashtag
    text = re.sub(r'#', '', text)

    # Hapus angka dan tanda baca
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenisasi
    tokens = text.split()

    # Stopword removal
    tokens_bersih = [kata for kata in tokens if kata not in stop_words_en]

    return " ".join(tokens_bersih)


# Konfigurasi halaman
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Twitter Sentiment Analysis")
st.write(
    "Analisis sentimen tweet menggunakan TF-IDF dan Logistic Regression."
)

# Input pengguna
tweet = st.text_area(
    "Masukkan teks tweet:",
    height=150
)

# Tombol prediksi
if st.button("Analisis Sentimen"):

    if tweet.strip() == "":
        st.warning("Silakan masukkan teks tweet terlebih dahulu.")

    else:
        # Preprocessing
        tweet_bersih = bersihkan_teks_twitter(tweet)

        # TF-IDF
        tweet_tfidf = tfidf_vectorizer.transform([tweet_bersih])

        # Prediksi
        hasil = model.predict(tweet_tfidf)[0]

        # Probabilitas
        probabilitas = model.predict_proba(tweet_tfidf)[0]
        confidence = max(probabilitas) * 100

        st.subheader("Hasil Analisis")

        if hasil.lower() == "positive":
            st.success(f"😊 Sentimen: {hasil}")

        elif hasil.lower() == "negative":
            st.error(f"😠 Sentimen: {hasil}")

        else:
            st.info(f"😐 Sentimen: {hasil}")

        st.write(f"Confidence Score: **{confidence:.2f}%**")

        with st.expander("Lihat hasil preprocessing"):
            st.write(tweet_bersih)