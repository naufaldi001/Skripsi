import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

from preprocess import clean_text
from predict import predict_sentiment

# Config
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Produk",
    layout="wide"
)

MODEL_ACCURACY = 0.8448
TOTAL_TRAIN_DATA = "±45.000 ulasan"

# Header
st.title("Analisis Sentimen Ulasan Produk")

st.caption(
    "Aplikasi ini mengklasifikasikan sentimen ulasan produk e-commerce"
    "menggunakan algoritma **Multinomial Naive Bayes**, "
    "dengan tiga kelas sentimen: **Positif, Netral, dan Negatif**."
)

# Metric card
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Naive Bayes")

with col2:
    st.metric("Data Latih", TOTAL_TRAIN_DATA)

with col3:
    st.metric("Akurasi Model", f"{MODEL_ACCURACY*100:.2f}%")


st.divider()

# Analis teks tunggal
st.subheader("Analisis Teks")

user_text = st.text_area(
    "Masukkan ulasan produk:",
    height=120
)

if st.button("Prediksi Sentimen"):
    if user_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        prediction = predict_sentiment(user_text)

        st.success(f"Hasil Prediksi Sentimen: **{prediction}**")

        with st.expander("Lihat hasil preprocessing"):
            st.write(clean_text(user_text))

st.divider()

# Analisis data CSV
st.subheader("Analisis Data CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV berisi ulasan produk",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview data:", df.head())

    text_column = st.selectbox(
        "Pilih kolom yang berisi teks ulasan:",
        df.columns
    )

    if st.button("Proses & Analisis Data"):
        df["clean_text"] = df[text_column].astype(str).apply(clean_text)
        df["predicted_label"] = df["clean_text"].apply(predict_sentiment)

        st.success("Analisis selesai.")

        total = len(df)
        pos = (df["predicted_label"] == "POS").sum()
        net = (df["predicted_label"] == "NET").sum()
        neg = (df["predicted_label"] == "NEG").sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Ulasan", total)
        with c2:
            st.metric("Positif", pos)
            st.caption(f"{pos/total*100:.1f}% dari total")
        with c3:
            st.metric("Netral", net)
            st.caption(f"{net/total*100:.1f}% dari total")
        with c4:
            st.metric("Negatif", neg)
            st.caption(f"{neg/total*100:.1f}% dari total")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Distribusi Sentimen")

            fig, ax = plt.subplots()
            ax.pie(
                [pos, net, neg],
                labels=["Positif", "Netral", "Negatif"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.4)
            )
            ax.axis("equal")
            st.pyplot(fig)

        with col_right:
            st.subheader("Masalah Utama di Ulasan Negatif")

            stopwords_simple = {
                "dan","yang","tidak","dengan","untuk","pada",
                "ini","itu","saya","barang","produk"
            }

            neg_text = " ".join(
                df[df["predicted_label"] == "NEG"]["clean_text"].astype(str)
            )

            words = [
                w for w in neg_text.split()
                if w not in stopwords_simple and len(w) > 3
            ]

            if words:
                issues = Counter(words).most_common(10)
                issues_df = pd.DataFrame(
                    issues, columns=["Masalah", "Frekuensi"]
                )
                st.bar_chart(issues_df.set_index("Masalah"))
            else:
                st.info("Tidak ada ulasan negatif.")

        st.divider()

        # Wordcloud
        st.subheader("Word Cloud per Sentimen")

        def show_wordcloud(text, title, cmap):
            wc = WordCloud(
                width=400,
                height=250,
                background_color="white",
                colormap=cmap
            ).generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title)
            st.pyplot(fig)

        wc1, wc2, wc3 = st.columns(3)

        with wc1:
            show_wordcloud(
                " ".join(df[df["predicted_label"]=="POS"]["clean_text"]),
                "Word Cloud Positif",
                "Greens"
            )

        with wc2:
            show_wordcloud(
                " ".join(df[df["predicted_label"]=="NET"]["clean_text"]),
                "Word Cloud Netral",
                "Greys"
            )

        with wc3:
            show_wordcloud(
                " ".join(df[df["predicted_label"]=="NEG"]["clean_text"]),
                "Word Cloud Negatif",
                "Reds"
            )

        st.divider()

        # Ringkasan
        st.subheader("Ringkasan Analisis")

        if pos > net and pos > neg:
            st.success(
                "Mayoritas ulasan memiliki sentimen **positif**, "
                "menunjukkan tingkat kepuasan pengguna yang tinggi."
            )
        elif neg > pos and neg > net:
            st.error(
                "Sentimen **negatif** mendominasi, "
                "menunjukkan adanya permasalahan pada produk atau layanan."
            )
        else:
            st.info(
                "Sebagian besar ulasan bersifat **netral**, "
                "menunjukkan pengguna cenderung memberikan deskripsi tanpa opini kuat."
            )

        # download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Hasil Analisis",
            csv,
            "hasil_analisis_sentimen.csv",
            "text/csv"
        )

st.divider()
st.caption(
    "Skripsi – Analisis Sentimen Ulasan Produk | "
    "Model: Multinomial Naive Bayes"
)
