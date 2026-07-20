import re
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Paramètres utilisés pendant l'entraînement
vocab_size = 10000
max_length = 200


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "imdb_bidirectional_lstm.keras"
    )


@st.cache_resource
def load_word_index():
    # Index fourni par le dataset IMDB
    word_index = imdb.get_word_index()

    # Les indices 0, 1, 2 et 3 sont réservés
    return {
        word: index + 3
        for word, index in word_index.items()
    }


def prepare_text(text, word_index):
    # Passage en minuscules
    text = text.lower()

    # Conservation des lettres et apostrophes
    words = re.findall(
        r"[a-zA-Z']+",
        text
    )

    sequence = []

    for word in words:
        index = word_index.get(word, 2)

        # On garde uniquement les mots du vocabulaire
        if index < vocab_size:
            sequence.append(index)
        else:
            sequence.append(2)

    # Ajout du symbole de début de critique
    sequence = [1] + sequence

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    return padded_sequence


# Configuration de la page
st.set_page_config(
    page_title="Analyse de sentiment IMDB",
    page_icon="🎬",
    layout="centered"
)


st.title("🎬 Analyse de sentiment IMDB")

st.write(
    "Entrez une critique de film en anglais. "
    "Le modèle indiquera si elle est positive ou négative."
)


# Chargement du modèle et du vocabulaire
try:
    model = load_model()
    word_index = load_word_index()

except Exception as error:
    st.error(
        "Impossible de charger le modèle."
    )

    st.code(str(error))
    st.stop()


review = st.text_area(
    "Critique du film",
    height=180,
    placeholder=(
        "This movie was excellent, "
        "the actors were amazing..."
    )
)


if st.button(
    "Analyser la critique",
    type="primary"
):
    if not review.strip():
        st.warning(
            "Veuillez écrire une critique."
        )

    else:
        prepared_review = prepare_text(
            review,
            word_index
        )

        prediction = model.predict(
            prepared_review,
            verbose=0
        )[0][0]

        positive_score = float(prediction)
        negative_score = 1 - positive_score


        st.subheader("Résultat")

        if positive_score >= 0.5:
            st.success("Sentiment positif 😊")
        else:
            st.error("Sentiment négatif 😞")


        st.write(
            "Probabilité positive :",
            f"{positive_score * 100:.2f} %"
        )

        st.write(
            "Probabilité négative :",
            f"{negative_score * 100:.2f} %"
        )

        st.progress(
            positive_score
        )


st.divider()

st.caption(
    "Modèle Bidirectional LSTM entraîné "
    "sur le dataset IMDB."
)