import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense
)

from tensorflow.keras.callbacks import EarlyStopping


# Pour avoir des résultats proches à chaque exécution
np.random.seed(42)
tf.random.set_seed(42)


# Paramètres
vocab_size = 10000
max_length = 200
embedding_dim = 64


# Chargement du dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)


# Padding
X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="post",
    truncating="post"
)


print("Shape X_train :", X_train.shape)
print("Shape X_test :", X_test.shape)


# Construction du modèle
model = Sequential([

    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )

])


model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)


print("\nRésumé du modèle\n")

model.summary()


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True

)


history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=10,

    batch_size=64,

    callbacks=[early_stopping],

    verbose=1

)


# Evaluation
loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print("\nAccuracy sur le jeu de test :")

print(round(accuracy * 100, 2), "%")


# Quelques prédictions
predictions = model.predict(
    X_test[:10],
    verbose=0
)


print("\nQuelques prédictions\n")

for i in range(10):

    prediction = predictions[i][0]

    sentiment = "Positif"

    if prediction < 0.5:
        sentiment = "Négatif"

    print(
        "Critique",
        i + 1,
        "- Réel :",
        y_test[i],
        "- Prédit :",
        sentiment,
        "- Score :",
        round(prediction, 3)
    )


# Courbes
plt.figure(figsize=(10,5))

plt.plot(
    history.history["accuracy"],
    label="Accuracy train"
)

plt.plot(
    history.history["val_accuracy"],
    label="Accuracy validation"
)

plt.title("Accuracy du modèle")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "phase5_accuracy.png"
)

plt.close()


plt.figure(figsize=(10,5))

plt.plot(
    history.history["loss"],
    label="Loss train"
)

plt.plot(
    history.history["val_loss"],
    label="Loss validation"
)

plt.title("Loss du modèle")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "phase5_loss.png"
)

plt.close()


# Sauvegarde
model.save(
    "imdb_bidirectional_lstm.keras"
)


print("\nFichiers créés :")

print("- phase5_accuracy.png")

print("- phase5_loss.png")

print("- imdb_bidirectional_lstm.keras")

print("\nPhase 5 terminée.")