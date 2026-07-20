import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# Pour avoir des résultats proches à chaque exécution
np.random.seed(42)
tf.random.set_seed(42)


# Chargement du dataset AG News
train_data, test_data = tfds.load(
    "ag_news_subset",
    split=["train", "test"],
    as_supervised=True
)


print("Dataset AG News chargé.")


# Affichage d'un exemple
for text, label in train_data.take(1):
    print("\nExemple d'article :")
    print(text.numpy().decode("utf-8"))

    print("\nClasse :", label.numpy())


# Paramètres
vocab_size = 15000
max_length = 100
embedding_dim = 64
batch_size = 128


# Transformation du texte en nombres
vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_length
)


# Le vectorizer apprend le vocabulaire
train_texts = train_data.map(
    lambda text, label: text
)

vectorizer.adapt(
    train_texts.batch(1024)
)


# Préparation des datasets
train_data = train_data.shuffle(
    10000,
    seed=42
)

train_data = train_data.batch(
    batch_size
).prefetch(
    tf.data.AUTOTUNE
)


test_data = test_data.batch(
    batch_size
).prefetch(
    tf.data.AUTOTUNE
)


# Construction du modèle
model = Sequential([
    vectorizer,

    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        4,
        activation="softmax"
    )
])


# Compilation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nRésumé du modèle :")
model.summary()


# Arrêt si la validation ne s'améliore plus
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# Entraînement
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=6,
    callbacks=[early_stopping],
    verbose=1
)


# Évaluation
test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=0
)


print(
    "\nAccuracy test :",
    round(test_accuracy * 100, 2),
    "%"
)


# Noms des catégories
class_names = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
]


# Test sur quelques phrases
examples = tf.constant([
    "The football team won the championship match",
    "The company announced a new financial investment",
    "A new artificial intelligence system was released",
    "World leaders met to discuss international security"
])


predictions = model.predict(
    examples,
    verbose=0
)


print("\nQuelques prédictions :")

for text, prediction in zip(examples.numpy(), predictions):
    predicted_class = np.argmax(prediction)
    score = np.max(prediction)

    print("\nTexte :", text.decode("utf-8"))
    print("Catégorie :", class_names[predicted_class])
    print("Score :", round(float(score), 3))


# Courbe accuracy
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Accuracy train"
)

plt.plot(
    history.history["val_accuracy"],
    label="Accuracy validation"
)

plt.title("Accuracy du modèle AG News")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("phase6_ag_news_accuracy.png")
plt.close()


# Courbe loss
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Loss train"
)

plt.plot(
    history.history["val_loss"],
    label="Loss validation"
)

plt.title("Loss du modèle AG News")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("phase6_ag_news_loss.png")
plt.close()


# Sauvegarde du modèle
model.save("ag_news_model.keras")


print("\nFichiers créés :")
print("- phase6_ag_news_accuracy.png")
print("- phase6_ag_news_loss.png")
print("- ag_news_model.keras")

print("\nPhase 6 terminée.")