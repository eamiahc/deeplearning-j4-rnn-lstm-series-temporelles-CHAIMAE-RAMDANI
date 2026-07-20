import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Nombre maximal de mots conservés
vocab_size = 10000

# Longueur maximale d'une critique
max_length = 200


# Chargement du dataset IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Dataset chargé avec succès.\n")

print("Nombre d'exemples d'entraînement :", len(X_train))
print("Nombre d'exemples de test :", len(X_test))

print("\nPremière étiquette :", y_train[0])

print("Longueur de la première critique :", len(X_train[0]))


# Padding des séquences
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


print("\nAprès padding :")

print("Shape X_train :", X_train.shape)
print("Shape X_test :", X_test.shape)

print("Shape y_train :", y_train.shape)
print("Shape y_test :", y_test.shape)


print("\nExemple de séquence :")
print(X_train[0][:30])


print("\nPremière étiquette :")
print(y_train[0])


print("\nNombre de critiques positives :",
      sum(y_train))

print("Nombre de critiques négatives :",
      len(y_train) - sum(y_train))


print("\nPhase 4 terminée.")