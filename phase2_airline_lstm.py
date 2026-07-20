import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Pour obtenir des résultats proches à chaque exécution
np.random.seed(42)
tf.random.set_seed(42)


# Chargement du dataset
url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "airline-passengers.csv"
)

data = pd.read_csv(url)

passengers = data["Passengers"].values.reshape(-1, 1)


# Séparation train et test
train_size = int(len(passengers) * 0.67)

train_data = passengers[:train_size]
test_data = passengers[train_size:]


# Normalisation
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


# Création des fenêtres de 12 mois
def create_dataset(dataset, window_size):
    X = []
    y = []

    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size, 0])
        y.append(dataset[i + window_size, 0])

    return np.array(X), np.array(y)


window_size = 12


# Données d'entraînement
X_train, y_train = create_dataset(
    train_scaled,
    window_size
)


# On ajoute les 12 derniers mois du train
# au début du test
test_with_context = np.concatenate(
    (
        train_scaled[-window_size:],
        test_scaled
    ),
    axis=0
)

X_test, y_test = create_dataset(
    test_with_context,
    window_size
)


# Format attendu par le LSTM
X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# Construction du modèle
model = Sequential([
    LSTM(
        50,
        input_shape=(window_size, 1)
    ),
    Dense(1)
])


# Compilation
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


print("\nRésumé du modèle :")
model.summary()


# Arrêt automatique si la validation ne s'améliore plus
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# Entraînement
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
    shuffle=False
)

training_time = time.time() - start_time


print(
    "\nDurée d'entraînement :",
    round(training_time, 2),
    "secondes"
)


# Prédictions
train_predictions = model.predict(
    X_train,
    verbose=0
)

test_predictions = model.predict(
    X_test,
    verbose=0
)


# Retour aux vraies valeurs
train_predictions = scaler.inverse_transform(
    train_predictions
)

test_predictions = scaler.inverse_transform(
    test_predictions
)

y_train_real = scaler.inverse_transform(
    y_train.reshape(-1, 1)
)

y_test_real = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# Calcul du RMSE
train_rmse = np.sqrt(
    mean_squared_error(
        y_train_real,
        train_predictions
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        test_predictions
    )
)


print(
    "\nRMSE train :",
    round(train_rmse, 2)
)

print(
    "RMSE test :",
    round(test_rmse, 2)
)


# Affichage de quelques prédictions
print("\nQuelques prédictions sur le test :")

for i in range(5):
    real_value = y_test_real[i, 0]
    predicted_value = test_predictions[i, 0]

    print(
        "Valeur réelle :",
        round(real_value, 2),
        "| Prédiction :",
        round(predicted_value, 2)
    )


# Courbe de la loss
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Loss train"
)

plt.plot(
    history.history["val_loss"],
    label="Loss validation"
)

plt.title("Évolution de la loss du modèle LSTM")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "phase2_lstm_loss.png"
)

plt.close()


# Courbe prédictions contre vraies valeurs
plt.figure(figsize=(11, 5))

plt.plot(
    y_test_real,
    label="Valeurs réelles"
)

plt.plot(
    test_predictions,
    label="Prédictions LSTM"
)

plt.title("Airline Passengers : prédictions LSTM")
plt.xlabel("Mois du jeu de test")
plt.ylabel("Nombre de passagers")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "phase2_lstm_predictions.png"
)

plt.close()


# Sauvegarde du modèle
model.save(
    "airline_lstm.keras"
)


print("\nFichiers créés :")
print("- phase2_lstm_loss.png")
print("- phase2_lstm_predictions.png")
print("- airline_lstm.keras")

print("\nPhase 2 terminée.")