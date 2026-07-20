import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
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


# Création des fenêtres temporelles
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


# Ajout du contexte nécessaire au test
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


# Format attendu par les modèles récurrents
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
print("X_test :", X_test.shape)


# Fonction pour créer le modèle LSTM
def create_lstm_model():
    model = Sequential([
        LSTM(
            50,
            input_shape=(window_size, 1)
        ),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# Fonction pour créer le modèle GRU
def create_gru_model():
    model = Sequential([
        GRU(
            50,
            input_shape=(window_size, 1)
        ),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# Même configuration pour les deux modèles
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ==============================
# Entraînement du modèle LSTM
# ==============================

print("\nEntraînement du modèle LSTM...")

lstm_model = create_lstm_model()

lstm_start = time.time()

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

lstm_time = time.time() - lstm_start


# ==============================
# Entraînement du modèle GRU
# ==============================

print("\nEntraînement du modèle GRU...")

gru_model = create_gru_model()

gru_start = time.time()

gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

gru_time = time.time() - gru_start


# Prédictions des deux modèles
lstm_predictions = lstm_model.predict(
    X_test,
    verbose=0
)

gru_predictions = gru_model.predict(
    X_test,
    verbose=0
)


# Retour aux vraies valeurs
lstm_predictions = scaler.inverse_transform(
    lstm_predictions
)

gru_predictions = scaler.inverse_transform(
    gru_predictions
)

y_test_real = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# Calcul du RMSE
lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        lstm_predictions
    )
)

gru_rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        gru_predictions
    )
)


# Nombre de paramètres
lstm_parameters = lstm_model.count_params()
gru_parameters = gru_model.count_params()


print("\n===== RÉSULTATS =====")

print("\nLSTM")
print("RMSE test :", round(lstm_rmse, 2))
print("Temps :", round(lstm_time, 2), "secondes")
print("Paramètres :", lstm_parameters)

print("\nGRU")
print("RMSE test :", round(gru_rmse, 2))
print("Temps :", round(gru_time, 2), "secondes")
print("Paramètres :", gru_parameters)


# Affichage du meilleur modèle
if lstm_rmse < gru_rmse:
    print("\nLe modèle LSTM obtient le meilleur RMSE.")
elif gru_rmse < lstm_rmse:
    print("\nLe modèle GRU obtient le meilleur RMSE.")
else:
    print("\nLes deux modèles ont le même RMSE.")


# Tableau récapitulatif
results = pd.DataFrame({
    "Modèle": ["LSTM", "GRU"],
    "RMSE test": [lstm_rmse, gru_rmse],
    "Temps entraînement": [lstm_time, gru_time],
    "Nombre paramètres": [
        lstm_parameters,
        gru_parameters
    ]
})

results.to_csv(
    "phase3_comparison_results.csv",
    index=False
)


print("\nTableau de comparaison :")
print(results)


# Graphique des prédictions
plt.figure(figsize=(11, 5))

plt.plot(
    y_test_real,
    label="Valeurs réelles"
)

plt.plot(
    lstm_predictions,
    label="Prédictions LSTM"
)

plt.plot(
    gru_predictions,
    label="Prédictions GRU"
)

plt.title("Comparaison des prédictions LSTM et GRU")
plt.xlabel("Mois du jeu de test")
plt.ylabel("Nombre de passagers")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "phase3_lstm_gru_predictions.png"
)

plt.close()


# Graphique des losses
plt.figure(figsize=(10, 5))

plt.plot(
    lstm_history.history["val_loss"],
    label="Validation LSTM"
)

plt.plot(
    gru_history.history["val_loss"],
    label="Validation GRU"
)

plt.title("Comparaison des losses de validation")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "phase3_lstm_gru_loss.png"
)

plt.close()


# Sauvegarde des modèles
lstm_model.save("airline_lstm_phase3.keras")
gru_model.save("airline_gru.keras")


print("\nFichiers créés :")
print("- phase3_comparison_results.csv")
print("- phase3_lstm_gru_predictions.png")
print("- phase3_lstm_gru_loss.png")
print("- airline_lstm_phase3.keras")
print("- airline_gru.keras")

print("\nPhase 3 terminée.")