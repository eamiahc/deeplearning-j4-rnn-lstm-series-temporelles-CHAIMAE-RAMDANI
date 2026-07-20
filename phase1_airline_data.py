import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# Adresse du dataset Airline Passengers
url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "airline-passengers.csv"
)


# Chargement des données
data = pd.read_csv(url)

print("Premières lignes :")
print(data.head())

print("\nTaille du dataset :", data.shape)

print("\nValeurs manquantes :")
print(data.isnull().sum())


# Conversion de la colonne Month en date
data["Month"] = pd.to_datetime(data["Month"])


# Récupération du nombre de passagers
passengers = data["Passengers"].values.reshape(-1, 1)


print("\nNombre total de mois :", len(passengers))
print("Valeur minimale :", passengers.min())
print("Valeur maximale :", passengers.max())


# Affichage de la série temporelle
plt.figure(figsize=(10, 5))

plt.plot(
    data["Month"],
    data["Passengers"]
)

plt.title("Évolution du nombre de passagers")
plt.xlabel("Date")
plt.ylabel("Nombre de passagers")
plt.grid()

plt.tight_layout()
plt.savefig("phase1_airline_series.png")
plt.close()


# Séparation temporelle :
# les premières valeurs pour le train,
# les dernières pour le test
train_size = int(len(passengers) * 0.67)

train_data = passengers[:train_size]
test_data = passengers[train_size:]


print("\nTaille train :", train_data.shape)
print("Taille test :", test_data.shape)


# Normalisation entre 0 et 1
# Le scaler apprend seulement sur le train
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


print("\nTrain normalisé :")
print("Minimum :", train_scaled.min())
print("Maximum :", train_scaled.max())


# Fonction de création des fenêtres
def create_dataset(dataset, window_size):
    X = []
    y = []

    for i in range(len(dataset) - window_size):
        window = dataset[i:i + window_size, 0]
        target = dataset[i + window_size, 0]

        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)


# On utilise les 12 derniers mois
# pour prédire le mois suivant
window_size = 12


# Création des données d'entraînement
X_train, y_train = create_dataset(
    train_scaled,
    window_size
)


# Pour créer la première fenêtre test,
# on récupère les 12 dernières valeurs du train
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


# Reshape attendu par un LSTM :
# nombre d'exemples, nombre de mois, nombre de features
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


print("\nShapes finales :")

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# Affichage de la première fenêtre
print("\nPremière fenêtre normalisée :")
print(X_train[0].flatten())

print("Cible normalisée :")
print(y_train[0])


# Retour aux vraies valeurs pour comprendre l'exemple
first_window = scaler.inverse_transform(
    X_train[0].reshape(-1, 1)
).flatten()

first_target = scaler.inverse_transform(
    [[y_train[0]]]
)[0, 0]


print("\nPremière fenêtre en vraies valeurs :")
print(first_window.astype(int))

print("Valeur à prédire :", int(first_target))


print("\nPhase 1 terminée.")
print("Graphique créé : phase1_airline_series.png")