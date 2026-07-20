
# Deep Learning Jour 4 lstm-series-temporelles-CHAIMAE-RAMDANI
# Deep Learning – Jour 4
## Réseaux de neurones récurrents (RNN, LSTM, GRU) et traitement du langage naturel
---

# Présentation

Ce projet a été réalisé dans le cadre du TP de Deep Learning consacré aux réseaux de neurones récurrents (RNN), aux modèles LSTM et GRU ainsi qu'au traitement du langage naturel (NLP).

L'objectif est de mettre en pratique les concepts étudiés en cours à travers plusieurs applications :

- prévision d'une série temporelle ;
- analyse de sentiments sur des critiques de films ;
- classification automatique d'articles de presse ;
- création d'une application interactive avec Streamlit.

---

# Technologies utilisées

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow Datasets
- Streamlit
- Git & GitHub

---

# Structure du projet

```
.
├── phase1_airline_data.py
├── phase2_airline_lstm.py
├── phase3_airline_gru_comparison.py
├── phase4_imdb_preprocessing.py
├── phase5_imdb_bidirectional_lstm.py
├── phase6_ag_news.py
├── app.py
├── requirements.txt
├── README.md
```

---

# Phase 1 – Préparation des données Airline Passengers

Cette première étape consiste à préparer les données avant l'entraînement du modèle.

Travail réalisé :

- chargement du dataset Airline Passengers ;
- analyse des données ;
- séparation Train / Test sans mélanger les données ;
- normalisation avec MinMaxScaler ;
- création des fenêtres temporelles (Sliding Window) de 12 mois ;
- préparation des données pour les modèles LSTM.

Cette étape permet de transformer une série temporelle en un problème d'apprentissage supervisé.

---

# Phase 2 – Prévision avec un modèle LSTM

Dans cette partie, un réseau LSTM est construit afin de prédire le nombre de passagers du mois suivant.

Travail réalisé :

- création du modèle LSTM ;
- entraînement du réseau ;
- calcul du RMSE ;
- visualisation des courbes d'entraînement ;
- comparaison entre les valeurs réelles et les valeurs prédites.

Le modèle apprend correctement la tendance générale de la série temporelle.

---

# Phase 3 – Comparaison entre LSTM et GRU

Cette étape consiste à comparer deux architectures de réseaux récurrents.

Les éléments comparés sont :

- temps d'entraînement ;
- nombre de paramètres ;
- erreur RMSE ;
- qualité des prédictions.

Résultat obtenu :

- le modèle **LSTM** fournit les meilleures prédictions sur le dataset Airline Passengers ;
- le modèle **GRU** est plus léger et plus rapide mais moins précis sur ce problème.

Cette comparaison montre les différences entre les deux architectures.

---

# Phase 4 – Prétraitement du dataset IMDB

Cette phase prépare les critiques de films avant l'entraînement.

Travail réalisé :

- chargement du dataset IMDB ;
- limitation du vocabulaire aux mots les plus fréquents ;
- padding des séquences ;
- préparation des données pour le modèle Bidirectional LSTM.

---

# Phase 5 – Analyse de sentiments

Un modèle Bidirectional LSTM est utilisé afin de déterminer si une critique est positive ou négative.

Travail réalisé :

- couche Embedding ;
- couche Bidirectional LSTM ;
- entraînement du modèle ;
- évaluation sur le jeu de test ;
- sauvegarde du modèle.

Le modèle obtient une bonne précision sur les critiques IMDB.

---

# Phase 6 – Classification AG News

Cette partie applique les réseaux récurrents à une tâche de classification de textes.

Les articles sont classés dans quatre catégories :

- World
- Sports
- Business
- Sci/Tech

Travail réalisé :

- préparation des textes ;
- vectorisation ;
- entraînement d'un modèle Bidirectional LSTM ;
- évaluation des performances ;
- sauvegarde du modèle.

Le modèle est capable de reconnaître automatiquement la catégorie d'un nouvel article.

---

# Application Streamlit

Une application web a été développée afin d'utiliser le modèle IMDB.

Fonctionnalités :

- saisie d'une critique de film ;
- analyse automatique du texte ;
- affichage du sentiment détecté ;
- affichage de la probabilité prédite.

Cette application permet d'utiliser le modèle sans écrire de code Python.

Pour lancer l'application :

```bash
streamlit run app.py
```

---

# Résultats

Les différentes expériences montrent que :

- le modèle LSTM est particulièrement adapté aux séries temporelles ;
- le modèle GRU est plus rapide mais légèrement moins performant sur le dataset Airline Passengers ;
- le Bidirectional LSTM obtient de bonnes performances sur les tâches de traitement du langage naturel ;
- Streamlit permet de rendre le modèle facilement utilisable par un utilisateur.

---

# Installation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# Exécution

Prévision de séries temporelles :

```bash
python phase1_airline_data.py
python phase2_airline_lstm.py
python phase3_airline_gru_comparison.py
```

Analyse de sentiments :

```bash
python phase4_imdb_preprocessing.py
python phase5_imdb_bidirectional_lstm.py
```

Classification AG News :

```bash
python phase6_ag_news.py
```

Application Streamlit :

```bash
streamlit run app.py
```

---

# Conclusion

Ce projet m'a permis de mettre en pratique les principales architectures de réseaux de neurones récurrents étudiées en cours.

À travers plusieurs jeux de données, j'ai appris à :

- préparer des séries temporelles ;
- construire des modèles LSTM et GRU ;
- comparer leurs performances ;
- appliquer les réseaux récurrents au traitement automatique du langage naturel ;
- développer une application simple avec Streamlit permettant d'utiliser un modèle d'intelligence artificielle.

Ce travail m'a permis d'approfondir ma compréhension des modèles séquentiels et de leur utilisation dans différents domaines du Deep Learning.