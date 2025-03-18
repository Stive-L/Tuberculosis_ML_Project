# Tuberculosis Machine Learning Project

## Description

Ce projet utilise un jeu de données sur la tuberculose pour explorer, prétraiter et analyser les facteurs influençant la maladie. L'objectif est de mettre en œuvre plusieurs modèles de classification afin de prédire la présence de la tuberculose chez les patients et d'en extraire des insights pertinents.

## Objectifs

- Réaliser une analyse exploratoire des données (EDA)
- Nettoyer et prétraiter les données (gestion des valeurs manquantes, encodage, normalisation)
- Implémenter et comparer plusieurs modèles de classification
- Évaluer les modèles avec des métriques de performance
- Extraire des recommandations basées sur les résultats

## Structure du projet

### 1. Exploration et Prétraitement des Données

- `Preprocessing.py` : Gestion des valeurs manquantes et transformation des données
- `Summary_statistics.py` : Calcul de statistiques descriptives
- `Age_distribution.py` : Visualisation de la distribution des âges
- `Gender_distribution.py` : Analyse de la distribution des genres

### 2. Implémentation des Modèles

- `Decision_Tree.py` : Implémentation d'un arbre de décision
- `KNN.py` : Implémentation du k-Nearest Neighbors (KNN)
- `logistics_regression.py` : Implémentation d'une régression logistique

### 3. Données

- `Raw_data.csv` : Jeu de données brut sur la tuberculose
- `Tuberculosis_data_processed_data.csv` : Données prétraitées
