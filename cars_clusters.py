import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns



# Charger les données
cars = pd.read_csv('cars.csv')

# Nettoyage initial
cars_clean = cars.dropna().drop_duplicates().reset_index(drop=True)

#selection des données donc l'essence est soit Diesel, Gasoline, Electric
filtered_cars = cars_clean[cars_clean['fuel'].isin(['Diesel', 'Gasoline', 'Electric'])]

# Sélection des colonnes pertinentes
features = filtered_cars[['price', 'model', 'fuel', 'gear', 'hp', 'year']]

# Encoder les colonnes catégoriques
le_fuel = LabelEncoder()
features['fuel'] = le_fuel.fit_transform(features['fuel'])

le_model = LabelEncoder()
features['model'] = le_model.fit_transform(features['model'])

# Mapping manuel de 'gear':
features['gear'] = features['gear'].map({'Manual': 0, 'Automatic': 1,'Semi-automatic':2 })

# Imputer les valeurs manquantes avec la moyenne
imputer = SimpleImputer(strategy='mean')
features.iloc[:, :] = imputer.fit_transform(features)

# Conserver une copie des valeurs d'origine pour l'analyse après clustering
# Inclure 'fuel' dans la copie
original_features = features[['price', 'hp', 'year', 'fuel']].copy()

# Standardisation des colonnes numériques
#Les colonnes numériques sont standardisées pour assurer que les valeurs de différentes échelles n'affectent pas négativement l'algorithme de clustering.
scaler = StandardScaler()
features[['price', 'hp', 'year']] = scaler.fit_transform(features[['price', 'hp', 'year']])

# Appliquer K-means avec le nombre de clusters optimal
nb_cluster = 3
kmeans = KMeans(n_clusters=nb_cluster, random_state=42)
features['Cluster'] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Ajouter les labels des clusters dans la copie originale
original_features['Cluster'] = features['Cluster']

# Analyse des clusters avec les valeurs d'origines
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    print(f"\nCluster {i}:")
    print(f"size: {len(cluster)}")
    print(f"Prix moyen: {cluster['price'].mean():.2f}")
    print(f"Puissance moyenne: {cluster['hp'].mean():.2f}")
    print(f"Voiture la plus  (ancienne/ressente): {cluster['year'].min()}/{cluster['year'].max()}")
    print(f"Répartition carburant: Diesel={cluster['fuel'].value_counts().get(0, 0)}, Gasoline={cluster['fuel'].value_counts().get(1, 0)}, Electric={cluster['fuel'].value_counts().get(2, 0)}")


# Calculer un ratio Puissance/Prix pour chaque cluster
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    cluster['HP/Price'] = cluster['hp'] / cluster['price']
    print(f"Cluster {i}:")
    print(f"Ratio moyen Puissance/Prix: {cluster['HP/Price'].mean():.4f}")


# Analyse des années moyennes par type de carburant dans chaque cluster
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    print(f"Cluster {i} :")
    for fuel_type, name in zip([0, 1, 2], ['Diesel', 'Gasoline', 'Electric']):
        avg_year = cluster[cluster['fuel'] == fuel_type]['year'].mean()
        print(f"   {name}: Année moyenne = {avg_year:.0f}")



# Prix et puissance moyens par type de carburant
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    print(f"Cluster {i} :")
    for fuel_type, name in zip([0, 1, 2], ['Electric', 'Diesel', 'Gasoline']):
        avg_price = cluster[cluster['fuel'] == fuel_type]['price'].mean()
        avg_hp = cluster[cluster['fuel'] == fuel_type]['hp'].mean()
        print(f"   {name}: Prix moyen = {avg_price:.2f}, Puissance moyenne = {avg_hp:.2f}")

#**************************VISUALISATION********************************************************

# Calcul de la matrice de corrélation entre le prix et la puissance (hp)
correlation_matrix = original_features[['price', 'hp']].corr()

# Afficher la matrice de corrélation
print("Matrice de corrélation :")
print(correlation_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation entre prix et puissance")
plt.show()


# Visualiser la répartition des clusters sur un graphique  (Prix vs Puissance)
plt.figure(figsize=(8, 6))

# Boucle pour afficher les points de chaque cluster
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    plt.scatter(cluster['hp'], cluster['price'], label=f'Cluster {i}', alpha=0.9)




# Visualiser la répartition des carburants pour chaque clusters
for i in range(nb_cluster):
    cluster = original_features[original_features['Cluster'] == i]
    cluster['fuel'].value_counts().plot.pie(
        labels=['Electric', 'Diesel', 'Gasoline'], autopct='%1.1f%%', figsize=(6, 6)
    )
    plt.title(f"Répartition des carburants dans le Cluster {i}")
    plt.ylabel('')
    plt.show()



# Visualiser la répartition des carburants dans les clusters
fuel_counts = original_features.groupby(['Cluster', 'fuel']).size().unstack(fill_value=0)
fuel_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Répartition des carburants par cluster")
plt.xlabel("Cluster")
plt.ylabel("Nombre de voitures")
plt.show()

