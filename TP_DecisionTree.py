import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# La donnée contient la liste de voiture allemande
cars = pd.read_csv('cars.csv', sep=",")

# Remove null values and duplicates
cars_clean = cars.dropna().drop_duplicates()

# Reset Index of all rows
cars_clean = cars_clean.reset_index(drop=True)

# ***********************************# DECISION TREE #************************************** #
# Decision tree with columns : "MILEAGE" et "HOURSEPOWER"

# Create DataFrame with data
cars_data = cars_clean[['make','model' , 'fuel', 'mileage', 'price', 'year', 'hp']]

# X  with MILEAGE AND HP Columns
X = cars_data[['mileage', 'hp']]
y = cars_data['fuel']

print('X:', X.head())
print('Y:', y.head())
print('Labels:', y.unique())


# Create the decision tree classifier
max_depth = 4  # Profondeur maximale de l'arbre
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X, y)

# Create Decision tree
plt.figure(figsize=(35, 15))
plot_tree(
    clf, feature_names=X.columns,
    class_names=y.unique(),
    filled=True,
    rounded=True,
    fontsize=6,
    precision=5
)
plt.show()

# ***********************************# ***************** #************************************** #

# ***********************************# DECISION TREE (2) #************************************** #
# Decision tree with columns : "PRICE" et "YEAR"

# X  with PRICE AND YEAR Columns
X_2 = cars_data[['price', 'year']]
y_2 = cars_data['make']

print('Labels2:', y_2.unique())



# Create the decision tree classifier

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_2, y_2)

# Create Decision tree
plt.figure(figsize=(35, 15))
plot_tree(
    clf, feature_names=X_2.columns,
    class_names=y_2.unique(),
    filled=True,
    rounded=True,
    fontsize=6,
    precision=5
)
plt.show()

# ***********************************# ***************** #************************************** #

