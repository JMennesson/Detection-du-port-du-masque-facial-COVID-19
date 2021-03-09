from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from sklearn.tree import plot_tree

# Chargement du dataset
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("data.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)
pickle_in = open("data1.pickle", "rb")
data1 = pickle.load(pickle_in)

# ............................................................................

# Mise à plat 
X = X.reshape(-1,64*64*3)
X1 = X1.reshape(-1,64*64*3)
data = data.reshape(-1,64*64*3)
data1 = data1.reshape(-1,64*64*3)


# Convertion en Dataframe
cols = []
for i in range(0, len(data[0])):
    cols.append("P" + str(i))

numpy_data = data
X = pd.DataFrame(data=numpy_data, columns=[cols])
y = pd.DataFrame(data=y, columns=["Mask_Target"])


# Normalisation
X = X / 255.0
X1 = X1 / 255.0

# ............................................................................

# Chargement du modèle
decision_trees = DecisionTreeClassifier()
decision_trees.fit(X, y.values.ravel())

# Prédictions
predictions_set1 = decision_trees.predict(X1)

# ............................................................................

# Evaluation du modèle 
print('Decision Trees Precision: %.3f' % precision_score(y1, predictions_set1, average='micro'))
print('Decision Trees Recall: %.3f' % recall_score(y1, predictions_set1, average='micro'))
print('Decision Trees F1 Score: %.3f' % f1_score(y1, predictions_set1, average='micro'))

print("\nClassification Report\n", classification_report(y1, predictions_set1))

# Affichage de la matrice de confusion 
cm = confusion_matrix(y1, predictions_set1)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
title = 'Accuracy Score, No Hyperparameter Tuning: {0}'.format(accuracy_score(y1, predictions_set1))
plt.title(title,size=15)
plt.show()

# Affichage de l'arbre décisionnel obtenu 
plot_tree(decision_trees)

