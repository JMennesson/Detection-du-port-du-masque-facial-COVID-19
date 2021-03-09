from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
import pickle


# Chargement du Dataset
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)
pickle_in = open("data.pickle", "rb")
data = pickle.load(pickle_in)
pickle_in = open("data1.pickle", "rb")
data1 = pickle.load(pickle_in)

# ............................................................................

# Mise à plat des données
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


# Normalisation des données
X = X / 255.0
X1 = X1 / 255.0


# ............................................................................

" Modèle : KNN NO HYPERPARAMETER TUNING "


# Chargement du modèle
knn = KNeighborsClassifier()
knn.fit(X, y.values.ravel())
predictions_set1 = knn.predict(X1)

# Evaluation de l'algorithme
print('KNN Accuracy: %.3f' % accuracy_score(y1, predictions_set1))

# Affichage de la matrice de confusion
cm = confusion_matrix(y1, predictions_set1)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title = 'KNN No Hyperparameter Tuning / Accuracy Score: {0}'.format(accuracy_score(y1, predictions_set1))
plt.title(all_sample_title,size=15)
plt.show()


# ............................................................................

" Modèle : KNN HYPERPARAMETER TUNING " 


best_params = {'weights': 'distance', 'n_neighbors': 2, 'metric': 'manhattan'}

# Chargement et entrainement du modèle
b_knn = KNeighborsClassifier(**best_params)
b_knn.fit(X, y.values.ravel())

# Prédictions 
train_pred = b_knn.predict(X)
y_pred = b_knn.predict(X1)

# Evaluation de l'algorithme
print('Accuracy Train: %.3f' % accuracy_score(y, train_pred))
print('Accuracy Test: %.3f' % accuracy_score(y1, y_pred))
print("\nClassification Report\n", classification_report(y1, y_pred))

# Affichage de la matrice de confusion 
cm = confusion_matrix(y1, y_pred)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
title = 'KNN Hyperparameter Tuning /Accuracy Score Best Params: {0}'.format(accuracy_score(y1, y_pred))
plt.title(title,size=15)
plt.show()



