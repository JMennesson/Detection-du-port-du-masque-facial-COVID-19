import pickle
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Chargement du dataset
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)

# Mise à plat des données
X = X.reshape(-1,64*64*3)
X1 = X1.reshape(-1,64*64*3)

# Chargement du modèle 
model_100 = svm.SVC()
model_100.fit(X, y) 

# Prédictions
y_pred = model_100.predict(X1)

# Evaluation du modèle
accuracy = model_100.score(X1, y1)  
print("Accuracy %f" % accuracy)
metrics.accuracy_score(y_true=y1, y_pred=y_pred)
print(metrics.classification_report(y1, y_pred))
print("\nClassification Report\n", metrics.classification_report(y1, y_pred))

# Affichage de la matrice de confusion 
cm = confusion_matrix(y1, y_pred)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Matrice de confusion')









