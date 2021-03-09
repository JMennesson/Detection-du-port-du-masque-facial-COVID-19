import sklearn
import sklearn.naive_bayes
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Chargement du dataset 
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)

# Labels
CATEGORIES = ['Without Mask', 'Wearing Mask', 'Incorrectly Wearing Mask']

# Mise à plat des données
X = X.reshape(-1, 64*64*3)
X1 = X1.reshape(-1, 64*64*3)

# Normalisation
X = X / 255.0
X1 = X1 / 255.0

# .............................................................................

" Modèle 1 : Distribution Gaussienne "


# Importation du modèle suivant une distribution Gaussienne
gaussianNB_model = sklearn.naive_bayes.GaussianNB()

# Training du modèle 
gaussianNB_model.fit(X, y)

# Prediction
gnb_y_pred = gaussianNB_model.predict(X1)


# Evaluation de l'algorithme
print('GaussianNB Metrics')
accuracy = sklearn.metrics.accuracy_score(y1, gnb_y_pred)
print('Accuracy: %f' % accuracy)
recall = sklearn.metrics.recall_score(y1, gnb_y_pred, average='micro')
print('Recall: %f' % recall)
f1_score = sklearn.metrics.f1_score(y1, gnb_y_pred, average='micro')
print('F1 Score: %f' % f1_score, end='\n\n')

# Affichage de la matrice de confusion
cm = sklearn.metrics.confusion_matrix(y1, gnb_y_pred)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=.5, square=True,cmap='Blues', yticklabels=CATEGORIES, xticklabels=CATEGORIES)
plt.title('Confusion Matrix for GaussianNB')


# .............................................................................

" Modèle 1 : Distribution Multinomiale "


# Modèle
multinomialNB_model = sklearn.naive_bayes.MultinomialNB()
multinomialNB_model.fit(X, y)
mnb_y_pred = multinomialNB_model.predict(X1)

# Evaluation de l'algorithme
print('MultinomialNB Metrics')
accuracy = sklearn.metrics.accuracy_score(y1, mnb_y_pred)
print('Accuracy: %f' % accuracy)
recall = sklearn.metrics.recall_score(y1, mnb_y_pred, average='micro')
print('Recall: %f' % recall)
f1_score = sklearn.metrics.f1_score(y1, mnb_y_pred, average='micro')
print('F1 Score: %f' % f1_score, end='\n\n')

# Affichage de la matrice de confusion
cm = sklearn.metrics.confusion_matrix(y1, mnb_y_pred)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=.5, square=True,cmap='Blues', yticklabels=CATEGORIES, xticklabels=CATEGORIES)
plt.title('Confusion Matrix for MultinomialNB')




