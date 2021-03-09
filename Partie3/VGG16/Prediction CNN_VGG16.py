import numpy as np
import cv2
import pickle
import tensorflow as tf
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


CNN = tf.keras.models.load_model('XXX')

# Chargement des données générées précedemment
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)
pickle_in = open("data1.pickle", "rb")
data1 = pickle.load(pickle_in)



# On normalise les valeurs de pixels
X1 = X1 / 255


# On transforme y1 pour l'utilisation
y1 = tf.keras.utils.to_categorical(y1, num_classes=None, dtype='int64')

# Prédiction
y_pred = CNN.predict(X1)
y_pred1 = np.argmax(y_pred,axis=1)
y2 = np.argmax(y1,axis=1)

# On crée un compteur d'erreurs
E = []
e = 0
for k in range(len(y1)) :
    E.append(y2[k]-y_pred1[k])
for k in range(len(E)) :
    if E[k] != 0 : 
        e += 1
        
print("Prédiction :")
print("Nombre d'erreurs : ", e, "sur", len(y1), ' ', "(" + str(round(e/len(E)*100, 3)) + "%" + ")")


# Affichage de la matrice de confusion
cm = sklearn.metrics.confusion_matrix(y2, y_pred1)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Matrice de Confusion')
plt.show()

    
# On peut aussi tester l'algorithme sur une image quelconque 
def test_image() :  
    image = cv2.imread('XXX.jpg')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image = cv2.resize(image, (64,64))
    image = np.array(image).reshape(-1, 64, 64, 3)
    y_pred = np.argmax(CNN.predict(image))
    if y_pred == 0 :
        print("L'individu ne porte pas de masque")
    if y_pred == 1 :
        print("L'individu porte un masque")
    if y_pred == 2 : 
        print("Lindividu porte mal son masque")
