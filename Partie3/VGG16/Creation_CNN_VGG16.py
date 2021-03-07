from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import keras
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt




# Créer un CNN et l'entraîner totalement est trop coûteux en ressources et ne présente aucun intérêt
# Le plus optimal est de partir d'un modèle modèle pré-construit et pré-entraîné, puis de réaliser du transfer-learning

# On récupère un CNN déjà entraîné (ici VGG16) afin de réaliser du fine-tuning total pour l'adapter à notre problème.
# Pour se faire, on enlève les couches fully connected du réseau (qui servent à classifier selon des catégories de base)
# On crée ensuite les couches qui seront utilisées pour notre classification



# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
 
for layer in model.layers:
   layer.trainable = False


# On récupère la sortie de ce réseau
output_vgg16_conv = model.output


# On ajoute les nouvelles couches fully-connected pour la classification à 3 classes
# On réalise une architecture classique d'un CNN :
# Une couche mise à plat en sortie de l'algorithme de base
# 2 couches de convolution fully connected (avec fonction d'activation ReLU pour chacune d'elles) 
# Une couche fully connected de classification
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
x = Dense(3, activation='softmax')(x)


# On définit le nouveau modèle
CNN = keras.models.Model(inputs=model.input, outputs=x)

# On compile le modèle 
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])






# On charge les données d'entraînement générées
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


# Affichage des caractéristiques des jeux de données
print('# of Samples:', len(y))
print('# of Without A Mask:', (y == 0).sum())
print('# of With A Mask:', (y == 1).sum())
print('# of With An Incorrectly Worn Mask:', (y == 2).sum())


# Affichage test d'une image
plt.imshow(X[5014])
plt.show()
   
# On normalise les valeurs des pixels
X = X / 255

# Transformation de y nécessaire pour l'entraînement
y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')

# Entraînement sur les données X et y
history = CNN.fit(X, y, epochs=20, verbose=1)


# Affichage des courbes de précision et de perte
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Précision et perte du modèle')
plt.ylabel('Précision/Perte')
plt.xlabel('epoch')
plt.legend(['Précision', 'Perte'], loc='right')
plt.show()




# Sauvegarde du modèle entraîné
CNN.save(XXX.h5)
