import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Modèle CNN pré-entraîné InceptionV3
pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
pre_trained_model.summary()


# Blocage de toutes les couches sauf la couche fully-connected
for layer in pre_trained_model.layers:
    layer.trainable = False
    
last_layer = pre_trained_model.get_layer('mixed7')
print('Last layer output shape :', last_layer.output_shape)
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x) # fonction d'activation ReLU
x = tf.keras.layers.Dropout(0.3)(x) # couche drop out 
#La dernière couche avec 3 sorties pour les 3 catégories
x = tf.keras.layers.Dense(3, activation='softmax')(x) # softmax : proba 

# ............................................................................

# Création du modèle
model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# ............................................................................

# Chargement des données pour le réapprentissage
X = np.load('data/X.npy')
Y = np.load('data/Y.npy')
print(X.shape, Y.shape)

# ............................................................................

# Affichage de la répartition des images en fonction de leur classe
ax = sns.countplot(np.argmax(Y, axis=1), palette="Set1", alpha=0.8)
ax.set_xticklabels(['without_mask', 'with_mask', 'mask_weared_incorrect'], rotation=30, ha="right", fontsize=15)
plt.show()

# ............................................................................

# Normalisation des données
X = X / 255.


# Division des données en 2 paquets : un pour l'entrainement et un pour la phase de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# ............................................................................

# Entrainement 
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='data/model-{epoch:03d}.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True, 
    verbose=0)

history = model.fit(X_train, 
                    Y_train, 
                    epochs=20, 
                    callbacks=[checkpoint], 
                    validation_split=0.1)

# ............................................................................

# Affichage de la précision du modèle
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(122)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# ............................................................................

# Enregistrement du modèle
model.save('data/inceptionV3-model.h5')

# ............................................................................

# Phase de test
model.evaluate(X_test, Y_test)
Y_pred = np.argmax(model.predict(X_test), axis=1)
Y_test = np.argmax(Y_test, axis=1)

# ............................................................................

# Affichage de la matrice de confusion
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt='g', cmap=plt.cm.Blues)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show() 