import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle


# Chargement du modèle CNN pré-entraîné : InceptionV3
pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
pre_trained_model.summary()


# Conception du réseau de prédiction
for layer in pre_trained_model.layers:
    layer.trainable = False
    
last_layer = pre_trained_model.get_layer('mixed7')
print('Last layer output shape :', last_layer.output_shape)
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x) # fonction d'activation ReLU
x = tf.keras.layers.Dropout(0.3)(x) # couche drop out 
x = tf.keras.layers.Dense(3, activation='softmax')(x) # softmax : proba 

# ............................................................................

# Compilation du modèle 
model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# ............................................................................

# Chargement du dataset
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)


# Normalisation des données
X = X / 255.
X1 = X1 / 255.

y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='int64')
y1 = tf.keras.utils.to_categorical(y1, num_classes=None, dtype='int64')

# ............................................................................

# Entrainement du modèle
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='data/model-{epoch:03d}.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True, 
    verbose=0)

history = model.fit(X, 
                    y, 
                    epochs=20, 
                    callbacks=[checkpoint], 
                    validation_split=0.1)


# Enregistrement du modèle
model.save('inceptionV3-model.h5')

# ............................................................................

# Evaluation du modèle 
model.evaluate(X1, y1)
Y_pred = np.argmax(model.predict(X1), axis=1)
y1 = np.argmax(y1, axis=1)

# Affichage de la matrice de confusion
sns.heatmap(confusion_matrix(y1, Y_pred), annot=True, fmt='g', cmap=plt.cm.Blues)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show() 
