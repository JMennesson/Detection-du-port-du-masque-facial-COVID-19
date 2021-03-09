import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
import keras


# Ce programme est destiné à determiner via cross validation les paramètres optimaux pour éviter l'overfitting
# Le praramètre choisi pour évaluer l'overfitting est l'erreur quadratique moyenne
# Le test est réalisé avec 1500 données (le jeu de test)


# Chargement du dataset
pickle_in = open("X1.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y = pickle.load(pickle_in)
pickle_in = open("data1.pickle", "rb")
data = pickle.load(pickle_in)

# Normalisation des données
X = X / 255
y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')

# Création du réseau 
model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
for layer in model.layers:
   layer.trainable = False

output_vgg16_conv = model.output

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
x = Dense(3, activation='softmax')(x)
CNN = keras.models.Model(inputs=model.input, outputs=x)
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Vérification d'overfitting via cross validation
def verif_over(model):
    kf = KFold(n_splits=20)
    list_training_error = []
    list_testing_error = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_train_data_pred = model.predict(x_train)
        y_test_data_pred = model.predict(x_test)
    
    
        fold_training_error = np.sqrt(mean_squared_error(y_train, y_train_data_pred)) 
        fold_testing_error = np.sqrt(mean_squared_error(y_test, y_test_data_pred))
        list_training_error.append(fold_training_error)
        list_testing_error.append(fold_testing_error)
    
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), c= 'blue')
    plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), c='red')
    plt.xlabel('number of fold')
    plt.ylabel('testing / training error')
    plt.title('Testing and training error across folds')
    plt.legend(['Train', 'Test'], loc='right')
    plt.tight_layout()
    plt.show()
