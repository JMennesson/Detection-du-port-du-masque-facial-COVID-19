import time
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Reprise du modèle entraîné par Transfert Learning
model = load_model('data/inceptionV3-model.h5')

# Importation des labels 
with open('data/category2label.pkl', 'rb') as pf:
    category2label = pickle.load(pf)

# Paramétrage
img_size = (100, 100)
colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}

# Taille de la vidéo
WIDTH = 640
HEIGHT = 480

# ............................................................................

# Haarcascade 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ............................................................................

# Lancement de la vidéo 
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0

# Emplacement de la vidéo sauvegardée
out = cv2.VideoWriter('/Users/clair/Desktop/Demonstration.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()      
    if not ret:
        break    
    frame_count += 1     
    frame = cv2.flip(frame, 1)  
    # Détection des visages par Viola et Jones      
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi =  frame[y : y+h, x : x+w]
        data = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), img_size)
        data = data / 255.
        data = data.reshape((1,) + data.shape)
        # Prédiction des visages masqués via le modèle entraîné 
        scores = model.predict(data)
        target = np.argmax(scores, axis=1)[0]
        # Affichage des bounding boxes 
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
        text = "{}: {:.2f}".format(category2label[target], scores[0][target])
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img=frame, text='FPS : ' + str(round(fps, 2)), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)

    # Affichage du résultat sous forme de cadre
    cv2.imshow('Face Mask Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # quitter l'enregistrement en appuyant sur q 
        break
    
# Arrêt et sauvegarde de la vidéo
cap.release()
out.release()
cv2.destroyAllWindows()