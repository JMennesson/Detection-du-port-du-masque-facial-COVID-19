import cv2
from tensorflow.compat.v1.keras.preprocessing.image import img_to_array
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt



faceCascade = cv2.CascadeClassifier('XXX.xml')
model = load_model('XXX.h5')

WIDTH = 640
HEIGHT = 480

video_capture = cv2.VideoCapture(0)

# Enregistrement de la vidéo
out = cv2.VideoWriter('XXX.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5., (WIDTH,HEIGHT))




while True:

    ret, frame = video_capture.read()
    faces = faceCascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)    
    faces_list=[]
    faces_list = np.array(faces_list).reshape(-1, 64, 64, 3)
    preds=[]
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.resize(face_frame, (64, 64))
        plt.imshow(face_frame)
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)
        # faces_list.append(face_frame)
        faces_list = np.append(faces_list, face_frame, axis=0)
        if len(faces_list)>0:
            preds = model.predict(faces_list)
        for pred in preds:
            (withoutMask, mask, incorrectMask) = pred
            if mask > withoutMask and mask > incorrectMask :
                label = "Mask"
            elif withoutMask > mask and withoutMask > incorrectMask :
                label = "NoMask"
            elif incorrectMask > mask and incorrectMask > withoutMask :
                label = "IncorrectMask"
            if label == "Mask" :
                color = (0, 255, 0) 
            if label == "IncorrectMask" :
                color = (0, 255, 255)
            if label == "NoMask" :
                color = (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(withoutMask, mask, incorrectMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
    
        
    cv2.imshow('Video', frame) 
    out.write(frame)
    
    
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
        
video_capture.release()
out.release()
cv2.destroyAllWindows()
