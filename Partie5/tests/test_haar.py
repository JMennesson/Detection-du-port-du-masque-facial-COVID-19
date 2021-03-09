import cv2
import sys
import os
from matplotlib import pyplot as plt


# On effectue les tests sur tous les classifieurs créés (mymaskedfacedetetor 0, 1, 2)
classCascade = cv2.CascadeClassifier('mymaskedfacedetector2.xml')


"""
Premiers tests avec des images venant de différentes sources
Affichage des rectangles pour vérifier la localisation du masque détecté
"""

imagePath = '0-with-mask - Copie.jpg' #modifiable

# Affichage de l'image
image = cv2.imread(imagePath)
plt.imshow(image)
plt.show()

# Detection
faces = classCascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors= 35, #paramètres optimaux donnés ligne 103
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print("Il y a {0} visage(s) masqué(s).".format(len(faces)))



# Dessine des rectangles autour des visages masqués trouvés
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(image)
plt.show()

"""
Recherche du nombre de minNeighbors optimal.
On cherche nimNeighbors tel que le nombre de visages masqués détecté 
soit le plus proche possible du nombre de visage
"""

def Optimal(classifieur_path,data_path):
    classCascade = cv2.CascadeClassifier(classifieur_path)
    files_test = os.listdir(data_path)
    
    minN_list = [i for i in range(5,100)]
    erreur_list = []
    
    minErreur = 100
    minNoptimal = 0
    i=0
    for minN in range(5,100):  #à changer selon le classifieur
        
        erreur = 0
        
        for name in files_test : 
            image = cv2.imread(data_path+'/'+name)
            # Detection
            faces = classCascade.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors= minN, 
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
                )
            
            if len(faces) > 1 : 
                erreur = erreur + len(faces) - 1
            elif len(faces) == 0 : 
                erreur += 1
                
        erreur_list.append(erreur)
        
        if erreur <= minErreur : 
            
            minErreur = erreur
            minNoptimal = minN
    return (minN_list, erreur_list, minErreur, minNoptimal)

# minN_list, erreur_list, minErreur, minNoptimal = Optimal('mymaskedfacedetector2.xml',  'data/positive/positive_test') #paramètres variables

# plt.plot(minN_list, erreur_list)
# plt.show()

"""
Les paramètres optimaux trouvés sont : 
mymaskedfacedetector0.xml -> 3
mymaskedfacedetector1.xml -> 7
mymaskedfacedetector2.xml -> 53
"""

"""
(1) Test sur des images où le masque a été ajouté avec photoshop
('data/positive/positive_test')

(2) Test sur des images de visages masqués avec des masques variés 
(différents de celui utilisé pour l'entraînement)
 ('data/positive/positive_varied_masks_test')
"""

path_test = 'data/positive/positive_varied_masks_test' #modifiable

files_test = os.listdir(path_test)

i = 0
presence_mask = 0
sum_masks = 0
max_masks = []
images_bug = []
names_images_bug = []
for name in files_test :
    
    image = cv2.imread(path_test+'/'+name)
    # Detection
    faces = classCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5, #prendre le minNeighbors optimal selon le classifieur
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
    
    # # Affichage des rectangles
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # plt.imshow(image)
    # plt.show()
    
    # Analyse de la performance
    i+= 1
    sum_masks = sum_masks + len(faces)
    if len(faces)>= 1 : 
        presence_mask += 1
    
    # On cherche les bugs
    if len(faces)> 1 : 
        max_masks.append(len(faces)) #par curiosité on regarde toutes les images 'mal traitrées'
        bug = image
        images_bug.append(bug)
        names_images_bug.append(name)

 
print("Il y a {0} visage(s) masqué(s) détéctés sur {1} images traitées.".format(sum_masks,i))
print("La présence de masque(s) a été observée sur {0} images parmi les sur {1} images traitées.".format(presence_mask,i))
print("Le score du classifieur est donc de {0}".format(presence_mask/i))
print("Les nombres maximum de visages masqués sur une photo sont de {0}.".format(max_masks))

"""
# Affichage des images bug
for image in images_bug :  
    plt.imshow(image)
    plt.show()
"""  

"""
Test sur des images négatives (sans visages masqués)
"""

"""
(3) On en profitera pour remarquer la réaction du classifieur pour des images avec masque seul.
On regarde donc, sur un petit jeu de données,
si le classifieur détecte le masque seul. 
On remarque que le classifieur réagit bien, il ne considère pas l'objet 'masque' seul.
"""
#path_test = 'data/negative/negative_masques_seuls'

"""
(4) On teste ensuite sur des images variées : visages sans masques / fruits / voitures ...
"""

path_test = 'data/negative/negative_test' 


files_test = os.listdir(path_test)

i = 0
presence_mask = 0
sum_masks = 0
max_masks = []
images_bug = []
names_images_bug = []
for name in files_test :
    
    image = cv2.imread(path_test+'/'+name)
    # Detection
    faces = classCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=53,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
    
    # Analyse de la performance
    i+= 1
    sum_masks = sum_masks + len(faces)
    if len(faces)>= 1 : 
        presence_mask += 1
    
    # On cherche les bugs
    if len(faces)>= 1 : 
        max_masks.append(len(faces)) #par curiosité on regarde toutes les images 'mal traitrées'
        bug = image
        images_bug.append(bug)
        names_images_bug.append(name)

print("Il y a {0} visage(s) masqué(s) détéctés sur {1} images traitées.".format(sum_masks,i))
print("La présence de masque(s) a été observée sur {0} images parmi les sur {1} images traitées.".format(presence_mask,i))
print("Le score du classifieur est donc de {0}".format(1 - presence_mask/i))
print("Le nombre maximum de visages masqués sur une photo est de {0}.".format(max_masks))

# #Affichage des images bug
# for name in names_images_bug :      
#     image = cv2.imread(path_test+'/'+name)
#     # Detection
#     faces = classCascade.detectMultiScale(
#         image,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags = cv2.CASCADE_SCALE_IMAGE
#         )
#     #Affichage des rectangles
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     plt.imshow(image)
#     plt.show()
    
