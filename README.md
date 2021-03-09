# Projet-P5
Détection du port du masque facial – COVID19  

<p align="center">
  <img src="https://github.com/ClaireDel/Projet-P5/blob/main/images/CNN_Demo.gif" width=400 height=auto>
</p>

# Objectif 
Le masque facial, autrefois pratiquement réservé au personnel médical, est devenu incontournable depuis le début de la pandémie de COVID19 pour l’ensemble de la population.
L’objectif de ce projet est de réaliser des algorithmes de détection du port du masque sur les individus. Réalisés en Python, ces algorithmes pourront détecter les viasges masqués sur des photos ou des flux vidéos selon différentes méthodes de traitement d’image et de Deep Learning.


# Démarches
`1. Présentation des données`

Le dataset utilisé pour réaliser l’entrainement des différents modèles qui vont suivre contient 15 284 images en couleur et de taille 64x64 appartenant à 3 classes différentes :   With Mask / Without Mask / Mask Incorrectly Worn. 

Il est accessible via le lien Drive : https://drive.google.com/drive/folders/1e70k-LQBAeDUumunceQqOAkQc23dsvHX 

`2. Etudes préliminaires`

L’enjeu de cette partie est d’illustrer les performances de différents modèles de classifieurs d’images de visages masqués par apprentissage supervisé : KNN, Decision Tree, Naive Bayes, SVM. 

`3. Réseau de neurones convolutif (CNN)`

Pour l'étude des CNN, nous nous sommes appuyés sur les modèles InceptionV3 et VGG16. 

`4. Implémentation sur flux vidéo`

L’objectif de cette partie est de lancer un flux vidéo et d'être capable d’afficher les prédictions du modèle VGG16 sur la vidéo étudiée en temps réel. 

`5. Utilisation d'un classifieur Viola et Jones`

Nous avons enfin tentés de créer notre propre classifieur Viola et Jones pour la détection de l'objet 'visage masqué'. 

Le dataset utilisé dans cette partie est accessible via le lien suivant : https://drive.google.com/drive/folders/1ZeTTiteR38-rl9l4GNWzWim8tntGgHVi




