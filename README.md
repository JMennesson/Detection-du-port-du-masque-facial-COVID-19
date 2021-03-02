<img align="right" src="https://github.com/ClaireDel/Projet-P5/blob/main/images/CNN_Demo_cropped.gif" width=400 height=auto>

# Projet-P5
Détection du port du masque facial – COVID19


# Objectif 
Le masque facial, autrefois pratiquement réservé au personnel médical, est devenu incontournable depuis le début de la pandémie de COVID19 pour l’ensemble de la population.
L’objectif de ce projet est de réaliser des algorithmes de détection du port du masque sur les individus. Réalisés en Python, ces algorithmes pourront détecter les viasges masqués sur des photos ou des flux vidéos selon différentes méthodes de traitement d’image et de Deep Learning.

# Démarches
`1. Comparaison des méthodes de détection de visages masqués sur des photos`

Les différentes méthodes d'apprentissage supervisé testées pour la détection de visages masqués (KNN, Decision Tree, Naive Bayes, SVM, CNN) sont présentes au sein du dossier *Detection_Methodes.zip*. 

`2. Utilisation de différentes méthodes de détection de visages masqués sur un flux vidéo`

  `2.1. Détection des visages par Viola Jones et reconnaissance des masques par un modèle de Deep Learning : MobileNetV2`
Le dossier *Detection_Video1_VJ-DL* comprend le code et fichiers nécessaires afin de détecter les visages par une méthode de Viola Jones , et les masques par un modèle de Deep Learning (MobileNetV2) déjà entraîné et prêt à l'emploi.

  `2.2. Détection des visages et reconnaissance des masques par un modèle de Deep Learning : InceptionV3`
Par la suite, nous sommes partis d'un modèle de Deep Learning pré-entraîné, InceptionV3, et avons réalisé un Transfert Learning pour pouvoir l'exploiter sur un des datasets que nous avons choisi. Les scripts python détaillant la préparation des données, l'entraînement et le lancement du flux vidéo sont présents dans le dossier *Detection_Video2_VJ-DL*. 

  `2.3. Détection des visages et reconnaissance des masques par un modèle de Deep Learning : VGG16`
Enfin, nous avons décidé de réaliser à nouveau un Transfert Learning sur un autre modèle pré-entraîné : VGG16. Les scripts sont disponibles dans le dossier *Detection_Video3_VJ-DL*. 

# Conclusion 

`Résultats globaux`


