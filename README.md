<img width="1470" height="929" alt="Screenshot 2026-04-20 111455" src="https://github.com/user-attachments/assets/ac97d0cf-59f0-4067-a72d-a804a32a89ff" />
# Drone-Vehicle-Tracking-Speed-Estimation

# Introduction Générale
Ce projet présente une solution intégrée pour la surveillance du trafic routier, combinant l'Intelligence Artificielle et la géométrie de vision. L'objectif est d'extraire des données métriques (vitesse et trajectoire) à partir d'un flux vidéo capturé à 40 mètres d'altitude. Le système repose sur trois piliers : la détection sémantique, l'analyse temporelle du mouvement et la transformation géométrique.

# 1-Architecture Logicielle Modulaire
Pour une structure professionnelle, le projet a été décomposé en trois modules sources :

-detec_and_tracking.py : Focus sur l'inférence du modèle YOLOv8 et la persistance des IDs.
-optical_flow.py : Analyse granulaire du mouvement pixel par pixel via l'algorithme de Farneback, visualisée dans l'espace colorimétrique HSV.
-My_project.py : Le cerveau intégré fusionnant la détection, la correction de distorsion MATLAB et le calcul de télémétrie en temps réel.

# 2.	Détection et Suivi par IA
# 2.1 Choix Stratégique du Modèle : YOLOv8 Nano (yolov8n)

Le choix de l'architecture YOLOv8 (You Only Look Once) dans sa version Nano n'est pas arbitraire ; il repose sur des impératifs de performance en temps réel :
•	Optimisation de l'Inférence : YOLOv8n utilise une architecture de "backbone" légère qui réduit le nombre de paramètres de calcul. Dans notre système de télémétrie, la vitesse est calculée par le déplacement entre deux frames. Un modèle trop lourd (comme YOLOv8x) introduirait une latence, décalant le calcul par rapport à la position réelle. Avec une inférence de <15ms, nous garantissons une synchronisation parfaite.
•	Analyse Multi-échelle : YOLOv8 excelle dans la détection d'objets de petite taille (véhicules vus de haut) grâce à son mécanisme de Feature Pyramid Network (FPN), qui permet d'extraire des caractéristiques sémantiques à différentes résolutions.

# 2.2. L'Algorithme BoT-SORT : Pourquoi cette supériorité ?

Pour le suivi (Tracking), nous avons implémenté BoT-SORT (Base Object Tracker with SORT), qui représente l'état de l'art actuel. Son choix se justifie par deux mécanismes absents des trackers classiques (comme SORT ou ByteTrack) :
1.	GMC (Global Motion Compensation) : Lorsqu'un drone filme, il subit des micro-vibrations ou des dérives dues au vent. Pour un tracker classique, ce mouvement de caméra est interprété comme un mouvement de l'objet, ce qui fausse la vitesse. BoT-SORT utilise l'extraction de points d'intérêt (Keypoints) sur l'arrière-plan pour compenser ce mouvement global. Ainsi, seule la vitesse propre du véhicule est mesurée.
2.	Filtre de Kalman Amélioré : BoT-SORT utilise un Filtre de Kalman pour prédire la position future du véhicule basée sur sa vitesse et sa direction actuelles. Si l'IA perd la détection pendant 1 ou 2 frames (à cause d'un obstacle ou d'un reflet), le filtre de Kalman "maintient" la trajectoire de manière prédictive. Cela évite la rupture de l'ID et garantit une courbe de vitesse continue et sans sauts brusques.

# 2.3. Paramétrage Technique : Seuil de Confiance (conf=0.45)

Le coefficient de confiance agit comme un filtre de précision. Nous avons fixé ce seuil à 0.45 pour établir un équilibre :
•	Filtrage du Bruit : Éviter que des éléments statiques (ombres, marquages au sol, mobilier urbain) ne soient interprétés comme des véhicules.
•	Rétention d'Objets : Maintenir une sensibilité suffisante pour détecter les voitures de couleur sombre ou celles situées en bordure de champ, là où la distorsion est la plus forte.
<img width="1417" height="961" alt="Screenshot 2026-04-20 111424" src="https://github.com/user-attachments/assets/2820936c-8dd4-495b-b3cc-efc532a2c77d" />

<img width="1470" height="929" alt="Screenshot 2026-04-20 111455" src="https://github.com/user-attachments/assets/7ce30483-e2af-41a1-b1e3-d8885eb47082" />


# 3.	Analyse du Flux Optique

# 3.1. Pourquoi l'algorithme de Farneback ?
Contrairement aux méthodes de flux optique "clairsemé" (comme Lucas-Kanade) qui ne suivent que quelques points clés, l'algorithme de Farneback est une méthode "Dense".
•	Analyse Totale : Il calcule le vecteur de déplacement pour chaque pixel de l'image.
•	Raison du choix : Dans un contexte de drone à 40m d'altitude, les véhicules peuvent paraître petits. Une analyse dense garantit que même les changements subtils de position sont captés, offrant une signature de mouvement complète pour chaque objet détecté par l'IA.

# 3.2. L'espace colorimétrique HSV : Une visualisation sémantique

Le flux optique génère des vecteurs de mouvement (coordonnées x, y). Pour rendre ces données interprétables par l'œil humain et par le système, nous les convertissons dans l'espace HSV (Teinte, Saturation, Valeur) :
•	La Teinte (Hue - Direction) : Chaque angle de mouvement est associé à une couleur spécifique (ex: le rouge pour le mouvement vers la droite, le bleu vers la gauche). Cela permet de vérifier instantanément si le véhicule suit une trajectoire rectiligne conforme à la route.
•	La Valeur (Value - Intensité) : La luminosité de la couleur représente la magnitude (vitesse en pixels). Plus la couleur est vive, plus l'objet se déplace rapidement.
•	Utilité : Cela permet de distinguer visuellement le mouvement des véhicules du mouvement résiduel de la caméra.

# 3.3. Filtrage du Bruit (mag < 1.0) : La Robustesse du Système

L'un des plus grands défis de la vision par drone est le "bruit" environnemental.
•	Problématique : Le vent dans l'herbe ou le balancement des arbres crée des micro-mouvements de pixels que l'algorithme détecte comme du flux.
•	Solution Technique : Nous appliquons un seuillage rigoureux. Si la magnitude du vecteur est inférieure à 1.0 pixel, elle est considérée comme du bruit de fond et mise à zéro.
•	Bénéfice : Cette étape "nettoie" la scène, ne laissant apparaître que les flux nets et puissants générés par les véhicules. Cela garantit que le système de calcul de vitesse ne traite que des données cinématiques pertinentes.

<img width="1826" height="974" alt="Screenshot 2026-04-20 111802" src="https://github.com/user-attachments/assets/baa8201c-22e7-4bd1-a480-213dc18bb4d1" />


<img width="1915" height="1003" alt="Screenshot 2026-04-20 111819" src="https://github.com/user-attachments/assets/c27f120b-54d6-4a5b-a9d7-0bd80bb5282b" />

# 4.	Calibration MATLAB et Paramètres Optiques

# 4.1. Pourquoi la calibration avec MATLAB ?

Une caméra de drone n'est pas un capteur parfait. La Toolbox Camera Calibrator de MATLAB nous a permis d'extraire l'ADN optique de la caméra. Sans cette étape, toute mesure de vitesse serait purement estimative et non scientifique.
Cette calibration répond à deux problèmes majeurs :

# 4.2. La Matrice Intrinsèque (K) : La clé de la métrologie

La matrice  K est le pont entre le monde réel (3D) et l'image (2D).
•	La Focale (f = 1844.4 px) : C'est la donnée la plus critique. Elle définit comment la lumière converge sur le capteur.
•	Utilité : Dans le calcul de vitesse, nous devons savoir combien de "mètres" représente un "pixel". Cette conversion est directement proportionnelle à la focale. Si la focale est fausse, la vitesse calculée sera systématiquement erronée (par exemple, afficher 80 km/h au lieu de 60 km/h).

# 4.3. Les Coefficients de Distorsion (DIST) : Redresser la réalité

Les lentilles grand-angle utilisées sur les drones provoquent une distorsion radiale (effet "Radial Distortion" ou "Fisheye").
•	Le problème : Les lignes droites (comme les bords d'une route) apparaissent courbes sur l'image brute. Plus un objet est loin du centre de l'image, plus il est "étiré" ou "compressé" par la lentille.
•	La Solution (cv2.undistort) : En utilisant les coefficients de distorsion calculés par MATLAB, nous appliquons une transformation mathématique inverse pour "redresser" chaque pixel.
•	Bénéfice : Cela garantit que la distance entre deux pixels est constante sur toute la surface de l'image. Sans cela, une voiture roulant à vitesse constante semblerait accélérer en s'approchant des bords de la vidéo.


<img width="1917" height="1016" alt="Screenshot 2026-04-16 195117" src="https://github.com/user-attachments/assets/08ecde6c-3d0f-4fd4-9367-52eab734e441" />

<img width="1406" height="998" alt="Screenshot 2026-04-16 195244" src="https://github.com/user-attachments/assets/b0618388-7d6e-46d6-850b-27de39c84fb0" />



# 5.	Lois Physiques et Défis Techniques

# 5.1. Logique de Calcul de la Vitesse : De l'Espace Image à l'Espace Réel

Le cœur de notre système de télémétrie repose sur la transformation des déplacements de pixels en unités métriques. Cette conversion n'est possible qu'en modélisant la projection de la lumière à travers la lentille.

A. Le Modèle Sténopé (Pinhole Camera Model)

Nous considérons que la caméra du drone suit le modèle sténopé, où chaque point dans l'espace 3D est projeté sur un plan image 2D. En utilisant le principe de la similarité des triangles, nous établissons un rapport constant entre l'altitude du drone (H), la focale de la caméra (f) et la taille des objets.

B. La Ground Sample Distance (GSD)

La GSD représente la distance réelle au sol couverte par un seul pixel sur le capteur. C'est l'unité de mesure fondamentale de notre projet.
GSD = Altitude (40m) / Focale (1844.4 px) = 0.0216 m/pixel.

Signification : Chaque pixel sur l'écran correspond à environ 2,16 cm sur la route. Sans la calibration MATLAB (qui nous a donné la focale exacte), cette valeur serait erronée, rendant toute estimation de vitesse invalide.

# C. Équation Finale de la Vitesse
Pour calculer la vitesse entre deux images successives (Frame n et Frame n+1) :
1.	Déplacement Pixel (delta P) : On utilise le théorème de Pythagore entre les coordonnées du centre du véhicule :
   Delta P = sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}
2.	Distance Réelle (D) : D = Delta P x GSD (résultat en mètres).
3.	Vitesse (V) : Comme le drone capture 30 images par seconde (30 FPS), le temps entre deux images est de 1/30 s.
V{km/h} = (D x FPS) x3.6
Le coefficient 3.6 est le facteur de conversion standard entre le mètre par seconde (m/s) et le kilomètre par heure (km/h).

# 6. Défis Techniques : Détection à Longue Distance et Stabilisation

# 6.1. Problématique de la Résolution Spatiale

Un défi majeur est survenu : lorsque les véhicules s'éloignent vers l'horizon, ils occupent une surface binaire de plus en plus petite (souvent moins de 10 x 10 pixels).
•	Conséquence : Le modèle YOLOv8 perd les "caractéristiques sémantiques" (roues, phares, vitres). La "confiance" du modèle chute alors en dessous de notre seuil de 0.45, provoquant la disparition de l'ID et de l'indicateur de vitesse.

# 6.2. Solutions et Optimisations Apportées

Pour pallier ces limites physiques, deux solutions stratégiques ont été implémentées :
1.	Augmentation de la Résolution d'Inférence (imgsz=1280) :
En doublant la résolution d'entrée par rapport au standard (640), nous permettons au réseau de neurones de conserver des détails exploitables sur les objets lointains. Cela améliore la rétention de l'ID à longue distance, bien que cela nécessite une puissance de calcul GPU plus élevée.
2.	Lissage par Filtre à Moyenne Mobile (Moving Average) :
À grande distance, un "tremblement" (Jitter) de seulement 1 pixel dans la détection peut provoquer une erreur de vitesse de 10 km/h.
o	Mécanisme : Nous utilisons un buffer deque de 20 frames. Le système calcule la moyenne pondérée de ces 20 dernières mesures.
o	Résultat : Cela "gomme" les erreurs de mesure instantanées et fournit une courbe de vitesse stable et fluide, indispensable pour une application de surveillance crédible.

<img width="625" height="479" alt="image" src="https://github.com/user-attachments/assets/b43c53e1-0b21-4c53-9001-a636021cc3f1" />


<img width="1411" height="837" alt="Screenshot 2026-04-20 112804" src="https://github.com/user-attachments/assets/23e5a3c9-fd2c-47d2-856e-8d89a3e6b18b" />

# Conclusion

Ce projet démontre une intégration réussie entre l'IA moderne et la vision géométrique classique. Le système est robuste, capable de corriger les défauts optiques et de fournir des données télémétriques fiables, faisant de ce drone un véritable outil de mesure radar automatisé.




  	




