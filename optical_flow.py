import cv2
import numpy as np

# 1. Chargement de la vidéo
cap = cv2.VideoCapture("0322(2).mp4")

# Lecture du premier cadre et préparation
ret, frame1 = cap.read()
if not ret:
    print("Erreur lors du chargement de la vidéo")
    exit()

# Redimensionnement de l'image pour accélérer le traitement et améliorer l'affichage
frame1 = cv2.resize(frame1, (960, 540))
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Création d'une matrice HSV pour la colorisation du mouvement
# H (Hue - Teinte) : Représente la direction du mouvement (la couleur)
# S (Saturation) : Définie à la valeur maximale (255)
# V (Value - Valeur) : Représente l'intensité du mouvement (la luminosité)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    frame2_resized = cv2.resize(frame2, (960, 540))
    next = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # 2. Calcul du Flux Optique Dense (Algorithme de Farneback)
    # Cet algorithme calcule le déplacement de chaque pixel entre deux images successives
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Conversion des vecteurs en coordonnées polaires (Magnitude et Angle)
    # Magnitude = Force du mouvement | Angle = Direction du mouvement
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 3. --- Filtrage du bruit (Nettoyage des micro-mouvements) ---
    # Tout pixel dont la force de mouvement est inférieure à 1.0 (ex: vent dans les arbres) 
    # sera mis à zéro (devient noir) pour isoler les véhicules
    mag[mag < 1.0] = 0 

    # Colorisation de la direction (Direction vers Couleur)
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Conversion de la magnitude en luminosité (Magnitude vers Intensité lumineuse)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Conversion du format HSV vers BGR pour l'affichage
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Affichage des résultats
    cv2.imshow('Video Originale', frame2_resized)
    cv2.imshow('Flux Optique Filtre (Carte Directionnelle)', bgr_flow)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    # Mise à jour de l'image précédente pour l'itération suivante
    prvs = next

cap.release()
cv2.destroyAllWindows()