import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================================
# 1. Configuration de la Géométrie de la Caméra
# ==========================================================
FOCAL_LENGTH = 1844.4  # Obtenu via les résultats de Matlab
ALTITUDE = 40.0        # Altitude du drone en mètres
FPS = 30.0             # Images par seconde (Frames Per Second)

# Matrice de la caméra et coefficients de distorsion (issus de Matlab)
K = np.array([[1844.4, 0, 2042.8], [0, 1838.2, 1347.3], [0, 0, 1]], dtype=np.float32)
DIST = np.array([-0.3426, 0.2996, 0, 0, 0], dtype=np.float32)

def estimate_speed(pixel_dist, altitude, focal_length, fps):
    """Calcul de la vitesse en utilisant la géométrie de la caméra (Modèle Pinhole)"""
    meters_per_pixel = altitude / focal_length
    speed_kmh = (pixel_dist * meters_per_pixel * fps) * 3.6
    return speed_kmh

# ==========================================================
# 2. Initialisation des Modèles et Données
# ==========================================================
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture("0322(2).mp4")

track_history = {}
# Buffer pour le lissage de la vitesse (élimination des fluctuations numériques)
speed_buffer = {} 

# Préparation pour le flux optique (Optical Flow)
ret, frame = cap.read()
if ret:
    frame = cv2.undistort(frame, K, DIST)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ==========================================================
# 3. Boucle de Traitement Intégrée
# ==========================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # A. Correction de l'image (Undistortion) - Exigence géométrique
    frame = cv2.undistort(frame, K, DIST)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # B. Étape 2 : Calcul et représentation du Flux Optique (Optical Flow)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Dessiner les lignes de flux optique (légèrement pour ne pas obstruer la vue)
    step = 20
    h, w = current_gray.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    for i in range(len(x)):
        if abs(fx[i]) > 2 or abs(fy[i]) > 2: # Dessiner uniquement les mouvements significatifs
            cv2.line(frame, (x[i], y[i]), (int(x[i]+fx[i]), int(y[i]+fy[i])), (255, 0, 0), 1)

    # C. Étape 1 : Détection et Suivi par IA (YOLOv8 + BoT-SORT)
    results = model.track(frame, persist=True, imgsz=640, conf=0.45, tracker="botsort.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h_box = box
            
            # D. Étape 3 : Calcul de la vitesse basé sur la géométrie de vision
            if track_id in track_history:
                prev_pos = track_history[track_id]
                pixel_move = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                
                raw_speed = estimate_speed(pixel_move, ALTITUDE, FOCAL_LENGTH, FPS)
                
                # --- Lissage de la vitesse (Speed Smoothing) ---
                if track_id not in speed_buffer:
                    speed_buffer[track_id] = deque(maxlen=15) # Moyenne sur les 15 dernières frames
                speed_buffer[track_id].append(raw_speed)
                smoothed_speed = sum(speed_buffer[track_id]) / len(speed_buffer[track_id])

                # Visualisation des résultats
                cv2.rectangle(frame, (int(x-w/2), int(y-h_box/2)), (int(x+w/2), int(y+h_box/2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} | {int(smoothed_speed)} km/h", 
                            (int(x-w/2), int(y-h_box/2-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            track_history[track_id] = (x, y)

    # Affichage de la vidéo finale
    prev_gray = current_gray
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Suivi IA + Flux Optique + Geometrie Camera", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()