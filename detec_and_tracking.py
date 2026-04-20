from ultralytics import YOLO
import cv2

# Chargement du modèle YOLOv8n (Version légère pour la performance)
model = YOLO('yolov8n.pt')

video_path = "0322(2).mp4" 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
    # Détection et Suivi avec BoT-SORT
    # imgsz=640 pour l'équilibre entre précision et vitesse
    # conf=0.45 pour filtrer les détections incertaines
        results = model.track(
            source=frame, 
            persist=True, 
            imgsz=1280, 
            conf=0.45, 
            iou=0.5,
            tracker="botsort.yaml" 
        )
    #small_image
        annotated_frame = results[0].plot()
        display_frame = cv2.resize(annotated_frame, (1280, 720))

        cv2.imshow("Advanced Drone Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()