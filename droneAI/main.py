from ultralytics import YOLO
import cv2

# φορτώνεις pretrained μοντέλο
model = YOLO("yolov8n.pt")

# ανοίγεις webcam (0 = laptop camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    # βάζει αυτόματα bounding boxes + labels
    annotated = results[0].plot()

    # εμφανίζει εικόνα
    cv2.imshow("YOLO Camera", annotated)

    # ESC για έξοδο
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()