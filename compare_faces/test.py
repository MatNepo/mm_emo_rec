import cv2
import torch
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Конфигурация
YOLO_MODEL_PATH = "source/fed/weights/yolov8l-face.pt"
INSIGHT_MODEL_DIR = "./weights/buffalo_l"
SIMILARITY_THRESHOLD = 0.5

# Загрузка моделей
yolo_model = YOLO(YOLO_MODEL_PATH)
insight_app = FaceAnalysis(name=INSIGHT_MODEL_DIR, providers=['CPUExecutionProvider'])
insight_app.prepare(ctx_id=-1)  # -1 = CPU, 0 = GPU

# Хранилище эмбеддингов и ID
face_db = []  # [(embedding, id)]
next_id = 0

# Инициализация камеры
cap = cv2.VideoCapture(0)

def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция лиц с помощью YOLO
    results = yolo_model(frame)[0]

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]

        # Получение эмбеддинга через InsightFace
        faces = insight_app.get(face_crop)
        if not faces:
            continue

        emb = faces[0].embedding

        # Поиск в базе
        matched_id = None
        for saved_emb, person_id in face_db:
            if compute_similarity(saved_emb, emb) > SIMILARITY_THRESHOLD:
                matched_id = person_id
                break

        global next_id
        if matched_id is None:
            matched_id = next_id
            face_db.append((emb, matched_id))
            next_id += 1

        # Отрисовка результата
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
