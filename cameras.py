import cv2
import time
import os
from datetime import datetime

# ВВЕДИ СВОЙ АДРЕС ПОТОКА ЗДЕСЬ
camera_url = "rtsp://rtsp:rtsp195030@10.150.81.105:554"

# Папка для сохранения кадров
output_folder = "saved_frames"
os.makedirs(output_folder, exist_ok=True)

print(f"Пытаемся открыть поток: {camera_url}")

video_capture = None

# Пытаемся открыть поток с повторными попытками
while True:
    video_capture = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    if video_capture.isOpened():
        print("Поток успешно открыт.")
        break
    else:
        print("Не удалось открыть поток. Повторная попытка через 5 секунд...")
        time.sleep(5)

print("Сохраняем кадры...")

frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Не удалось получить кадр. Повторная попытка через 1 сек...")
        time.sleep(1)
        continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
    cv2.imwrite(frame_filename, frame)

    print(f"Сохранён кадр: {frame_filename}")

    frame_count += 1

    # Для теста ограничим до 10 кадров
    if frame_count >= 10:
        print("Достигнут лимит 10 кадров. Завершение.")
        break

    time.sleep(1)  # Сохраняем 1 кадр в секунду

video_capture.release()
print("Завершено.")
