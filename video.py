import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from deepface import DeepFace
from tqdm import tqdm
import tensorflow as tf
import os
import subprocess

def get_gpu_memory():
    """Получить информацию об использовании памяти GPU"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        used, total = map(int, result.strip().split(','))
        return used, total
    except:
        return None, None

# Проверка доступности GPU
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("CUDA Available:", tf.test.is_built_with_cuda())

# Проверка памяти GPU
used, total = get_gpu_memory()
if used is not None:
    print(f"\nGPU Memory: {used}MB used of {total}MB total")

# Проверка переменных окружения CUDA
print("\nCUDA Environment Variables:")
print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))

# Настройка GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\nGPU memory growth enabled")
        print("Available GPUs:", gpus)
    else:
        print("\nNo GPU devices found. Running on CPU.")
except Exception as e:
    print("\nError configuring GPU:", e)

# Проверка устройства по умолчанию
print("\nDefault device:", tf.config.get_visible_devices())

def analyze_video_emotions(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Ошибка открытия видео файла")

    emotions_data = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Обработка видео") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], silent=True)
                    emotions = analysis[0]['emotion']
                    emotions['time'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    emotions_data.append(emotions)
                except:
                    pass

            frame_count += 1
            pbar.update(1)

    cap.release()
    return pd.DataFrame(emotions_data)

def process_video_with_emotions(input_path, output_path, frame_interval=1):
    """
    Обрабатывает видео, определяет эмоции всех людей в кадре и сохраняет результат
    с отображением эмоций в реальном времени.
    
    Args:
        input_path (str): Путь к входному видео
        output_path (str): Путь для сохранения обработанного видео
        frame_interval (int): Интервал между кадрами для обработки
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Ошибка открытия видео файла")

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создаем VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    emotions_data = []

    with tqdm(total=total_frames, desc="Обработка видео") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    # Анализируем все лица в кадре
                    results = DeepFace.analyze(frame, actions=['emotion'], 
                                            detector_backend='retinaface',
                                            enforce_detection=False,
                                            silent=True)
                    
                    # Если обнаружено несколько лиц
                    if not isinstance(results, list):
                        results = [results]

                    # Обрабатываем каждое обнаруженное лицо
                    for result in results:
                        emotions = result['emotion']
                        region = result['region']
                        
                        # Получаем координаты лица
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        
                        # Находим доминирующую эмоцию
                        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                        emotion_value = emotions[dominant_emotion]
                        
                        # Рисуем рамку вокруг лица
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Добавляем текст с эмоцией
                        text = f"{dominant_emotion}: {emotion_value:.1f}%"
                        cv2.putText(frame, text, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                  (0, 255, 0), 2)
                        
                        # Сохраняем данные об эмоциях
                        emotions['time'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        emotions['face_id'] = f"face_{x}_{y}"  # Уникальный идентификатор лица
                        emotions_data.append(emotions)

                except Exception as e:
                    print(f"Ошибка при обработке кадра: {e}")
                    pass

            # Сохраняем кадр
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    
    # Создаем DataFrame с данными об эмоциях
    df_emotions = pd.DataFrame(emotions_data)
    return df_emotions

def plot_emotions(df, window_size=5):
    plt.figure(figsize=(18, 10))

    # Яркие контрастные цвета для каждой эмоции
    emotion_colors = {
        'angry': '#FF0000',
        'disgust': '#800080',
        'fear': '#000000',
        'happy': '#00FF00',
        'sad': '#0000FF',
        'surprise': '#FFA500',
        'neutral': '#808080'
    }

    # Сглаживание с помощью скользящего среднего
    df_smooth = df.copy()
    for emotion in emotion_colors.keys():
        df_smooth[emotion] = df[emotion].rolling(window=window_size, min_periods=1).mean()

    # Рисуем сглаженные графики
    for emotion, color in emotion_colors.items():
        plt.plot(df_smooth['time'], df_smooth[emotion],
                color=color,
                linewidth=2,
                alpha=0.8,
                label=emotion.capitalize())

    plt.title('Динамика эмоций в видео', fontsize=16, pad=20)
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Интенсивность (%)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.2)

    # Рассчет процентного соотношения эмоций
    total_power = df.drop('time', axis=1).sum().sum()
    emotion_percent = (df.drop('time', axis=1).sum() / total_power * 100).sort_values(ascending=False)
    top_emotions = emotion_percent.head(3)

    # Форматирование текста для топ-3
    top_text = "Топ-3 эмоции:\n" + "\n".join(
        [f"{i+1}. {emotion}: {percent:.1f}%"
         for i, (emotion, percent) in enumerate(top_emotions.items())]
    )

    plt.figtext(0.5, -0.1, top_text,
               ha="center",
               fontsize=14,
               bbox={"facecolor":"#f0f0f0", "alpha":0.8, "pad":10},
               fontweight='bold')

    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    input_video = "./fed/test_emotions.mp4"
    output_video = "output_emotions.mp4"
    
    # Обработка видео и сохранение результата
    df_emotions = process_video_with_emotions(input_video, output_video)
    
    # Построение графика эмоций
    plot_emotions(df_emotions)