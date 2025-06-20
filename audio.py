import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_audio_emotions(audio_path, chunk_size=1.0):
    """Анализ эмоций в аудио с временной разметкой"""
    # Загрузка аудио
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    
    emotions_data = []
    
    # Создаем прогрессбар
    with tqdm(total=int(duration), desc="Анализ аудио") as pbar:
        for start_time in np.arange(0, duration, chunk_size):
            end_time = start_time + chunk_size
            chunk = y[int(start_time*sr):int(end_time*sr)]
            
            # Получаем предсказание эмоций
            inputs = feature_extractor(
                chunk, 
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                logits = model_(**inputs).logits
            
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            
            # Формируем запись
            emotions = {config.id2label[i]: float(score) for i, score in enumerate(scores)}
            emotions['time'] = start_time + chunk_size/2  # Центр интервала
            emotions_data.append(emotions)
            
            pbar.update(chunk_size)
    
    return pd.DataFrame(emotions_data)

def plot_audio_emotions(df, window_size=3):
    """Визуализация эмоций с временными метками"""
    plt.figure(figsize=(18, 10))
    
    # Цвета для эмоций
    emotion_colors = {
        'anger': '#FF0000',
        'disgust': '#800080',
        'enthusiasm': '#00FF00',
        'fear': '#000000',
        'happiness': '#FFA500',
        'neutral': '#808080',
        'sadness': '#0000FF'
    }
    
    # Сглаживание данных
    df_smooth = df.copy()
    for emotion in emotion_colors.keys():
        df_smooth[emotion] = df[emotion].rolling(window=window_size, min_periods=1).mean()
    
    # Построение графиков
    for emotion, color in emotion_colors.items():
        plt.plot(df_smooth['time'], df_smooth[emotion]*100,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=emotion.capitalize())
    
    plt.title('Динамика эмоций в аудио', fontsize=16, pad=20)
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Вероятность (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.2)
    
    # Расчет топ-3 эмоций
    total_power = df.drop('time', axis=1).sum().sum()
    emotion_percent = (df.drop('time', axis=1).sum() / total_power * 100).sort_values(ascending=False)
    top_emotions = emotion_percent.head(3)
    
    # Текст с топ-3 эмоциями
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
audio_df = process_audio_emotions("test_1.wav", chunk_size=1.0)
plot_audio_emotions(audio_df)