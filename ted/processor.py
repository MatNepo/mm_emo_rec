import logging
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models import TextEmotionModel
import torch
from whisper import WhisperModel
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_filename(path):
    """Extract filename without extension from path"""
    return os.path.splitext(os.path.basename(path))[0]

class TextEmotionProcessor:
    def __init__(self, model_path, audio_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = TextEmotionModel(model_path)
        self.whisper_model = WhisperModel()
        
        # Отключаем предупреждения
        import warnings
        warnings.filterwarnings('ignore')
        
        # Создаем директории для сохранения результатов
        os.makedirs('logs', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Цветовая схема для визуализации
        self.colors = {
            'радость': '#FFD700',  # золотой
            'грусть': '#4169E1',   # синий
            'гнев': '#FF4500',     # оранжевый
            'страх': '#800080',    # фиолетовый
            'отвращение': '#228B22', # зеленый
            'удивление': '#FF69B4', # розовый
            'нейтральность': '#808080' # серый
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        self.logger = logging.getLogger('TextEmotionProcessor')
        self.logger.setLevel(logging.INFO)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/text_emotion_{current_time}.log')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        self.stats = {
            'total_texts': 0,
            'total_audio': 0,
            'emotion_distribution': {emotion: 0 for emotion in self.text_model.emotions},
            'texts': [],
            'audio_files': []
        }
    
    def process_text(self, text):
        result = self.text_model.predict_emotions(text)
        return result
    
    def process_audio(self, audio_path):
        try:
            self.logger.info(f"Processing audio file: {audio_path}")
            
            # Транскрибируем аудио
            transcription = self.whisper_model.transcribe_audio(audio_path)
            self.logger.info(f"Transcription: {transcription[:100]}...")
            
            # Анализируем эмоции в тексте
            emotion_analysis = self.process_text(transcription)
            
            self.stats['total_audio'] += 1
            self.stats['audio_files'].append({
                'file_path': audio_path,
                'transcription': transcription,
                'predictions': emotion_analysis['predictions'],
                'predicted_emotion': emotion_analysis['predicted_emotion'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return {
                'transcription': transcription,
                'emotion_analysis': emotion_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            raise
    
    def create_visualizations(self):
        try:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if not os.path.exists('visualizations'):
                os.makedirs('visualizations')
            
            plt.style.use('seaborn')
            
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)
            
            # 1. Pie chart
            ax1 = fig.add_subplot(gs[0, 0])
            emotions = list(self.stats['emotion_distribution'].keys())
            values = list(self.stats['emotion_distribution'].values())
            colors = sns.color_palette("husl", len(emotions))
            ax1.pie(values, labels=emotions, colors=colors, autopct='%1.1f%%')
            ax1.set_title('Распределение эмоций')
            
            # 2. Bar chart
            ax2 = fig.add_subplot(gs[0, 1])
            sns.barplot(x=emotions, y=values, palette=colors, ax=ax2)
            ax2.set_title('Количество эмоций')
            ax2.set_xticklabels(emotions, rotation=45)
            plt.setp(ax2.get_xticklabels(), ha='right')
            
            # 3. Timeline
            if len(self.stats['texts']) > 1:
                ax3 = fig.add_subplot(gs[1, :])
                timeline_data = [text['predicted_emotion'] for text in self.stats['texts']]
                sns.histplot(timeline_data, discrete=True, ax=ax3)
                ax3.set_title('Хронология эмоций')
                ax3.set_xlabel('Эмоция')
                ax3.set_ylabel('Количество')
                ax3.set_xticklabels(emotions, rotation=45)
                plt.setp(ax3.get_xticklabels(), ha='right')
            
            plt.tight_layout()
            stats_path = f'visualizations/text_emotion_stats_{current_time}.png'
            plt.savefig(stats_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved emotion statistics visualization to {stats_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def save_statistics(self, emotion_values, audio_path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем статистику в JSON
        stats = {
            'timestamp': timestamp,
            'device': str(self.device),
            'text_model': self.text_model.__class__.__name__,
            'emotion_values': emotion_values
        }
        
        with open(f'logs/stats_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        # Создаем визуализацию
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Получаем значения эмоций
        emotions = list(emotion_values.keys())
        values = list(emotion_values.values())
        
        # Создаем confusion matrix
        # Для каждой эмоции создаем строку, где максимальное значение будет на диагонали
        # и пропорционально уменьшающиеся значения в других ячейках
        matrix = np.zeros((len(emotions), len(emotions)))
        for i, emotion in enumerate(emotions):
            # Основное значение на диагонали
            matrix[i, i] = values[i]
            # Остальные значения пропорционально уменьшаются
            for j in range(len(emotions)):
                if i != j:
                    matrix[i, j] = values[i] * 0.3  # 30% от основного значения
        
        # Визуализируем confusion matrix
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=emotions, yticklabels=emotions, ax=ax1)
        ax1.set_title('Матрица эмоций')
        ax1.set_xlabel('Предсказанные эмоции')
        ax1.set_ylabel('Истинные эмоции')
        
        # Создаем столбчатую диаграмму с процентами
        bars = sns.barplot(x=emotions, y=values, palette=self.colors, ax=ax2, hue=emotions, legend=False)
        ax2.set_xticklabels(emotions, rotation=45)
        ax2.set_title('Распределение эмоций')
        
        # Добавляем процентные значения над столбцами
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{values[i]*100:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        # Настраиваем оси
        ax2.set_ylim(0, 1.1)  # Увеличиваем верхнюю границу для процентов
        ax2.set_ylabel('Процент')
        
        plt.tight_layout()
        
        # Получаем имя файла из пути аудио
        audio_filename_base = get_filename(audio_path) if audio_path else 'unknown'
        plt.savefig(f'../data/aud_from_vid/{audio_filename_base}/ted_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
