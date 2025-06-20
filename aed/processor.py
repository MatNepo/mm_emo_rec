import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging # Import logging

# Импорты теперь абсолютные, так как 'aed', 'aed/pytorch' и 'aed/utils' в sys.path
from pytorch.models import Cnn14
from pytorch.pytorch_utils import move_data_to_device
from utils.utilities import create_folder, get_filename # utilities.py находится в aed/utils
from utils import config # config находится в aed/utils/config.py

# Define path for visualizations
VISUALIZATIONS_DIR = 'visualizations'
LOGS_DIR = 'logs'

class AudioTaggingProcessor:
    def __init__(self, model_type, checkpoint_path, sample_rate=32000, 
                 window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Загружаем параметры модели из config
        self.classes_num = config.classes_num
        self.labels = config.labels
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size

        # Инициализируем модель
        Model = eval(model_type) # Это будет Cnn14
        self.model = Model(sample_rate=sample_rate, window_size=window_size, 
                           hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                           classes_num=self.classes_num)
        
        # Загружаем чекпоинт
        logging.info(f"Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        logging.info("Model checkpoint loaded successfully.")

        # Параллелим модель, если используем GPU
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)
            logging.info(f"Model moved to {self.device} and wrapped in DataParallel.")
        
        self.model.eval() # Переводим модель в режим оценки
        logging.info("Model set to evaluation mode.")
        
        # Создаем директорию для визуализаций - это теперь делается в run.py
        # os.makedirs('visualizations', exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration for the processor"""
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
            
        self.logger = logging.getLogger('AudioTaggingProcessor')
        self.logger.setLevel(logging.INFO)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f'audio_tagging_{current_time}.log'))
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def process_audio(self, audio_path):
        try:
            logging.info(f"Processing audio file: {audio_path}")
            (waveform, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
            waveform = waveform[None, :] # (1, audio_length)
            waveform = move_data_to_device(waveform, self.device)

            with torch.no_grad():
                batch_output_dict = self.model(waveform, None)
            
            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            # Convert numpy array to dictionary with labels
            clipwise_output_dict = dict(zip(self.labels, clipwise_output))
            
            logging.info("Audio processing completed.")
            
            return {
                'clipwise_output': clipwise_output_dict,
                'audio_path': audio_path,
                'waveform': waveform.cpu().numpy()[0] # Возвращаем форму волны для построения спектрограммы
            }
        except Exception as e:
            logging.error(f"Error during audio processing: {e}")
            raise

    def save_visualization(self, results):
        logging.info("Generating audio tagging visualization...")
        clipwise_output = results['clipwise_output']
        audio_path = results['audio_path']
        waveform = results['waveform']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename_base = get_filename(audio_path)
        fig_path = os.path.join(VISUALIZATIONS_DIR, f'{audio_filename_base}', f'aed_{timestamp}.png')
        create_folder(os.path.dirname(fig_path))

        # Sort predictions by value
        sorted_items = sorted(clipwise_output.items(), key=lambda x: x[1], reverse=True)
        top_k = 10  # Показываем топ-10 результатов
        top_labels = [item[0] for item in sorted_items[:top_k]]
        top_scores = [item[1] for item in sorted_items[:top_k]]

        # Построение графиков - аналогично части audio_tagging в inference.py
        stft = librosa.core.stft(y=waveform, n_fft=self.window_size,
                                 hop_length=self.hop_size, window='hann', center=True)
        
        # Добавляем небольшое эпсилон, чтобы избежать log(0)
        stft_abs = np.abs(stft)
        stft_abs = np.maximum(stft_abs, 1e-10) 
        stft_log = np.log(stft_abs)

        plt.style.use('seaborn') # Применяем стиль seaborn
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Спектрограмма
        img = axs[0].matshow(stft_log, origin='lower', aspect='auto', cmap='jet')
        axs[0].set_title('Log Spectrogram')
        axs[0].set_ylabel('Frequency bins')
        fig.colorbar(img, ax=axs[0], format='%+2.0f dB') 

        # Столбчатая диаграмма топ-10 предсказаний
        # Обратный порядок для того, чтобы самое высокое значение было сверху
        axs[1].barh(range(top_k), top_scores[::-1]) 
        axs[1].set_yticks(range(top_k))
        # Обратный порядок меток для соответствия графику
        axs[1].set_yticklabels(top_labels[::-1])
        axs[1].set_xlabel('Probability')
        axs[1].set_title('Top {} predictions'.format(top_k))
        axs[1].set_xlim(0, 1) # Устанавливаем пределы оси X для вероятности

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f'Saved audio tagging visualization to {fig_path}')
        
        return fig_path 