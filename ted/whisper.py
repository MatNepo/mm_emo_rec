import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperModel:
    def __init__(self, model_path='./weights/audio_checkpoints'):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
    
    def transcribe_audio(self, audio_path):
        try:
            # Загружаем аудио файл
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Подготавливаем входные данные
            input_features = self.processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features
            
            if torch.cuda.is_available():
                input_features = input_features.to('cuda')
            
            # Генерируем транскрипцию
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Декодируем результат
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # print(f"Транскрибированный текст: {transcription}") # Закомментировано: теперь вывод будет управляться ted/run.py
            
            return transcription
            
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")