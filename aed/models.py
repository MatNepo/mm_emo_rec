# Этот файл просто импортирует Cnn14 из aed/pytorch/models.py
# Это позволяет нам использовать Cnn14 как AudioTaggingModel
from pytorch.models import Cnn14

# Оставляем AudioEmotionModel, если она нужна для других целей
# Если она не используется, этот класс можно удалить
# import os
# import torch
# import numpy as np
# from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# class AudioEmotionModel:
#     def __init__(self, model_path):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
#         self.model = AutoModelForAudioClassification.from_pretrained(model_path)
#         self.model.to(self.device)
        
#         # Словарь для маппинга индексов на эмоции
#         self.id2label = {
#             0: "радость",
#             1: "грусть",
#             2: "гнев",
#             3: "страх",
#             4: "отвращение",
#             5: "удивление",
#             6: "нейтральность"
#         }

#     def predict_emotions(self, audio_path):
#         # Загружаем аудио
#         import librosa
#         audio, sr = librosa.load(audio_path, sr=16000)
        
#         # Подготавливаем входные данные
#         inputs = self.feature_extractor(
#             audio, 
#             sampling_rate=sr, 
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Получаем предсказания
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probabilities = torch.softmax(logits, dim=1)
        
#         # Преобразуем в numpy и получаем предсказания
#         probs = probabilities[0].cpu().numpy()
#         predictions = {self.id2label[i]: float(prob) for i, prob in enumerate(probs)}
        
#         # Находим основную эмоцию
#         predicted_emotion = self.id2label[np.argmax(probs)]
        
#         return {
#             'predictions': predictions,
#             'predicted_emotion': predicted_emotion
#         } 