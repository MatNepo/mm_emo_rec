import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

class TextEmotionModel:
    def __init__(self, model_path='./weights/text_checkpoints'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5EncoderModel.from_pretrained(model_path)
        self.model.eval()
        
        # Define emotions list
        self.emotions = [
            'радость', 'грусть', 'гнев', 'страх', 
            'удивление', 'отвращение', 'нейтральность'
        ]
        
        # Pre-compute embeddings for emotions
        self.emotion_embeddings = [
            self.get_embedding('categorize_sentiment: ' + emotion) 
            for emotion in self.emotions
        ]
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0], dim=-1)
    
    def predict_emotions(self, text):
        text_emb = self.get_embedding('categorize_sentiment: ' + text)
        
        # Получаем косинусное сходство для каждой эмоции
        similarities = {
            emotion: F.cosine_similarity(text_emb, emb, dim=-1).item()
            for emotion, emb in zip(self.emotions, self.emotion_embeddings)
        }
        
        # Нормализуем значения в диапазон [0, 1]
        min_val = min(similarities.values())
        max_val = max(similarities.values())
        normalized_similarities = {
            emotion: (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            for emotion, val in similarities.items()
        }
        
        # Нормализуем так, чтобы сумма была равна 1
        total = sum(normalized_similarities.values())
        probabilities = {
            emotion: val / total
            for emotion, val in normalized_similarities.items()
        }
        
        sorted_probabilities = dict(
            sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        )
        
        predicted_emotion = max(probabilities, key=probabilities.get)
        
        return {
            'predictions': sorted_probabilities,
            'predicted_emotion': predicted_emotion
        }
