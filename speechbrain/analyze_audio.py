import torch
import torchaudio
import sys
import os
import warnings
import logging

# Отключаем предупреждения
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем вывод TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN сообщения

# Перенаправляем stderr в null
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

sys.path.append("weights")
from custom_interface import CustomEncoderWav2vec2Classifier

# Initialize the model with pre-trained weights
model = CustomEncoderWav2vec2Classifier.from_hparams(
    source="weights",
    savedir="weights",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Load and process the audio file
signal, sr = torchaudio.load("hap.wav")
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)

# Get predictions (emotion classification)
out_prob, score, index, text_lab = model.classify_batch(signal)

# Восстанавливаем stderr
sys.stderr = stderr

# Получаем список всех эмоций
emotions = list(model.hparams.label_encoder.lab2ind.keys())

# Выводим вероятности для каждой эмоции
print("Probabilities for each emotion:")
for i, emotion in enumerate(emotions):
    print(f"{emotion}: {out_prob[0][i].item():.4f}")

# Выводим наиболее вероятную эмоцию
print(f"\nPredicted emotion: {text_lab[0]}")