import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

import librosa
import numpy as np

TRUST = True

# Конфигурация и модель
config = AutoConfig.from_pretrained('Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition', trust_remote_code=TRUST)
model_ = Wav2Vec2ForSequenceClassification.from_pretrained(  # <-- Исправлено здесь
    "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition",
    trust_remote_code=TRUST
)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_.to(device)



def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model_(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs


result = predict("/path/to/russian_audio_speech.wav", 16000)
print(result)
