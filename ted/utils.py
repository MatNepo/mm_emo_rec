import os
from huggingface_hub import snapshot_download

def download_model():
    if not os.path.exists("weights/text_checkpoints"):
        snapshot_download(
            repo_id="ai-forever/FRIDA",
            local_dir="weights/text_checkpoints",
            local_dir_use_symlinks=False
        )

def format_emotion_scores(scores):
    # Сортируем эмоции по убыванию значений
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    # Форматируем вывод построчно
    result = []
    for emotion, score in sorted_scores.items():
        percentage = f"{score * 100:.1f}%"
        result.append(f"{emotion}: {percentage}")
    
    return "\n".join(result)
