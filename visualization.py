import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def create_emotion_visualization(results_json_path, output_png_path):
    """
    Строит и сохраняет итоговую визуализацию эмоций по формату results.json.
    Все подписи и стиль — на русском языке, как в примере пользователя.
    """
    # Загрузка данных
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    plt.figure(figsize=(16, 12))
    plt.suptitle("Мультимодальный анализ эмоций", fontsize=20, fontweight="bold", y=0.98)

    # --- Общий анализ эмоций ---
    plt.subplot2grid((3, 4), (0, 0), colspan=4)
    combined = pd.Series(data.get("combined_top_emotions", {}))
    if not combined.empty:
        sns.barplot(x=combined.values, y=combined.index, palette="Purples_r", alpha=0.7)
        plt.title("Общий анализ эмоций", fontsize=14, fontweight="bold")
        plt.xlim(0, 1)
        for i, v in enumerate(combined.values):
            plt.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=12)
    else:
        plt.title("Общий анализ эмоций (нет данных)", fontsize=14, fontweight="bold")

    # --- Текстовые эмоции ---
    plt.subplot2grid((3, 4), (1, 0), colspan=2)
    ted_data = data.get("ted") or {}
    ted = pd.Series(ted_data.get("emotions", {}))
    if not ted.empty:
        sns.barplot(x=ted.values, y=ted.index, palette="Blues_r", alpha=0.7)
        plt.title("Анализ текстовых эмоций", fontsize=12, fontweight="bold")
        plt.xlim(0, 1)
        for i, v in enumerate(ted.values):
            plt.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=10)
    else:
        plt.title("Анализ текстовых эмоций (нет данных)", fontsize=12, fontweight="bold")

    # --- Лицевые эмоции ---
    plt.subplot2grid((3, 4), (1, 2), colspan=2)
    fed_data = data.get("fed") or {}
    fed = pd.Series(fed_data.get("emotions", {}))
    if not fed.empty:
        sns.barplot(x=fed.values, y=fed.index, palette="Reds_r", alpha=0.7)
        plt.title("Анализ лицевых эмоций", fontsize=12, fontweight="bold")
        plt.xlim(0, 1)
        for i, v in enumerate(fed.values):
            plt.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=10)
    else:
        plt.title("Анализ лицевых эмоций (нет данных)", fontsize=12, fontweight="bold")

    # --- Аудио эмоции ---
    plt.subplot2grid((3, 4), (2, 0), colspan=4)
    aed = pd.Series((data.get("aed") or {}).get("emotions", {}))
    if not aed.empty:
        sns.barplot(x=aed.values, y=aed.index, palette="Greens_r", alpha=0.7)
        plt.title("Анализ аудио эмоций", fontsize=12, fontweight="bold")
        plt.xlim(0, 1)
        for i, v in enumerate(aed.values):
            plt.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=10)
    else:
        plt.title("Анализ аудио эмоций (нет данных)", fontsize=12, fontweight="bold")

    # --- Подпись ---
    available = []
    if not ted.empty:
        available.append("Текстовые")
    if not fed.empty:
        available.append("Лицевые")
    if not aed.empty:
        available.append("Аудио")
    plt.figtext(0.5, 0.01, f"Доступные данные: {', '.join(available) if available else 'Нет'}", ha="center", fontsize=12, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_png_path, dpi=200)
    plt.close()
