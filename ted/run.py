# Отключаем сообщения TensorFlow до импорта других модулей
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import subprocess
import sys
import argparse
from processor import TextEmotionProcessor
from utils import download_model, format_emotion_scores

def install_tabulate():
    try:
        import tabulate
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])

def process_audio(file_path: str, model_path: str = './weights/text_checkpoints'):
    """Process audio file and return results"""
    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return None
    
    download_model()
    
    processor = TextEmotionProcessor(model_path)
    
    try:
        # Проверяем расширение файла
        if not file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            print("Error: Unsupported audio file format. Supported formats: .wav, .mp3, .ogg, .flac")
            return None
            
        result = processor.process_audio(file_path)
        print("Транскрипция:", result['transcription'].strip())
        print("\nРаспределение эмоций:")
        print(format_emotion_scores(result['emotion_analysis']['predictions']))
        print(f"\nОсновная эмоция: {result['emotion_analysis']['predicted_emotion']}")
        
        # Сохраняем статистику с реальными значениями эмоций и путем к аудио файлу
        processor.save_statistics(result['emotion_analysis']['predictions'], file_path)
        
        return result
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

def main():
    # Устанавливаем необходимые зависимости
    install_tabulate()
    
    # Отключаем все логи и предупреждения
    logging.getLogger('transformers').setLevel(logging.ERROR)  # Отключаем логи transformers
    import warnings
    warnings.filterwarnings('ignore')  # Отключаем все предупреждения Python
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process audio for text emotion detection')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model_path', type=str, default='./weights/text_checkpoints', 
                       help='Path to model weights')
    
    args = parser.parse_args()
    
    # Process audio
    result = process_audio(args.audio, args.model_path)
    
    if result:
        # Return results in a format that can be parsed by the processor
        print("\nRESULTS_START")
        print("transcription:", result['transcription'].strip())
        print("emotions:")
        for emotion, score in result['emotion_analysis']['predictions'].items():
            print(f"{emotion}: {score}")
        print("top_emotion:", result['emotion_analysis']['predicted_emotion'])
        print("RESULTS_END")

if __name__ == "__main__":
    main()
