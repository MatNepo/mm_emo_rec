# Отключаем сообщения TensorFlow до импорта других модулей
import os
import sys
import argparse
# Добавляем корневую директорию 'aed' в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Добавляем поддиректорию 'pytorch' в sys.path
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch'))
# Добавляем поддиректорию 'utils' в sys.path (для config и utilities)
sys.path.insert(2, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

# Define paths for visualizations and logs
VISUALIZATIONS_DIR = 'visualizations'
LOGS_DIR = 'logs'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
from datetime import datetime

# Configure logging
log_filename = os.path.join(LOGS_DIR, datetime.now().strftime('log_%Y%m%d_%H%M%S.log'))
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up file handler (for all logs)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Set up stream handler (for console output - only show critical errors by default)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.CRITICAL) # Set to CRITICAL to suppress INFO/WARNING in console
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO, 
    handlers=[
        file_handler,
        stream_handler
    ]
)

# Suppress transformers and other warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

from processor import AudioTaggingProcessor
from audio_tagging_utils import format_predictions
import config # utils/config.py

def process_audio(audio_path: str, model_type: str = "Cnn14", checkpoint_path: str = "./weights/cnn14.pth"):
    """Process audio file and return results"""
    logging.info(f"Starting audio tagging process for file: {audio_path}")

    # Check if audio file exists
    if not os.path.exists(audio_path):
        logging.error(f"Error: Audio file '{audio_path}' not found.")
        return None

    # Initialize processor
    processor = AudioTaggingProcessor(
        model_type=model_type,
        checkpoint_path=checkpoint_path
    )

    # Get labels from config
    labels = config.labels

    # Process audio
    results = processor.process_audio(audio_path)

    # Print predictions to console and log
    console_output = format_predictions(results['clipwise_output'], labels)
    print("Audio Tagging Predictions:")
    print(console_output)
    logging.info(f"\nAudio Tagging Predictions:\n{console_output}")

    # Save visualization
    processor.save_visualization(results)

    logging.info("Audio tagging process completed.")
    
    return results

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process audio for emotion detection')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model_type', type=str, default="Cnn14", help='Model type')
    parser.add_argument('--checkpoint', type=str, default="./weights/cnn14.pth", help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Process audio
    results = process_audio(args.audio, args.model_type, args.checkpoint)
    
    if results:
        # Return results in a format that can be parsed by the processor
        print("\nRESULTS_START")
        # Convert numpy array to dictionary if needed
        if isinstance(results['clipwise_output'], dict):
            clipwise_output = results['clipwise_output']
        else:
            # If it's a numpy array, create a dictionary with labels
            clipwise_output = dict(zip(config.labels, results['clipwise_output']))
            
        print(f"has_speech: {any(score > 0.8 for score in clipwise_output.values())}")
        print("emotions_data:")
        for label, score in clipwise_output.items():
            print(f"{label}: {score}")
        print("RESULTS_END")

if __name__ == "__main__":
    main() 