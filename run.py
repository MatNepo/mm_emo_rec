import os
import subprocess
import json
from pathlib import Path
import logging
from typing import Dict, Optional
import argparse
from utils import transform_results
from visualization import create_emotion_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the source directory
SOURCE_DIR = Path(__file__).parent.absolute()

# Speech-related tags from AED model
SPEECH_TAGS = [
    'Speech', 'Male speech, man speaking', 'Female speech, woman speaking',
    'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
    'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell',
    'Battle cry', 'Children shouting', 'Screaming', 'Whispering'
]

def extract_audio(video_path: str) -> Optional[str]:
    """Extract audio from video file"""
    video_name = Path(video_path).stem
    audio_dir = SOURCE_DIR / "data/aud_from_vid"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{video_name}.wav"
    
    if not audio_path.exists():
        logger.info(f"Extracting audio from {video_path}")
        
        # Check if video has audio stream
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                logger.warning("No audio stream found in video")
                return None
                
            command = [
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                str(audio_path)
            ]
            subprocess.run(command, capture_output=True, check=True)
            logger.info(f"Audio extracted to {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg first.")
            return None
    
    return str(audio_path)

def parse_model_output(output):
    """Parse model output and extract results between markers"""
    try:
        # Find content between markers
        start_idx = output.find("RESULTS_START")
        end_idx = output.find("RESULTS_END")
        
        if start_idx == -1 or end_idx == -1:
            return None
            
        content = output[start_idx:end_idx].strip()
        
        # Initialize result dictionary
        result = {}
        
        # Parse different sections
        current_section = None
        current_data = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line in ["RESULTS_START", "RESULTS_END"]:
                continue
                
            if line.endswith(':'):
                # Save previous section if exists
                if current_section and current_data:
                    result[current_section] = current_data
                
                # Start new section
                current_section = line[:-1]
                current_data = {}
                continue
            
            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle different data types
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                
                # If we are inside a section, add to current_data
                if current_section:
                    current_data[key] = value
                # Otherwise, add to the main result dictionary directly
                else:
                    result[key] = value
        
        # Save last section
        if current_section and current_data:
            result[current_section] = current_data
            
        # Handle special cases for different models
        if 'emotions' in result:
            # For AED: Convert emotions to dictionary if it's a list
            if isinstance(result['emotions'], list):
                emotions_dict = {}
                for emotion in result['emotions']:
                    if isinstance(emotion, dict):
                        for key, value in emotion.items():
                            emotions_dict[key] = value
                result['emotions'] = emotions_dict
            # For TED: Convert emotion names to scores
            elif isinstance(result['emotions'], dict):
                emotions_dict = {}
                for emotion, score in result['emotions'].items():
                    if isinstance(score, (int, float)):
                        emotions_dict[emotion] = float(score)
                        
                    else:
                        # If score is not a number, try to parse it
                        try:
                            emotions_dict[emotion] = float(score)
                        except (ValueError, TypeError):
                            # If parsing fails, use default score of 1.0
                            emotions_dict[emotion] = 1.0
                result['emotions'] = emotions_dict
                
        return result
        
    except Exception as e:
        print(f"Error parsing model output: {str(e)}")
        return None

def process_audio(audio_path: str) -> Optional[Dict]:
    """Process audio with AED model"""
    try:
        logger.info("Processing audio with AED...")
        # Change to AED directory
        aed_dir = SOURCE_DIR / 'aed'
        result = subprocess.run(
            [
                'python', 'run.py',
                '--audio', str(Path(audio_path).absolute()),
                '--model_type', 'Cnn14',
                '--checkpoint', './weights/cnn14.pth'
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(aed_dir)
        )
        # Parse the output to get emotions
        parsed_result = parse_model_output(result.stdout)
        if parsed_result and 'emotions_data' in parsed_result:
            # Get all emotions and sort them by score
            emotions = parsed_result['emotions_data']
            # Sort emotions by score in descending order and take top 10
            top_emotions = dict(sorted(emotions.items(), 
                                     key=lambda x: float(x[1]), 
                                     reverse=True)[:10])
            
            # Check if any speech-related emotion is in top 5
            top_5_emotions = list(top_emotions.items())[:5]
            has_speech = any(emotion in SPEECH_TAGS for emotion, _ in top_5_emotions)
            
            return {
                'has_speech': has_speech,
                'emotions': top_emotions
            }
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"AED processing failed: {e.stderr}")
        return None

def process_text(audio_path: str) -> Optional[Dict]:
    """Process audio with TED model"""
    try:
        logger.info("Processing text with TED...")
        # Change to TED directory
        ted_dir = SOURCE_DIR / 'ted'
        result = subprocess.run(
            [
                'python', 'run.py',
                '--audio', str(Path(audio_path).absolute())
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(ted_dir)
        )
        
        # Parse the output to get emotions
        parsed_result = parse_model_output(result.stdout)
        if parsed_result and 'emotions' in parsed_result:
            # Get emotions and top_emotion
            emotions = parsed_result['emotions']
            top_emotion = parsed_result.get('top_emotion', '')
            
            # Convert emotion scores to float and keep all emotions
            emotions = {k: float(v) for k, v in emotions.items()}
            
            return {
                'transcription': parsed_result.get('transcription', ''),
                'emotions': emotions,
                'top_emotion': top_emotion
            }
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"TED processing failed: {e.stderr}")
        return None

def process_video(video_path: str) -> Optional[Dict]:
    """Process video with FED model"""
    try:
        logger.info("Processing video with FED...")
        
        # Create output directory for processed videos
        output_dir = SOURCE_DIR / "data/processed_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output video path
        video_name = Path(video_path).stem
        output_video = output_dir / f"{video_name}_processed.mp4"
        
        # Change to FED directory
        fed_dir = SOURCE_DIR / 'fed'
        result = subprocess.run(
            [
                'python', 'run.py',
                '--input_video', str(Path(video_path).absolute()),
                '--output_video', str(output_video.absolute()),
                '--model', 'Aff-Wild2'
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(fed_dir)
        )
        
        # Parse the output to get emotions
        parsed_result = parse_model_output(result.stdout)
        if parsed_result and 'top_emotions' in parsed_result:
            # Convert emotions to Russian
            emotion_mapping = {
                'Neutral': 'нейтральность',
                'Happiness': 'радость',
                'Sadness': 'грусть',
                'Surprise': 'удивление',
                'Fear': 'страх',
                'Disgust': 'отвращение',
                'Anger': 'гнев'
            }
            
            # Use raw counts from parsed_result['top_emotions'] and map to Russian
            emotions = {}
            for emotion, count in parsed_result['top_emotions'].items():
                if emotion in emotion_mapping:
                    russian_emotion = emotion_mapping[emotion]
                    emotions[russian_emotion] = count
            
            return {
                'emotions': emotions,
                'total_frames': parsed_result.get('total_frames', 0),
                'frames_with_faces': parsed_result.get('frames_with_faces', 0),
                'total_faces_detected': parsed_result.get('total_faces_detected', 0),
                'average_processing_time_ms': parsed_result.get('average_processing_time_ms', 0)
            }
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"FED processing failed: {e.stderr}")
        return None

def normalize_emotions(emotions):
    """Normalize emotions to a standard format"""
    # Define standard emotion names in Russian
    standard_emotions = {
        'нейтральность': 'нейтральность',
        'радость': 'радость',
        'грусть': 'грусть',
        'удивление': 'удивление',
        'страх': 'страх',
        'отвращение': 'отвращение',
        'гнев': 'гнев'
    }
    
    normalized = {}
    for emotion, score in emotions.items():
        # Skip non-emotion keys
        if emotion in ['top_emotion', 'has_speech']:
            continue
        # Convert to standard format if needed
        if emotion in standard_emotions:
            normalized[emotion] = float(score)
    
    return normalized

def get_sound_emotion_mapping():
    """Get mapping of sound tags to emotions with their weights"""
    return {
        # Радость
        'Laughter': {'emotion': 'радость', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Giggle': {'emotion': 'радость', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        'Chuckle, chortle': {'emotion': 'радость', 'weights': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]},
        'Snicker': {'emotion': 'радость', 'weights': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01]},
        
        # Грусть
        'Sobbing': {'emotion': 'грусть', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Crying': {'emotion': 'грусть', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        'Whimper': {'emotion': 'грусть', 'weights': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]},
        
        # Страх
        'Screaming': {'emotion': 'страх', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Shouting': {'emotion': 'страх', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        'Yell': {'emotion': 'страх', 'weights': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]},
        
        # Гнев
        'Angry': {'emotion': 'гнев', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Growl': {'emotion': 'гнев', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        'Roar': {'emotion': 'гнев', 'weights': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]},
        
        # Удивление
        'Gasp': {'emotion': 'удивление', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Sigh': {'emotion': 'удивление', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        
        # Отвращение
        'Gag': {'emotion': 'отвращение', 'weights': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
        'Cough': {'emotion': 'отвращение', 'weights': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]},
        
        # Нейтральность
        'Speech': {'emotion': 'нейтральность', 'weights': [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]},
        'Conversation': {'emotion': 'нейтральность', 'weights': [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01]},
    }

def get_emotion_from_sounds(aed_results):
    """Extract emotions from sound tags in AED results"""
    if not aed_results or 'emotions' not in aed_results:
        return {}
        
    sound_mapping = get_sound_emotion_mapping()
    emotion_scores = {}
    
    # Get top 10 sounds
    top_sounds = dict(sorted(aed_results['emotions'].items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:10])
    
    # Convert to list to maintain order
    sounds_list = list(top_sounds.items())
    
    # Process each sound and its position
    for idx, (sound, score) in enumerate(sounds_list):
        if sound in sound_mapping:
            emotion = sound_mapping[sound]['emotion']
            weight = sound_mapping[sound]['weights'][idx]
            
            # Add weighted score to emotion
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += score * weight
    
    return emotion_scores

def combine_emotions(ted_results, fed_results, aed_results):
    """Combine emotions from different modalities with weighted averaging"""
    combined = {}
    
    # Define audio emotions that indicate happiness/joy
    joy_indicators = {
        'Laughter': 0.3,  # Increased weight for laughter
        'Snicker': 0.2,   # Increased weight for snicker
        'Chuckle, chortle': 0.2,  # Increased weight for chuckle
        'Giggle': 0.15,   # Increased weight for giggle
        'Singing': 0.1,   # Some weight for singing
        'Male singing': 0.1  # Some weight for male singing
    }
    
    # Process text emotions (TED)
    if ted_results and 'emotions' in ted_results:
        for emotion, value in ted_results['emotions'].items():
            if emotion != 'top_emotion':  # Skip the top_emotion field
                combined[emotion] = value * 0.4  # Base weight for text emotions
    
    # Process facial emotions (FED)
    if fed_results and 'emotions' in fed_results:
        for emotion, value in fed_results['emotions'].items():
            if emotion in combined:
                combined[emotion] += value * 0.4  # Base weight for facial emotions
            else:
                combined[emotion] = value * 0.4
    
    # Process audio emotions (AED) and add bonus for joy indicators
    if aed_results and 'emotions' in aed_results:
        # Check for joy indicators in top emotions
        for emotion, value in aed_results['emotions'].items():
            if emotion in joy_indicators:
                # Add bonus to радость based on the joy indicator's weight
                if 'радость' in combined:
                    combined['радость'] += value * joy_indicators[emotion]
                else:
                    combined['радость'] = value * joy_indicators[emotion]
    
    # Normalize the combined emotions
    total = sum(combined.values())
    if total > 0:
        combined = {k: v/total for k, v in combined.items()}
    
    # Sort by value in descending order
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

def main():
    parser = argparse.ArgumentParser(description='Process video for emotion detection')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', help='Path to save results (optional, defaults to ./results/<video_name>/results.json)')
    args = parser.parse_args()
    
    # Create results directory structure
    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create video-specific directory
        video_name = video_path.stem
        video_dir = results_dir / video_name
        video_dir.mkdir(exist_ok=True)
        
        # Set default output path
        output_path = video_dir / "results.json"
    
    # Extract audio from video
    audio_path = extract_audio(str(video_path))
    
    # Process with different models
    aed_results = process_audio(audio_path) if audio_path else None
    
    # Process text emotions (TED)
    ted_results = None
    if audio_path and aed_results and aed_results.get('has_speech', False):
        ted_results = process_text(audio_path)
        if ted_results and 'transcription' in ted_results:
            logger.info(f"Текстовая транскрипция: {ted_results['transcription']}")
    elif audio_path:
        logger.info("Пропуск анализа текстовых эмоций: речь не обнаружена в аудиодорожке.")
    
    results = {
        'aed': aed_results,
        'ted': ted_results,
        'fed': None
    }

    # Process facial emotions (FED)
    fed_results = process_video(str(video_path))
    results['fed'] = fed_results

    # Combine emotions from all modalities
    # Only combine if at least TED or FED results are available
    if results['ted'] or results['fed'] or results['aed']:
        results['combined_top_emotions'] = combine_emotions(results['ted'], results['fed'], results['aed'])
    else:
        logger.info("Недостаточно данных для комбинированного анализа эмоций (нет текстовых, лицевых или аудио данных).")

    # Transform results into required format
    final_results = transform_results(results)
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Create visualization
    viz_path = output_path.parent / f"{output_path.stem}_visualization.png"
    create_emotion_visualization(output_path, viz_path)
    logger.info(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    main() 