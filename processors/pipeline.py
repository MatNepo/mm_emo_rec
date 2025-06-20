import sys
from pathlib import Path
import logging
import subprocess
from typing import Dict, Optional
import traceback

from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.text_processor import TextProcessor

class EmotionAnalysisPipeline:
    """Pipeline for processing video files with multiple models"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.text_processor = TextProcessor()
        
        # Create necessary directories
        self.audio_output_dir = Path("data/video_to_audio")
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video file"""
        video_name = Path(video_path).stem
        audio_path = self.audio_output_dir / f"{video_name}.wav"
        
        if not audio_path.exists():
            self.logger.info(f"Extracting audio from {video_path}")
            
            # First check if video has audio stream
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
                    self.logger.warning("No audio stream found in video")
                    return None
                    
                command = [
                    'ffmpeg', '-i', str(video_path),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    str(audio_path)
                ]
                subprocess.run(command, capture_output=True, check=True)
                self.logger.info(f"Audio extracted to {audio_path}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
                return None
            except FileNotFoundError:
                self.logger.error("ffmpeg not found. Please install ffmpeg first.")
                return None
        
        return str(audio_path)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """Process video file with all models"""
        self.logger.info(f"Processing video: {video_path}")
        
        # Validate input video exists
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio
        audio_path = self.extract_audio(str(video_path))
        
        # Initialize results with default values
        results = {
            'video': None,
            'audio': None,
            'text': None,
            'combined_top_emotions': None,
            'processing_status': {
                'video': False,
                'audio': False,
                'text': False
            }
        }
        
        # Process video
        self.logger.info("Processing video...")
        try:
            video_results = self.video_processor.process(str(video_path))
            if video_results and 'top_emotions' in video_results:
                results['video'] = video_results
                results['processing_status']['video'] = True
                results['combined_top_emotions'] = video_results['top_emotions']
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
        
        # If we have audio, process it
        if audio_path:
            # Process audio
            self.logger.info("Processing audio...")
            try:
                audio_results = self.audio_processor.process(audio_path)
                if audio_results and audio_results.get('has_speech'):
                    results['audio'] = audio_results
                    results['processing_status']['audio'] = True
                    
                    # Process text
                    self.logger.info("Processing text...")
                    try:
                        text_results = self.text_processor.process(audio_path)
                        if text_results:
                            results['text'] = text_results
                            results['processing_status']['text'] = True
                    except Exception as e:
                        self.logger.error(f"Text processing failed: {str(e)}")
                        self.logger.debug(traceback.format_exc())
            except Exception as e:
                self.logger.error(f"Audio processing failed: {str(e)}")
                self.logger.debug(traceback.format_exc())
        
        # Combine emotions if we have multiple modalities
        if sum(results['processing_status'].values()) > 1:
            try:
                results['combined_top_emotions'] = self._combine_emotions(
                    results['audio']['top_emotions'] if results['audio'] else None,
                    results['video']['top_emotions'] if results['video'] else None,
                    results['text']['top_emotions'] if results['text'] else None
                )
            except Exception as e:
                self.logger.error(f"Error combining emotions: {str(e)}")
                self.logger.debug(traceback.format_exc())
        
        # Save results if output path is provided
        if output_path:
            try:
                self._save_results(results, output_path)
            except Exception as e:
                self.logger.error(f"Error saving results: {str(e)}")
                self.logger.debug(traceback.format_exc())
        
        return results
    
    def _combine_emotions(self, audio_emotions: Dict, video_emotions: Dict, text_emotions: Dict) -> Dict:
        """Combine emotions from different modalities"""
        if not any([audio_emotions, video_emotions, text_emotions]):
            return None
            
        # Combine emotions with equal weights
        combined = {}
        all_emotions = set()
        
        # Collect all unique emotions
        if audio_emotions:
            all_emotions.update(audio_emotions.keys())
        if video_emotions:
            all_emotions.update(video_emotions.keys())
        if text_emotions:
            all_emotions.update(text_emotions.keys())
        
        # Calculate weighted average
        for emotion in all_emotions:
            scores = []
            if audio_emotions and emotion in audio_emotions:
                scores.append(audio_emotions[emotion])
            if video_emotions and emotion in video_emotions:
                scores.append(video_emotions[emotion])
            if text_emotions and emotion in text_emotions:
                scores.append(text_emotions[emotion])
            
            if scores:
                combined[emotion] = sum(scores) / len(scores)
            
        # Get top 3 emotions
        return dict(sorted(combined.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:3])
    
    def _save_results(self, results: Dict, output_path: str):
        """Save results to file"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    pipeline = EmotionAnalysisPipeline()
    
    # Process all videos in data directory
    data_dir = Path("data")
    video_files = list(data_dir.glob("*.mp4"))
    
    for video_path in video_files:
        print(f"\nProcessing {video_path.name}...")
        results = pipeline.process_video(str(video_path))
        
        if results and results['combined_top_emotions']:
            print("\nTop emotions detected:")
            for emotion, score in results['combined_top_emotions'].items():
                print(f"{emotion}: {score:.2%}")
        else:
            print("\nNo significant emotions detected")
            if results:
                print("Processing status:")
                for modality, status in results['processing_status'].items():
                    print(f"- {modality}: {'Success' if status else 'Failed'}")

if __name__ == "__main__":
    main() 