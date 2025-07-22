import os
import cv2
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
import subprocess
from pathlib import Path

# Speech-related tags from AED model
SPEECH_TAGS = ['speech', 'conversation', 'talking', 'voice', 'speaking']
SPEECH_THRESHOLD = 0.8

class EmotionAnalyzer:
    def __init__(self):
        # Initialize audio model
        self.trust = True
        self.audio_config = AutoConfig.from_pretrained(
            'Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition',
            trust_remote_code=self.trust
        )
        self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition",
            trust_remote_code=self.trust
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_model = self.audio_model.to(self.device)
        
        # Create necessary directories
        self.audio_output_dir = Path("data/video_to_audio")
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio_from_video(self, video_path):
        """Extract audio from video and save as WAV file"""
        video_name = Path(video_path).stem
        audio_path = self.audio_output_dir / f"{video_name}.wav"
        
        if not audio_path.exists():
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
                    print("No audio stream found in video")
                    return None
                    
                command = [
                    'ffmpeg', '-i', str(video_path),
                    '-vn', '-acodec', 'pcm_s16le',
                    '-ar', '16000', '-ac', '1',
                    str(audio_path)
                ]
                subprocess.run(command, capture_output=True, check=True)
                print(f"Audio extracted to {audio_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
                return None
            except FileNotFoundError:
                print("ffmpeg not found. Please install ffmpeg first.")
                return None
        
        return str(audio_path)

    def check_for_speech(self, audio_path):
        """Check if audio contains speech using AED model"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process in chunks
        chunk_size = 1.0  # 1 second chunks
        duration = librosa.get_duration(y=y, sr=sr)
        
        for start_time in np.arange(0, duration, chunk_size):
            end_time = start_time + chunk_size
            chunk = y[int(start_time*sr):int(end_time*sr)]
            
            # Get predictions
            inputs = self.feature_extractor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                logits = self.audio_model(input_values=input_values).logits
            
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            
            # Check if any speech-related tag has high probability
            for i, score in enumerate(scores):
                label = self.audio_config.id2label[i]
                if label.lower() in SPEECH_TAGS and score > SPEECH_THRESHOLD:
                    return True
        
        return False

    def analyze_audio_emotions(self, audio_path, chunk_size=1.0):
        """Analyze emotions in audio with timestamps"""
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        emotions_data = []
        
        with tqdm(total=int(duration), desc="Analyzing audio emotions") as pbar:
            for start_time in np.arange(0, duration, chunk_size):
                end_time = start_time + chunk_size
                chunk = y[int(start_time*sr):int(end_time*sr)]
                
                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs.input_values.to(self.device)
                with torch.no_grad():
                    logits = self.audio_model(input_values=input_values).logits
                
                scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                
                emotions = {self.audio_config.id2label[i]: float(score) 
                          for i, score in enumerate(scores)}
                emotions['time'] = start_time + chunk_size/2
                emotions_data.append(emotions)
                
                pbar.update(chunk_size)
        
        return pd.DataFrame(emotions_data)

    def analyze_video_emotions(self, video_path, frame_interval=30):
        """Analyze emotions in video frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        emotions_data = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="Analyzing video emotions") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    try:
                        analysis = DeepFace.analyze(frame, actions=['emotion'], silent=True)
                        emotions = analysis[0]['emotion']
                        emotions['time'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        emotions_data.append(emotions)
                    except:
                        pass

                frame_count += 1
                pbar.update(1)

        cap.release()
        return pd.DataFrame(emotions_data)

    def get_top_emotions(self, audio_df, video_df):
        """Combine and get top emotions from both audio and video analysis"""
        # Calculate average emotions from audio
        audio_emotions = audio_df.drop('time', axis=1).mean()
        
        # Calculate average emotions from video
        video_emotions = video_df.drop('time', axis=1).mean()
        
        # Combine emotions (you might want to adjust weights)
        combined_emotions = pd.concat([audio_emotions, video_emotions]).groupby(level=0).mean()
        
        # Get top 3 emotions
        if isinstance(combined_emotions, pd.DataFrame):
            combined_emotions = combined_emotions.squeeze()
        if isinstance(combined_emotions, pd.Series):
            top_emotions = combined_emotions.nlargest(3)
        else:
            top_emotions = combined_emotions
        
        return top_emotions

    def process_video(self, video_path):
        """Main method to process a video file"""
        print(f"Processing video: {video_path}")
        
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        if not audio_path:
            print("No audio stream found in video, skipping audio and text analysis")
            # Analyze only video emotions
            print("Analyzing video emotions...")
            video_emotions = self.analyze_video_emotions(video_path)
            
            # Get top emotions from video only
            if isinstance(video_emotions, pd.DataFrame) and not video_emotions.empty:
                # Convert to Series and get mean values
                emotions_mean = pd.Series(video_emotions.drop('time', axis=1).mean())
                # Get top 3 emotions
                top_emotions = dict(emotions_mean.nlargest(3))
                
                print("\nTop emotions detected from video:")
                for emotion, score in top_emotions.items():
                    print(f"{emotion}: {score:.2%}")
                
                return {
                    'video_emotions': video_emotions,
                    'top_emotions': top_emotions
                }
            else:
                print("No emotions detected in video")
                return None
        
        print(f"Extracted audio to: {audio_path}")
        
        # Check for speech
        has_speech = self.check_for_speech(audio_path)
        print(f"Speech detected: {has_speech}")
        
        if has_speech:
            # Analyze audio emotions
            print("Analyzing audio emotions...")
            audio_emotions = self.analyze_audio_emotions(audio_path)
            
            # Analyze video emotions
            print("Analyzing video emotions...")
            video_emotions = self.analyze_video_emotions(video_path)
            
            # Get top emotions
            top_emotions = self.get_top_emotions(audio_emotions, video_emotions)
            
            print("\nTop emotions detected:")
            for emotion, score in top_emotions.items():
                print(f"{emotion}: {score:.2%}")
            
            return {
                'audio_emotions': audio_emotions,
                'video_emotions': video_emotions,
                'top_emotions': top_emotions
            }
        else:
            print("No significant speech detected in the audio")
            return None

def main():
    analyzer = EmotionAnalyzer()
    
    # Process all videos in the data directory
    data_dir = Path("data")
    video_files = list(data_dir.glob("*.mp4"))
    
    for video_path in video_files:
        print(f"\nProcessing {video_path.name}...")
        results = analyzer.process_video(str(video_path))
        
        if results:
            print(f"\nAnalysis complete for {video_path.name}")
        else:
            print(f"\nSkipped {video_path.name} due to no speech detected")

if __name__ == "__main__":
    main() 