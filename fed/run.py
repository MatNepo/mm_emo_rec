from processor import EmotionProcessor
import json

def process_video(input_video_path, output_video_path, lstm_model_name='Aff-Wild2'):
    """
    Process a video file to detect and display emotions.
    
    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path where the processed video will be saved
        lstm_model_name (str): Name of the LSTM model to use (default: 'Aff-Wild2')
        
    Returns:
        dict: Processing results including emotions data
    """
    processor = EmotionProcessor(
        backbone_model_path='weights/FER_static_ResNet50_AffectNet.pt',
        lstm_model_name=lstm_model_name
    )
    
    # Process video and get results
    results = processor.process_video(input_video_path, output_video_path)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video for emotion detection')
    parser.add_argument('--input_video', required=True, help='Path to input video file')
    parser.add_argument('--output_video', required=True, help='Path to save processed video')
    parser.add_argument('--model', default='Aff-Wild2', help='LSTM model name (default: Aff-Wild2)')
    
    args = parser.parse_args()
    
    # Process video
    results = process_video(args.input_video, args.output_video, args.model)
    
    if results:
        # Return results in a format that can be parsed by the processor
        print("\nRESULTS_START")
        print("top_emotions:")
        for emotion, count in results['top_emotions'].items():
            print(f"{emotion}: {count}")
        print("total_frames:", results['total_frames'])
        print("frames_with_faces:", results['frames_with_faces'])
        print("total_faces_detected:", results['total_faces_detected'])
        print("average_processing_time_ms:", results['average_processing_time_ms'])
        print("RESULTS_END")

if __name__ == '__main__':
    main() 