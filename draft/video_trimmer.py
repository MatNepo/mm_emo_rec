from moviepy.editor import VideoFileClip
import os
import argparse

def trim_video(input_path, start_time, end_time):
    """
    Trim video from start_time to end_time and save with '_cut' suffix
    
    Args:
        input_path (str): Path to input video file
        start_time (float): Start time in seconds (can include milliseconds)
        end_time (float): End time in seconds (can include milliseconds)
    """
    try:
        # Load the video
        video = VideoFileClip(input_path)
        
        # Trim the video
        trimmed_video = video.subclip(start_time, end_time)
        
        # Generate output path
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_cut{ext}")
        
        # Write the result
        trimmed_video.write_videofile(output_path)
        
        # Close the video to free up resources
        video.close()
        trimmed_video.close()
        
        print(f"Video successfully trimmed and saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Trim video from start time to end time')
    parser.add_argument('input_path', help='Path to input video file')
    parser.add_argument('start_time', type=float, help='Start time in seconds (can include milliseconds)')
    parser.add_argument('end_time', type=float, help='End time in seconds (can include milliseconds)')
    
    args = parser.parse_args()
    
    trim_video(args.input_path, args.start_time, args.end_time)

if __name__ == "__main__":
    main() 