import sys
from pathlib import Path
import os
import logging
import subprocess

# Add parent directory to path to import from fed
sys.path.append(str(Path(__file__).parent.parent))
from processors.base_processor import BaseProcessor

class VideoProcessor(BaseProcessor):
    """Processor for facial emotion detection using FED model"""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fed_path = Path(__file__).parent.parent / 'fed'
        self.root_path = Path(__file__).parent.parent
        
    def load_model(self):
        """Load the FED model"""
        self.logger.info("Loading FED model...")
        # Model is loaded in run.py
        self.logger.info("FED model loaded successfully")
        
    def process(self, input_path: str, output_path: str = None) -> dict:
        """Process video file using FED run.py"""
        self.validate_paths(input_path, output_path)
        
        # Convert paths to absolute paths
        abs_input_path = self.root_path / input_path
        if not output_path:
            output_path = str(self.root_path / 'output.mp4')
        else:
            output_path = str(self.root_path / output_path)
        
        # Change to FED directory
        original_dir = os.getcwd()
        os.chdir(self.fed_path)
        
        try:
            # Run FED processing with absolute paths
            result = subprocess.run(
                [sys.executable, 'run.py', 
                 '--input_video', str(abs_input_path),
                 '--output_video', output_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"FED processing failed: {result.stderr}")
                return None
                
            # Parse results from output
            # TODO: Implement proper result parsing from FED output
            results = {
                'emotions_data': None,  # This should be parsed from FED output
                'top_emotions': None  # This should be parsed from FED output
            }
            
            return results
            
        finally:
            # Return to original directory
            os.chdir(original_dir)