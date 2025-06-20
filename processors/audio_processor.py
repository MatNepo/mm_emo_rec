import sys
from pathlib import Path
import os
import logging
import subprocess

# Add parent directory to path to import from aed
sys.path.append(str(Path(__file__).parent.parent))
from processors.base_processor import BaseProcessor

class AudioProcessor(BaseProcessor):
    """Processor for audio emotion detection using AED model"""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.aed_path = Path(__file__).parent.parent / 'aed'
        self.root_path = Path(__file__).parent.parent
        
    def load_model(self):
        """Load the AED model"""
        self.logger.info("Loading AED model...")
        # Model is loaded in run.py
        self.logger.info("AED model loaded successfully")
        
    def process(self, input_path: str, output_path: str = None) -> dict:
        """Process audio file using AED run.py"""
        self.validate_paths(input_path, output_path)
        
        # Convert input path to absolute path
        abs_input_path = self.root_path / input_path
        
        # Change to AED directory
        original_dir = os.getcwd()
        os.chdir(self.aed_path)
        
        try:
            # Run AED processing with absolute path
            result = subprocess.run(
                [sys.executable, 'run.py', '--audio', str(abs_input_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"AED processing failed: {result.stderr}")
                return None
                
            # Parse results from output
            # TODO: Implement proper result parsing from AED output
            results = {
                'has_speech': True,  # This should be determined from AED output
                'emotions_data': None,  # This should be parsed from AED output
                'top_emotions': None  # This should be parsed from AED output
            }
            
            return results
            
        finally:
            # Return to original directory
            os.chdir(original_dir) 