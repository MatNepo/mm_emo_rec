import sys
from pathlib import Path
import os
import logging
import subprocess

# Add parent directory to path to import from ted
sys.path.append(str(Path(__file__).parent.parent))
from processors.base_processor import BaseProcessor

class TextProcessor(BaseProcessor):
    """Processor for text emotion detection using TED model"""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ted_path = Path(__file__).parent.parent / 'ted'
        self.root_path = Path(__file__).parent.parent
        
    def load_model(self):
        """Load the TED model"""
        self.logger.info("Loading TED model...")
        # Model is loaded in run.py
        self.logger.info("TED model loaded successfully")
        
    def process(self, input_path: str, output_path: str = None) -> dict:
        """Process audio file using TED run.py"""
        self.validate_paths(input_path, output_path)
        
        # Convert input path to absolute path
        abs_input_path = self.root_path / input_path
        
        # Change to TED directory
        original_dir = os.getcwd()
        os.chdir(self.ted_path)
        
        try:
            # Run TED processing with absolute path
            result = subprocess.run(
                [sys.executable, 'run.py', '--audio', str(abs_input_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"TED processing failed: {result.stderr}")
                return None
                
            # Parse results from output
            # TODO: Implement proper result parsing from TED output
            results = {
                'transcription': None,  # This should be parsed from TED output
                'emotions': None,  # This should be parsed from TED output
                'top_emotions': None  # This should be parsed from TED output
            }
            
            return results
            
        finally:
            # Return to original directory
            os.chdir(original_dir) 