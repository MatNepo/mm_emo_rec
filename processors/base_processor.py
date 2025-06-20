from abc import ABC, abstractmethod
from pathlib import Path
import logging

class BaseProcessor(ABC):
    """Base class for all processors"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process(self, input_path: str, output_path: str = None) -> dict:
        """
        Process the input data and return results
        
        Args:
            input_path: Path to input data
            output_path: Optional path to save results
            
        Returns:
            dict: Processing results
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    def validate_paths(self, input_path: str, output_path: str = None):
        """Validate input and output paths"""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
            
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True) 