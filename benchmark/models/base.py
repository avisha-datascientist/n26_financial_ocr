from abc import ABC, abstractmethod
from typing import Dict, Any
import time

class BaseModel(ABC):
    """Base class for all document processing models."""
    
    def __init__(self, api_key: str):
        """Initialize the base model."""
        pass
    
    @abstractmethod
    async def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document and return results.
        
        Args:
            document: Dictionary containing document data including:
                - image_path: Path to the document image
                - text: Text content of the document
                
        Returns:
            Dictionary containing:
                - document_type: Type of document
                - language: Document language
                - key_information: Extracted information
                - confidence_score: Confidence in the extraction
                - processing_time: Time taken to process
                - error: Any error message if processing failed
        """
        pass

    def format_output(self, raw_output: Any) -> Dict:
        """Format model output to standardized JSON"""
        if isinstance(raw_output, str):
            try:
                import json
                return json.loads(raw_output)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON output"}
        return raw_output

    async def measure_processing_time(self, func, *args, **kwargs) -> float:
        """Measure processing time of a function"""
        start_time = time.time()
        await func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time 