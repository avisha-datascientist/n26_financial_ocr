from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    """Base class for all document processing models."""
    
    def __init__(self, api_key: str):
        """Initialize the base model."""
        self.api_key = api_key
    
    @abstractmethod
    async def process_document(self, document: Dict[str, Any]) -> str:
        """Process a document and return the extracted key details.
        
        Args:
            document: Dictionary containing document data including:
                - image_path: Path to the document image
                
        Returns:
            String containing the extracted key details
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
