from typing import Dict, Any, List
import json
import torch
from pathlib import Path
from benchmark.models.qwen import QwenModel
from pipeline.preprocessor import DocumentPreprocessor

class DocumentProcessor:
    def __init__(self, model: QwenModel):
        """Initialize the document processor.
        
        Args:
            model: Qwen model instance for processing documents
        """
        self.model = model
        self.preprocessor = DocumentPreprocessor(model)
        self.extraction_fields = self._load_extraction_fields()
        
    def _load_extraction_fields(self) -> Dict[str, Dict[str, List[str]]]:
        """Load extraction fields from JSON file."""
        with open("/Users/avishabhiryani/Documents/private/N26_GenAI_Take_Home_Assignment/extraction-fields.json", "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _create_extraction_prompt(self, document_type: str, language: str, extracted_text: str) -> str:
        """Create the prompt for field extraction.
        
        Args:
            document_type: Type of document to process
            language: Language of the document
            extracted_text: Text extracted from the document
            
        Returns:
            Formatted prompt string
        """
        # Get relevant fields for the document type
        fields = self.extraction_fields.get(document_type, {})
        if not fields:
            fields = self.extraction_fields.get("general", {})
            
        # Format fields for the prompt
        fields_str = "\n".join([
            f"- {field}: [value]" for field in fields
        ])
        
        return f"""You are a document processing assistant. Given the following document text, extract the specified fields.

        Document Type: {document_type}
        Language: {language}
        
        Required Fields:
        {fields_str}
        
        Document Text:
        {extracted_text}
        
        Do not add any other text or comments. Return your response ONLY in the following JSON format and nothing else:
        {{
            "fields": {{
                "field_name": "extracted_value",
                ...
            }},
            "confidence_score": "confidence score between 0 and 1"
        }}
        
        Rules:
        1. If a field cannot be found, set its value to null
        2. Maintain the original format of dates, names, addresses, and monetary amounts
        3. For monetary amounts, include the currency symbol if present
        4. For dates, use the original format found in the document
        5. For names and addresses, preserve the original capitalization and formatting"""
        
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a document to extract relevant fields.
        
        Args:
            document_path: Path to the document image
            
        Returns:
            Dictionary containing:
            - document_type: Type of document
            - language: Detected language
            - fields: Extracted field values
            - confidence_score: Confidence in the extraction
        """
        # First, preprocess the document
        preprocess_result = await self.preprocessor.preprocess_document(document_path)
        torch.cuda.empty_cache()
        # Create extraction prompt
        prompt = self._create_extraction_prompt(
            preprocess_result["document_type"],
            preprocess_result["detected_language"],
            preprocess_result["extracted_text"]
        )
        
        # Process with model
        result = await self.model.process_document(None, prompt)
        result = " ".join(result)
        torch.cuda.empty_cache()
        # Parse the result
        try:
            extraction_result = json.loads(result)
            return {
                "document_type": preprocess_result["document_type"],
                "language": preprocess_result["detected_language"],
                "fields": extraction_result.get("fields", {}),
                "confidence_score": extraction_result.get("confidence_score", 0.0)
            }
        except json.JSONDecodeError:
            # If the result isn't valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                extraction_result = json.loads(json_match.group())
                return {
                    "document_type": preprocess_result["document_type"],
                    "language": preprocess_result["detected_language"],
                    "fields": extraction_result.get("fields", {}),
                    "confidence_score": extraction_result.get("confidence_score", 0.0)
                }
            raise ValueError("Could not parse model output as JSON") 