from typing import Dict, Any, Tuple
import json
from pathlib import Path
from benchmark.models.qwen import QwenModel

class DocumentPreprocessor:
    def __init__(self, model: QwenModel):
        """Initialize the document preprocessor.
        
        Args:
            model: Qwen model instance for processing documents
        """
        self.model = model
        self.document_keywords = self._load_document_keywords()
        
    def _load_document_keywords(self) -> Dict[str, Dict[str, list]]:
        """Load document keywords from JSON file."""
        with open("/Users/avishabhiryani/Documents/private/N26_GenAI_Take_Home_Assignment/document_keywords.json", "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _create_preprocessing_prompt(self, document_path: str) -> str:
        """Create the prompt for preprocessing step.
        
        Args:
            document_path: Path to the document image
            
        Returns:
            Formatted prompt string
        """
        # Convert document keywords to a readable format for the prompt
        doc_types = "\n".join([f"- {doc_type}" for doc_type in self.document_keywords.keys()])
        keywords_by_lang = {}
        for doc_type, langs in self.document_keywords.items():
            for lang, keywords in langs.items():
                if lang not in keywords_by_lang:
                    keywords_by_lang[lang] = []
                keywords_by_lang[lang].extend(keywords)
                
        keywords_str = "\n".join([
            f"{lang.upper()} keywords:\n" + "\n".join([f"- {kw}" for kw in kws])
            for lang, kws in keywords_by_lang.items()
        ])
        
        return f"""You are a document processing assistant. Given a document image, perform the following tasks:

            1. Extract all text from the document and format it in clean, readable markdown.
            2. Detect the language of the document text.
            3. Classify the document type based on the following document types and their keywords:

            Document Types:
            {doc_types}

            Keywords by Language:
            {keywords_str}

            Return your response in the following JSON format:
            {{
                "extracted_text": "markdown formatted text from the document",
                "detected_language": "ISO language code (e.g., 'en', 'de', 'fr', 'es', 'it')",
                "document_type": "one of the document types listed above"
            }}

            Please ensure the classification is based on the document content and the provided keywords."""
    
    async def preprocess_document(self, document_path: str) -> str:
        """Preprocess a document to extract text, detect language, and classify type.
        
        Args:
            document_path: Path to the document image
            
        Returns:
            Dictionary containing:
            - extracted_text: Markdown formatted text
            - detected_language: ISO language code
            - document_type: Classified document type
        """
        prompt = self._create_preprocessing_prompt(document_path)
        
        # Process document using Qwen model
        result = await self.model.process_document(document_path, prompt)
        
        try:
            preprocess_result = json.loads(result)
            return {
                "document_type": preprocess_result["document_type"],
                "detected_language": preprocess_result["detected_language"],
                "extracted_text": preprocess_result["detected_language"]
            }
        except json.JSONDecodeError:
            # If the result isn't valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                extraction_result = json.loads(json_match.group())
                return {
                "document_type": preprocess_result["document_type"],
                "detected_language": preprocess_result["detected_language"],
                "extracted_text": preprocess_result["detected_language"]
            }
            raise ValueError("Could not parse model output as JSON") 