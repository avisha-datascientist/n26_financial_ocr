from typing import Dict, Any, Tuple, List
import json
import os
import torch
from pathlib import Path
from benchmark.models.qwen import QwenModel
from pdf2image import convert_from_path

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

    def _load_pdf_to_image(self, pdf_path: str) -> List[str]:
        """Load the PDF and convert it to an image."""
        # Path to your PDF file
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # Convert PDF to list of images (one per page)
        images = convert_from_path(pdf_path)

        output_dir = "/content/n26_financial_ocr/pipeline/docs_images"
        all_image_paths = []
        # Save each page as an image file
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'{doc_name}_page_{i + 1}.jpeg')
            image.save(image_path, 'JPEG')
            all_image_paths.append(image_path)
        return all_image_paths
            
    def _create_classification_prompt(self, extracted_text: str) -> str:
        """Create the prompt for document classification and language detection.
        
        Args:
            extracted_text: Text extracted from the document
            
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
        
        return f"""Analyze the following document text and classify it based on the provided document types and keywords.

        Document Types:
        {doc_types}

        Keywords by Language:
        {keywords_str}

        Document Text:
        {extracted_text}

        Return your response in the following JSON format:
        {{
            "document_type": "one of the document types listed above",
            "detected_language": "ISO language code (e.g., 'en', 'de', 'fr', 'es', 'it')",
            "confidence_score": "confidence score between 0 and 1"
        }}

        Please ensure the classification is based on the document content and the provided keywords."""

    def _create_extraction_prompt(self) -> str:
        """Create the prompt for document extraction.
        
        Args:
            extracted_text: Text extracted from the document

        """
        return f"""You are a document processing assistant. Given a document image, perform the following task:

            1. Extract all text and tables from the document and format it in clean, readable markdown.

            Return your response in the following JSON format:
            {{
                "extracted_text": "markdown formatted text from the document"
            }}"""
            
        
    
    async def preprocess_document(self, document_path: str) -> Dict[str, Any]:
        """Preprocess a document to extract text, detect language, and classify type.
        
        Args:
            document_path: Path to the document image
            
        Returns:
            Dictionary containing:
            - extracted_text: Extracted text from the document
            - detected_language: ISO language code
            - document_type: Classified document type
            - confidence_score: Confidence in classification
        """
        image_paths = self._load_pdf_to_image(document_path)

        # Process each page sequentially
        #for image_path in image_paths:
        #    page_text = await self.model.process_document(image_path, extraction_prompt)
        #    if page_text:
        #        all_extracted_text.append(page_text)
        
        # Combine text from all pages
        #extracted_text = "\n\n".join(all_extracted_text)


        # Step 1: Extract text from the document
        extraction_prompt = self._create_extraction_prompt()
        extracted_text = await self.model.process_document(image_paths[0], extraction_prompt)
        extracted_text = " ".join(extracted_text)
        torch.cuda.empty_cache()
        # Step 2: Classify document and detect language
        classification_prompt = self._create_classification_prompt(extracted_text)
        classification_result = await self.model.process_document(None, classification_prompt)
        classification_result = " ".join(classification_result)
        
        torch.cuda.empty_cache()
        try:
            classification_data = json.loads(classification_result)
            return {
                "extracted_text": extracted_text,
                "detected_language": classification_data.get("detected_language"),
                "document_type": classification_data.get("document_type"),
                "confidence_score": classification_data.get("confidence_score", 0.0)
            }
        except json.JSONDecodeError:
            # If the result isn't valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\{.*\}', classification_result, re.DOTALL)
            if json_match:
                classification_data = json.loads(json_match.group())
                return {
                    "extracted_text": extracted_text,
                    "detected_language": classification_data.get("detected_language"),
                    "document_type": classification_data.get("document_type"),
                    "confidence_score": classification_data.get("confidence_score", 0.0)
                }
            raise ValueError("Could not parse classification result as JSON") 