from typing import Dict, Any
import json
import os
import base64
from mistralai import Mistral
from .base import BaseModel

class MistralModel(BaseModel):
    def __init__(self, api_key: str):
        """Initialize Mistral model with API configuration."""
        super().__init__(api_key)
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-ocr-latest"

    def _encode_image(self, image_path: str) -> str:
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {image_path} was not found.")
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")

    async def process_document(self, document: Dict[str, Any]) -> str:
        """Process a document using Mistral OCR model.
        
        Args:
            document: Dictionary containing document data including:
                - image_path: Path to the document image
                
        Returns:
            String containing the extracted key details
        """
        
        try:
            # Get the full image path
            image_path = '/content/' + document["image_path"]
            
            # Encode the image to base64
            base64_image = self._encode_image(image_path)
            
            # Make OCR API call
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            )
            
            # Combine markdown from all pages
            combined_markdown = "\n\n".join(page.markdown for page in ocr_response.pages)
            
            # Postprocess to extract key details
            key_details = await self._extract_key_details(combined_markdown)
            
            return key_details
                
        except Exception as e:
            return str(e)

    async def _extract_key_details(self, text: str) -> str:
        """Extract key details from the document text and organize them into bullet points.
        
        Args:
            text: The combined markdown text from OCR
            
        Returns:
            String containing organized bullet points of key details
        """
        try:
            # Create prompt for key details extraction
            prompt = f"""**Key Details**: Extract all the important and readable information from the document and organize it into clear and concise bullet points.

            Document text:
            {text}"""

            # Make API call to get key details
            chat_response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Return the response content
            return chat_response.choices[0].message.content
            
        except Exception as e:
            return f"Error extracting key details: {str(e)}"

    def _create_prompt(self, text: str) -> str:
        """Create a prompt for document processing."""
        return f"""Analyze this document and provide the following information in JSON format:
{{
    "document_type": "Type of document (e.g., invoice, receipt, contract)",
    "language": "Language of the document",
    "key_information": {{
        "issuer": "Who issued the document",
        "recipient": "Who is the document for",
        "date": "Document date",
        "amount": "Monetary amount if present",
        "reference": "Document reference number",
        "additional_info": "Any other important information"
    }},
    "confidence_score": "Confidence score between 0 and 1"
}}

Document text:
{text}""" 