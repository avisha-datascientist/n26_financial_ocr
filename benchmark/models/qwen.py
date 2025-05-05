from typing import Dict, Any
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import time
import json
import os
from .base import BaseModel

class QwenModel(BaseModel):
    def __init__(self, device: str = "auto", use_flash_attention: bool = False):
        """Initialize Qwen model with Hugging Face Transformers.
        
        Args:
            device: Device to run the model on ("auto", "cuda", "cpu")
            use_flash_attention: Whether to use Flash Attention 2 (requires flash-attn package)
        """
        print("Loading Qwen model Qwen/Qwen2.5-VL-7B-Instruct...")  # Using 7B instead of 32B
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Try to use Flash Attention 2 if requested
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True  # Optimize memory usage
        }
        
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2 for better performance")
            except ImportError:
                print("Flash Attention 2 not available, falling back to standard attention")
                use_flash_attention = False
        
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("Model loaded successfully!")

    async def process_document(self, document: Dict[str, Any]) -> str:
        """Process a document using Qwen model.
        
        Args:
            document: Document to process with image path and text
            
        Returns:
            String containing extracted key details
        """
        # First prompt to extract table in markdown format
        table_prompt = """You are a highly accurate Optical Character Recognition (OCR) and table extraction assistant. Given an image that contains a table (including scanned documents, photos of printed pages, screenshots, etc.), your task is to:

1. Recognize and extract all the text in the image
2. Identify and interpret the tabular structure of the content
3. Output the table in clean and readable Markdown format, preserving the correct structure and cell values
4. Handle cases where the table has borders or no borders
5. If any parts of the table are unclear or unreadable, indicate them using [[UNREADABLE]]

Please return ONLY the markdown table, nothing else."""
        
        try:
            # First call to get the table in markdown format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": '/content/' + document["image_path"]
                        },
                        {"type": "text", "text": table_prompt},
                    ]
                }
            ]

            # Prepare inputs for table extraction
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate table output
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            table_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Second prompt to extract key details from the table
            key_details_prompt = f"""**Key Details**: Extract all the important and readable information from the table and organize it into clear and concise bullet points.

Table:
{table_text}"""

            # Prepare inputs for key details extraction
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": key_details_prompt},
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate key details output
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            key_details = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return key_details

        except Exception as e:
            return f"Error processing document: {str(e)}"

    def _create_prompt(self, text: str) -> str:
        """Create a prompt for document processing.
        
        Args:
            text: Document text to process
            
        Returns:
            Formatted prompt string
        """
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