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

    async def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document using Qwen model.
        
        Args:
            document: Document to process with image path and text
            
        Returns:
            Dictionary containing processing results
        """
        prompt = "You are a highly accurate Optical Character Recognition (OCR) and table extraction assistant. Given an image that contains a table (including scanned documents, photos of printed pages, screenshots, etc.), your task is to: \
        1. Recognize and extract all the text in the image. \
        2. Identify and interpret the tabular structure of the content, including: \
        3. Row and column boundaries \
        4. Header rows (if any) \
        5. Merged or spanned cells \
        6. Numerical and textual data \
        7. Output the table in clean and readable Markdown format (or CSV if specified), preserving the correct structure and cell values. \
        8. Handle cases where the table has borders or no borders, or if the text is slightly rotated or imperfectly aligned. \
        9. If the table contains multiple sections or subtables, separate them clearly and label them accordingly (e.g., “Table 1”, “Table 2”). \
        If any parts of the table are unclear or unreadable, indicate them using [[UNREADABLE]]."
        
        try:
            
            # Create messages for the model using actual document content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": '/content/' + document["image_path"]
                        },
                        {"type": "text", "text": "Describe the image in detail"},
                    ]
                }
            ]
            print("after messages")

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("after text")
            image_inputs, video_inputs = process_vision_info(messages)
            print("after image_inputs")
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            print("after inputs")
            print("is cude", self.model.device)
            inputs = inputs.to(self.model.device)
            print("after inputs to device")
            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=120)
            print("after generate")
            print(generated_ids)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            print("after generated_ids_trimmed")
            print(generated_ids_trimmed)
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(output_text)
            # Parse the output
            
            return output_text

        except Exception as e:
            return str(e)

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