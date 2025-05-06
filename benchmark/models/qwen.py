from typing import Dict, Any, Union
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
        print("Loading Qwen model Qwen/Qwen2.5-VL-32B-Instruct...")  # Using 7B instead of 32B
        
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
            "Qwen/Qwen2.5-VL-32B-Instruct",
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
        print("Model loaded successfully!")

    async def process_document(self, document: Union[str, Dict[str, Any]], prompt: str = None) -> str:
        """Process a document using Qwen model.
        
        Args:
            document: Document to process with image path and text
            
        Returns:
            String containing extracted key details
        """
        # First prompt to extract table in markdown format
        table_prompt = """You are a highly accurate Optical Character Recognition (OCR) and text and table extraction assistant. Given an image that contains a text and table or tables (including scanned documents, photos of printed pages, screenshots, etc.), your task is to:\
        1. Visually understand the structure of the document, identifying all relevant blocks such as headers, sender/client info, tables, and totals\
        2. Extract and summarize all key information into a clean, readable list of bullet point\
        3. Organize the bullet points into meaningful categories\
        4. Follow this formatting guideline for the output:\
            -- Use plain text bullet points (with -)\
            -- Use bold labels (with **) for key-value style fields\
            -- For line items or repeating entries, describe them briefly rather than listing all rows unless necessary\
            -- Do not return raw tables or Markdown — return a summary in clean bullet points\

        5. Do not include any explanations, notes, or OCR artifacts. Only return the final structured summary."""
          
        try:
          if document is not None and isinstance(document, str):
            # Handle string path case
            image_path = document
          elif document is not None and isinstance(document, dict):
            # Handle dictionary case
            image_path = document.get("image_path")
        else:
            image_path = ""


          if prompt:
            if len(image_path) > 0:
              messages = [
              {
                  "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": document
                    },
                    {"type": "text", "text": prompt},
                ]}]
            else:
                messages = [
                {
                  "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]}]
          else:
            prompt = table_prompt
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": '/content/' + document["image_path"]
                    },
                    {"type": "text", "text": prompt},
                ]
            }]

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
          )

          # for benchmark purposes only
          #key_details = self._extract_key_details(table_text)

          return table_text

        except Exception as e:
            return f"Error processing document: {str(e)}"

    async def _extract_key_details(self, table_text: str) -> str:

        # Second prompt to extract key details from the table
        key_details_prompt = f"""**Key Details**: Extract all the important and readable information from the table and organize it into clear and concise bullet points.
        Return information in bullet points only, nothing else.
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        key_details = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return key_details

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