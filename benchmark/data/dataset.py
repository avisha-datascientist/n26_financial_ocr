from datasets import load_dataset
from typing import List, Dict, Optional
import random
import os
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from ..config.config import DatasetConfig

class BenchmarkDataset:
    def __init__(self, num_samples: int = DatasetConfig.NUM_SAMPLES):
        self.num_samples = num_samples
        self.dataset = None
        self.filtered_data = None
        self.image_dir = "benchmark_images"

    def load_data(self) -> List[Dict]:
        """Load and process the Sujet-Finance-Vision-10k dataset"""
        # Load dataset
        dataset = load_dataset("sujet-ai/Sujet-Finance-Vision-1Ok")
        
        # Convert to DataFrame and sample
        df = dataset['train'].to_pandas()
        if len(df) > self.num_samples:
            df = df.sample(n=self.num_samples, random_state=42)

        # Create image directory
        os.makedirs(self.image_dir, exist_ok=True)

        # Process and save images
        processed_data = []
        for _, row in df.iterrows():
            try:
                # Save image and get path
                image_path = self._process_image(row)
                
                # Create document entry
                doc_entry = {
                    "image_path": image_path,
                    "doc_id": row["doc_id"],
                    "content": row["content"],
                    "document_type": row["document_type"],
                    "key_details": row["key_details"],
                    "insights": row["insights"],
                    "ground_truth": {
                        "document_type": row["document_type"],
                        "key_information": {
                            "key_details": row["key_details"],
                            "insights": row["insights"]
                        }
                    }
                }
                processed_data.append(doc_entry)
                
            except Exception as e:
                print(f"Error processing document {row['doc_id']}: {str(e)}")
                continue

        self.filtered_data = processed_data
        return self.filtered_data

    def _process_image(self, row: pd.Series) -> str:
        """Process and save a single image"""
        doc_id = row['doc_id']
        encoded_image = row['encoded_image']
        
        # Decode base64 image
        decoded_image = base64.b64decode(encoded_image)
        
        # Save image path
        image_path = os.path.join(self.image_dir, f"{doc_id}.jpg")
        
        # Save image using PIL
        with Image.open(BytesIO(decoded_image)) as img:
            img.save(image_path, 'JPEG')
        
        return image_path

    def get_document_types(self) -> List[str]:
        """Get unique document types in the dataset"""
        if self.filtered_data:
            return list(set(doc["document_type"] for doc in self.filtered_data))
        return []

    def get_sample_by_type(self, doc_type: str) -> Optional[Dict]:
        """Get a random sample document of specified type"""
        if self.filtered_data:
            samples = [doc for doc in self.filtered_data if doc["document_type"] == doc_type]
            return random.choice(samples) if samples else None
        return None 