from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ModelConfig:
    MISTRAL_API_KEY: str = "your-mistral-api-key"
    QWEN_API_KEY: str = "your-qwen-api-key"
    GPT4_API_KEY: str = "your-gpt4-api-key"
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    BATCH_SIZE: int = 10

@dataclass
class DatasetConfig:
    NUM_SAMPLES: int = 100
    SUPPORTED_LANGUAGES: Tuple[str, ...] = ("English", "German", "French", "Spanish", "Italian")
    DOCUMENT_TYPES: Tuple[str, ...] = ("credit", "investment", "personal_account", "garnishment") 