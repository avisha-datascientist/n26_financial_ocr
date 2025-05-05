from dataclasses import dataclass
from typing import Dict, Any, Optional
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class EvaluationMetrics:
    key_details_accuracy: float
    key_details_completeness: float
    key_details_relevance: float
    embedding_accuracy: Optional[float] = None  # Added for benchmarking

    def __init__(self, key_details_accuracy: float, key_details_completeness: float, key_details_relevance: float, embedding_accuracy: Optional[float] = None):
        self.key_details_accuracy = key_details_accuracy
        self.key_details_completeness = key_details_completeness
        self.key_details_relevance = key_details_relevance
        self.embedding_accuracy = embedding_accuracy

    def to_dict(self) -> Dict[str, float]:
        metrics_dict = {
            "key_details_accuracy": self.key_details_accuracy,
            "key_details_completeness": self.key_details_completeness,
            "key_details_relevance": self.key_details_relevance
        }
        if self.embedding_accuracy is not None:
            metrics_dict["embedding_accuracy"] = self.embedding_accuracy
        return metrics_dict

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvaluationMetrics':
        # Handle missing embedding_accuracy in older data
        if "embedding_accuracy" not in data:
            data["embedding_accuracy"] = None
        return cls(**data)

class MetricsConfig:
    WEIGHTS = {
        "key_details_accuracy": 0.4,
        "key_details_completeness": 0.3,
        "key_details_relevance": 0.3
    }

    # Initialize sentence transformer for embedding-based similarity
    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._embedding_model

    @classmethod
    def calculate_weighted_score(cls, metrics: EvaluationMetrics) -> float:
        metrics_dict = metrics.to_dict()
        return sum(metrics_dict[key] * weight for key, weight in cls.WEIGHTS.items())

    @classmethod
    def calculate_metrics(cls, prediction: str, ground_truth: str, use_embeddings: bool = False) -> EvaluationMetrics:
        """Calculate metrics for key details extraction.
        
        Args:
            prediction: The extracted key details from the model
            ground_truth: The expected key details
            use_embeddings: Whether to use embedding-based similarity for benchmarking
            
        Returns:
            EvaluationMetrics object with calculated scores
        """
        # Normalize texts for comparison
        prediction = cls._normalize_text(prediction)
        ground_truth = cls._normalize_text(ground_truth)
        
        # Calculate accuracy using sequence matching
        accuracy = SequenceMatcher(None, prediction, ground_truth).ratio()
        
        # Calculate completeness based on bullet points
        pred_points = cls._count_bullet_points(prediction)
        gt_points = cls._count_bullet_points(ground_truth)
        completeness = min(pred_points / max(gt_points, 1), 1.0)
        
        # Calculate relevance (placeholder - could be enhanced with semantic similarity)
        relevance = 0.8  # This could be improved with more sophisticated analysis

        # Calculate embedding-based accuracy if requested
        embedding_accuracy = None
        if use_embeddings:
            embedding_accuracy = cls._calculate_embedding_similarity(prediction, ground_truth)
        
        return EvaluationMetrics(
            key_details_accuracy=accuracy,
            key_details_completeness=completeness,
            key_details_relevance=relevance,
            embedding_accuracy=embedding_accuracy
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    @staticmethod
    def _count_bullet_points(text: str) -> int:
        """Count the number of bullet points in the text."""
        # Count lines that start with bullet points or dashes
        return len([line for line in text.split('\n') 
                   if line.strip().startswith(('-', '*', '•'))])

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using SequenceMatcher.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()

    @classmethod
    def _calculate_embedding_similarity(cls, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using embeddings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        model = cls._get_embedding_model()
        
        # Get embeddings for both texts
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        # Move tensors to CPU and convert to numpy
        embedding1 = embedding1.cpu().numpy()
        embedding2 = embedding2.cpu().numpy()
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Convert to float and ensure it's between 0 and 1
        return float(max(0.0, min(1.0, similarity))) 