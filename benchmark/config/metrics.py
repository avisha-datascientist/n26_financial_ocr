from dataclasses import dataclass
from typing import Dict, Any
import re
from difflib import SequenceMatcher

@dataclass
class EvaluationMetrics:
    key_details_accuracy: float
    key_details_completeness: float
    key_details_relevance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "key_details_accuracy": self.key_details_accuracy,
            "key_details_completeness": self.key_details_completeness,
            "key_details_relevance": self.key_details_relevance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvaluationMetrics':
        return cls(**data)

class MetricsConfig:
    WEIGHTS = {
        "key_details_accuracy": 0.4,
        "key_details_completeness": 0.3,
        "key_details_relevance": 0.3
    }

    @classmethod
    def calculate_weighted_score(cls, metrics: EvaluationMetrics) -> float:
        metrics_dict = metrics.to_dict()
        return sum(metrics_dict[key] * weight for key, weight in cls.WEIGHTS.items())

    @classmethod
    def calculate_metrics(cls, prediction: str, ground_truth: str) -> EvaluationMetrics:
        """Calculate metrics for key details extraction.
        
        Args:
            prediction: The extracted key details from the model
            ground_truth: The expected key details
            
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
        
        return EvaluationMetrics(
            key_details_accuracy=accuracy,
            key_details_completeness=completeness,
            key_details_relevance=relevance
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
        """Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Use SequenceMatcher for text similarity
        return SequenceMatcher(None, text1, text2).ratio() 