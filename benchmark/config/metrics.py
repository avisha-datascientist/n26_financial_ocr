from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvaluationMetrics:
    accuracy: float
    completeness: float
    format_adherence: float
    language_handling: float
    processing_time: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "format_adherence": self.format_adherence,
            "language_handling": self.language_handling,
            "processing_time": self.processing_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvaluationMetrics':
        return cls(**data)

class MetricsConfig:
    WEIGHTS = {
        "accuracy": 0.3,
        "completeness": 0.25,
        "format_adherence": 0.2,
        "language_handling": 0.15,
        "processing_time": 0.1
    }

    @classmethod
    def calculate_weighted_score(cls, metrics: EvaluationMetrics) -> float:
        metrics_dict = metrics.to_dict()
        return sum(metrics_dict[key] * weight for key, weight in cls.WEIGHTS.items()) 