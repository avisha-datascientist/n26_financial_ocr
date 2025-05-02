from typing import Dict, List, Optional
import json
from benchmark.config.metrics import EvaluationMetrics, MetricsConfig

def evaluate_document_type_accuracy(prediction: Dict, ground_truth: Dict) -> float:
    """Evaluate accuracy of document type prediction"""
    pred_type = prediction.get("document_type", "").lower()
    true_type = ground_truth.get("document_type", "").lower()
    return 1.0 if pred_type == true_type else 0.0

def evaluate_key_information_completeness(prediction: Dict, ground_truth: Dict) -> float:
    """Evaluate completeness of key information extraction"""
    if not prediction or "key_information" not in prediction:
        return 0.0
    
    # Get ground truth key details and insights
    true_details = set(str(item).lower() for item in ground_truth["key_information"]["key_details"])
    true_insights = set(str(item).lower() for item in ground_truth["key_information"]["insights"])
    
    # Get predicted key details and insights
    pred_info = prediction.get("key_information", {})
    pred_details = set(str(item).lower() for item in pred_info.get("key_details", []))
    pred_insights = set(str(item).lower() for item in pred_info.get("insights", []))
    
    # Calculate overlap ratios
    details_ratio = len(pred_details.intersection(true_details)) / len(true_details) if true_details else 1.0
    insights_ratio = len(pred_insights.intersection(true_insights)) / len(true_insights) if true_insights else 1.0
    
    # Weight the scores (giving more weight to key details)
    return 0.7 * details_ratio + 0.3 * insights_ratio

def evaluate_format_adherence(prediction: Dict) -> float:
    """Evaluate if the output adheres to expected JSON format"""
    required_fields = {"document_type", "key_information"}
    
    if not prediction:
        return 0.0
    
    # Check main structure
    format_score = sum(1 for field in required_fields if field in prediction) / len(required_fields)
    
    # Check key_information structure
    if "key_information" in prediction:
        key_info = prediction["key_information"]
        if isinstance(key_info, dict) and "key_details" in key_info and "insights" in key_info:
            format_score = (format_score + 1) / 2
    
    return format_score

def evaluate_language_handling(prediction: Dict, content: str) -> float:
    """Evaluate model's ability to handle document content"""
    if not prediction or not content:
        return 0.0
    
    # Check if key information contains relevant content
    key_info = prediction.get("key_information", {})
    all_extracted_text = " ".join([
        str(item) for item in key_info.get("key_details", []) + key_info.get("insights", [])
    ]).lower()
    
    # Calculate content coverage
    content_words = set(content.lower().split())
    extracted_words = set(all_extracted_text.split())
    
    if not content_words:
        return 0.0
    
    return len(extracted_words.intersection(content_words)) / len(content_words)

def calculate_metrics(
    prediction: Dict,
    ground_truth: Dict,
    processing_time: float,
    content: str,
    max_processing_time: float = 10.0  # Maximum acceptable processing time in seconds
) -> EvaluationMetrics:
    """Calculate all evaluation metrics for a single prediction"""
    
    # Calculate individual metrics
    accuracy = evaluate_document_type_accuracy(prediction, ground_truth)
    completeness = evaluate_key_information_completeness(prediction, ground_truth)
    format_adherence = evaluate_format_adherence(prediction)
    language_handling = evaluate_language_handling(prediction, content)
    
    # Normalize processing time score (lower is better)
    processing_time_score = max(0, 1 - (processing_time / max_processing_time))
    
    # Create metrics object
    metrics = EvaluationMetrics(
        accuracy=accuracy * 100,  # Convert to percentage
        completeness=completeness * 100,
        format_adherence=format_adherence * 100,
        language_handling=language_handling * 100,
        processing_time=processing_time_score * 100
    )
    
    return metrics

def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
    """Aggregate metrics across multiple documents"""
    if not metrics_list:
        return {}
    
    aggregated = {
        "accuracy_mean": sum(m.accuracy for m in metrics_list) / len(metrics_list),
        "completeness_mean": sum(m.completeness for m in metrics_list) / len(metrics_list),
        "format_adherence_mean": sum(m.format_adherence for m in metrics_list) / len(metrics_list),
        "language_handling_mean": sum(m.language_handling for m in metrics_list) / len(metrics_list),
        "processing_time_mean": sum(m.processing_time for m in metrics_list) / len(metrics_list),
        "weighted_score_mean": sum(
            MetricsConfig.calculate_weighted_score(m) for m in metrics_list
        ) / len(metrics_list)
    }
    
    return aggregated 