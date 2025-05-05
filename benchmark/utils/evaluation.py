from typing import Dict, List, Optional
import json
from benchmark.config.metrics import EvaluationMetrics, MetricsConfig

def evaluate_key_details_accuracy(prediction: str, ground_truth: str, use_embeddings: bool = False) -> float:
    """Evaluate accuracy of key details extraction"""
    if not prediction or not ground_truth:
        return 0.0
    
    # Split into individual points
    pred_points = [p.strip() for p in prediction.split('\n') if p.strip()]
    true_points = [p.strip() for p in ground_truth.split('\n') if p.strip()]
    
    if not true_points:
        return 1.0  # If no ground truth points, consider it perfect
    
    # Calculate accuracy for each point
    total_accuracy = 0.0
    for true_point in true_points:
        best_match = max(
            (MetricsConfig._normalize_text(pred_point), 
             MetricsConfig._normalize_text(true_point))
            for pred_point in pred_points
        )
        if use_embeddings:
            total_accuracy += MetricsConfig._calculate_embedding_similarity(best_match[0], best_match[1])
        else:
            total_accuracy += MetricsConfig._calculate_similarity(best_match[0], best_match[1])
    
    return total_accuracy / len(true_points)

def evaluate_key_details_completeness(prediction: str, ground_truth: str) -> float:
    """Evaluate completeness of key details extraction"""
    if not prediction or not ground_truth:
        return 0.0
    
    # Count bullet points in prediction and ground truth
    pred_points = MetricsConfig._count_bullet_points(prediction)
    true_points = MetricsConfig._count_bullet_points(ground_truth)
    
    if true_points == 0:
        return 1.0  # If no ground truth points, consider it complete
    
    # Calculate completeness as ratio of points found
    return min(pred_points / true_points, 1.0)

def evaluate_key_details_relevance(prediction: str, ground_truth: str) -> float:
    """Evaluate relevance of extracted key details"""
    if not prediction or not ground_truth:
        return 0.0
    
    # Split into individual points
    pred_points = [p.strip() for p in prediction.split('\n') if p.strip()]
    true_points = [p.strip() for p in ground_truth.split('\n') if p.strip()]
    
    if not true_points:
        return 1.0  # If no ground truth points, consider it perfect
    
    # Calculate relevance score based on semantic similarity
    total_relevance = 0.0
    for pred_point in pred_points:
        best_match = max(
            MetricsConfig._calculate_similarity(
                MetricsConfig._normalize_text(pred_point),
                MetricsConfig._normalize_text(true_point)
            )
            for true_point in true_points
        )
        total_relevance += best_match
    
    return total_relevance / len(pred_points)

def calculate_metrics(
    prediction: str,
    ground_truth: str,
    use_embeddings: bool = False
) -> EvaluationMetrics:
    """Calculate all evaluation metrics for key details extraction"""
    
    # Calculate individual metrics
    accuracy = evaluate_key_details_accuracy(prediction, ground_truth, use_embeddings)
    completeness = evaluate_key_details_completeness(prediction, ground_truth)
    relevance = evaluate_key_details_relevance(prediction, ground_truth)
    
    # Create metrics object
    metrics = EvaluationMetrics(
        key_details_accuracy=accuracy * 100,  # Convert to percentage
        key_details_completeness=completeness * 100,
        key_details_relevance=relevance * 100,
        embedding_accuracy=accuracy * 100 if use_embeddings else None  # Set embedding accuracy if requested
    )
    
    return metrics

def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> Dict[str, float]:
    """Aggregate metrics across multiple documents"""
    if not metrics_list:
        return {}
    
    aggregated = {
        "key_details_accuracy_mean": sum(m.key_details_accuracy for m in metrics_list) / len(metrics_list),
        "key_details_completeness_mean": sum(m.key_details_completeness for m in metrics_list) / len(metrics_list),
        "key_details_relevance_mean": sum(m.key_details_relevance for m in metrics_list) / len(metrics_list),
        "weighted_score_mean": sum(
            MetricsConfig.calculate_weighted_score(m) for m in metrics_list
        ) / len(metrics_list)
    }
    
    # Add embedding accuracy if available
    if metrics_list[0].embedding_accuracy is not None:
        aggregated["embedding_accuracy_mean"] = sum(
            m.embedding_accuracy for m in metrics_list
        ) / len(metrics_list)
    
    return aggregated 