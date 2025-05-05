import os
import asyncio
from datetime import datetime
from typing import Dict, List
import json
from tqdm import tqdm

from benchmark.data.dataset import BenchmarkDataset
from benchmark.models.mistral import MistralModel
from benchmark.models.qwen import QwenModel
from benchmark.utils.evaluation import calculate_metrics, aggregate_metrics
from benchmark.utils.visualization import BenchmarkVisualizer

async def run_benchmark(
    num_samples: int = 100,
    output_dir: str = "benchmark_results"
):
    print("Starting benchmark process...")
    
    # Initialize components
    dataset = BenchmarkDataset(num_samples=num_samples)
    mistral_model = MistralModel(api_key='vh6EK34LkmpEkBjvXsuVG84v1Sh4O0tu')
    qwen_model = QwenModel()  # No API key needed for Qwen
    visualizer = BenchmarkVisualizer(output_dir=output_dir)
    
    # Load dataset
    print("\nLoading and processing dataset...")
    documents = dataset.load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Initialize results storage
    results = {
        "mistral": {
            "predictions": [],
            "metrics": []
        },
        "qwen": {
            "predictions": [],
            "metrics": []
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(documents),
            "document_types": dataset.get_document_types()
        }
    }
    
    # Process documents
    print("\nProcessing documents with both models...")
    for doc in tqdm(documents, desc="Processing documents"):
        # Process with Mistral
        try:
            mistral_result = await mistral_model.process_document(doc)
            
            mistral_metrics = calculate_metrics(
                prediction=mistral_result,
                ground_truth=doc["key_details"]
            )
            
            results["mistral"]["predictions"].append(mistral_result)
            results["mistral"]["metrics"].append(mistral_metrics)
            
        except Exception as e:
            print(f"\nError processing document {doc['doc_id']} with Mistral: {str(e)}")
        
        # Process with Qwen
        try:
            qwen_result = await qwen_model.process_document(doc)
            
            qwen_metrics = calculate_metrics(
                prediction=qwen_result,
                ground_truth=doc["key_details"]
            )
            
            results["qwen"]["predictions"].append(qwen_result)
            results["qwen"]["metrics"].append(qwen_metrics)
            
        except Exception as e:
            print(f"\nError processing document {doc['doc_id']} with Qwen: {str(e)}")
    
    # Aggregate results
    print("\nAggregating results...")
    results["mistral"]["aggregated"] = aggregate_metrics(results["mistral"]["metrics"])
    results["qwen"]["aggregated"] = aggregate_metrics(results["qwen"]["metrics"])
    
    # Generate visualizations and reports
    print("\nGenerating visualizations and reports...")
    visualizer.save_results(results)
    
    visualizer.plot_metrics_comparison(
        {
            "Mistral": [m.to_dict() for m in results["mistral"]["metrics"]],
            "Qwen": [m.to_dict() for m in results["qwen"]["metrics"]]
        },
        ["key_details_accuracy", "key_details_completeness", "key_details_relevance"]
    )
    
    # Print summary
    print("\nBenchmark Summary:")
    print("\nMistral Results:")
    for metric, value in results["mistral"]["aggregated"].items():
        print(f"{metric}: {value:.2f}")
    
    print("\nQwen Results:")
    for metric, value in results["qwen"]["aggregated"].items():
        print(f"{metric}: {value:.2f}")
    
    print(f"\nDetailed results and visualizations saved to: {output_dir}")
    return results

if __name__ == "__main__":
    # Run benchmark
    asyncio.run(run_benchmark(
        num_samples=2,  # Number of documents to process
        output_dir="/content/n26_financial_ocr/benchmark/results"  # Output directory for results
    )) 