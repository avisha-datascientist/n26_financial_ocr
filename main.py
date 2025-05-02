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
    visualizer = BenchmarkVisualizer(output_dir='/Users/avishabhiryani/Documents/private/N26_GenAI_Take_Home_Assignment/benchmark/results')
    
    # Load dataset
    print("\nLoading and processing dataset...")
    documents = dataset.load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Initialize results storage
    results = {
        "mistral": {
            "predictions": [],
            "metrics": [],
            "processing_times": []
        },
        "qwen": {
            "predictions": [],
            "metrics": [],
            "processing_times": []
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
            mistral_prediction, mistral_time = await mistral_model.process_document(doc)
            mistral_metrics = calculate_metrics(
                prediction=mistral_prediction,
                ground_truth=doc["ground_truth"],
                processing_time=mistral_time,
                content=doc["content"]
            )
            
            results["mistral"]["predictions"].append(mistral_prediction)
            results["mistral"]["metrics"].append(mistral_metrics)
            results["mistral"]["processing_times"].append(mistral_time)
            
        except Exception as e:
            print(f"\nError processing document {doc['doc_id']} with Mistral: {str(e)}")
        
        # Process with Qwen
        try:
            qwen_prediction, qwen_time = await qwen_model.process_document(doc)
            qwen_metrics = calculate_metrics(
                prediction=qwen_prediction,
                ground_truth=doc["ground_truth"],
                processing_time=qwen_time,
                content=doc["content"]
            )
            
            results["qwen"]["predictions"].append(qwen_prediction)
            results["qwen"]["metrics"].append(qwen_metrics)
            results["qwen"]["processing_times"].append(qwen_time)
            
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
        ["accuracy", "completeness", "format_adherence", "language_handling", "processing_time"]
    )
    
    visualizer.plot_processing_times({
        "Mistral": results["mistral"]["processing_times"],
        "Qwen": results["qwen"]["processing_times"]
    })
    
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
    # Load Mistral API key from environment variable
    # Run benchmark
    asyncio.run(run_benchmark(
        num_samples=2,  # Number of documents to process
        output_dir="benchmark_results"  # Output directory for results
    )) 