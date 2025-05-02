from typing import Dict, List
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

class BenchmarkVisualizer:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_results(self, results: Dict):
        """Save benchmark results to JSON file"""
        output_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def plot_metrics_comparison(self, model_metrics: Dict[str, List[float]], metric_names: List[str]):
        """Create bar plot comparing metrics across models"""
        df = pd.DataFrame(model_metrics, index=metric_names)
        
        plt.figure(figsize=(10, 6))
        df.plot(kind="bar")
        plt.title("Model Performance Comparison")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.legend(title="Models")
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"metrics_comparison_{self.timestamp}.png")
        plt.close()

    def plot_processing_times(self, processing_times: Dict[str, List[float]]):
        """Create box plot of processing times"""
        plt.figure(figsize=(8, 6))
        plt.boxplot(processing_times.values(), labels=processing_times.keys())
        plt.title("Processing Time Distribution")
        plt.xlabel("Models")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"processing_times_{self.timestamp}.png")
        plt.close()

    def plot_language_performance(self, language_scores: Dict[str, Dict[str, float]]):
        """Create heatmap of model performance across languages"""
        df = pd.DataFrame(language_scores)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(df, cmap="YlOrRd", aspect="auto")
        plt.colorbar(label="Score")
        plt.title("Language Performance Heatmap")
        plt.xticks(range(len(df.columns)), df.columns, rotation=45)
        plt.yticks(range(len(df.index)), df.index)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"language_performance_{self.timestamp}.png")
        plt.close()

    def generate_summary_report(self, results: Dict):
        """Generate a markdown summary report"""
        report = f"""# Benchmark Results Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Performance

"""
        for model, metrics in results["model_metrics"].items():
            report += f"### {model}\n"
            for metric, value in metrics.items():
                report += f"- {metric}: {value:.3f}\n"
            report += "\n"

        report += """## Processing Time Statistics\n\n"""
        for model, times in results["processing_times"].items():
            avg_time = sum(times) / len(times)
            report += f"- {model}: {avg_time:.3f}s (avg)\n"

        report_file = self.output_dir / f"benchmark_report_{self.timestamp}.md"
        with open(report_file, "w") as f:
            f.write(report) 