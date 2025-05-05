from typing import Dict, List
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

class BenchmarkVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_results(self, results: Dict):
        """Save benchmark results to JSON file"""
        # Convert metrics objects to dictionaries
        for model in ["mistral", "qwen"]:
            results[model]["metrics"] = [m.to_dict() for m in results[model]["metrics"]]
        
        output_file = self.output_dir / f"results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def plot_metrics_comparison(self, metrics_data: Dict[str, List[Dict[str, float]]], metrics: List[str]):
        """Plot comparison of metrics between models."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            data = []
            labels = []
            for model, model_metrics in metrics_data.items():
                data.append([m[metric] for m in model_metrics])
                labels.append(model)
            
            ax.boxplot(data, labels=labels)
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylabel("Score")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"metrics_comparison_{self.timestamp}.png")
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