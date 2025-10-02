import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import cv2
from ultralytics import YOLO
import torch


@dataclass
class ModelMetrics:
    """Store comprehensive metrics for a YOLO model"""
    model_name: str
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time: float = 0.0
    fps: float = 0.0
    model_size_mb: float = 0.0
    total_params: int = 0
    predictions: List = None

    def __post_init__(self):
        if self.predictions is None:
            self.predictions = []


class YOLOABTester:
    """A/B Testing framework for YOLO models"""

    def __init__(self, dataset_path: str = None, test_data_path: str = None, confidence_threshold: float = 0.25):

        if dataset_path:
            self.dataset_path = Path(dataset_path)
            # Fix: Look for test images in the correct structure
            possible_test_paths = [
                self.dataset_path / "test" / "images",
                self.dataset_path / "images" / "test",
                self.dataset_path / "images",
                self.dataset_path
            ]

            self.test_data_path = None
            for path in possible_test_paths:
                if path.exists() and any(path.glob("*.jpg")) or any(path.glob("*.png")):
                    self.test_data_path = path
                    print(f" Found test images in: {self.test_data_path}")
                    break

            if not self.test_data_path:
                raise ValueError(f"No test images found in any of: {possible_test_paths}")

        elif test_data_path:
            self.test_data_path = Path(test_data_path)
            self.dataset_path = None
        else:
            raise ValueError("Either dataset_path or test_data_path must be provided")

        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f" Using device: {self.device}")

    def add_model(self, model_path: str, model_name: str):

        try:
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                return False

            model = YOLO(model_path)
            self.models[model_name] = model
            size_mb = self.get_model_size(model_path)
            print(f"✓ Added model: {model_name} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"✗ Failed to load model {model_name}: {e}")
            return False

    def get_model_size(self, model_path: str) -> float:
        try:
            return os.path.getsize(model_path) / (1024 * 1024)
        except:
            return 0.0

    def find_yaml_file(self) -> Optional[str]:
        if not self.dataset_path:
            return None

        possible_yaml_paths = [
            self.dataset_path / "data.yaml",
            self.dataset_path.parent / "data.yaml",
            Path(r"C:\Users\elbek\PyCharmMiscProject\aias-11\data.yaml")
        ]

        for yaml_path in possible_yaml_paths:
            if yaml_path.exists():
                print(f"Found data.yaml: {yaml_path}")
                return str(yaml_path)

        print("⚠️ data.yaml not found in expected locations")
        return None

    def benchmark_inference_speed(self, model: YOLO, test_images: List[str],
                                  num_runs: int = 50) -> Tuple[float, float]:
        if not test_images:
            return 0.0, 0.0

        times = []
        successful_runs = 0

        print(" Warming up model...")
        for _ in range(min(5, len(test_images))):
            try:
                model(test_images[0], verbose=False)
            except:
                continue

        print(f"Running {num_runs} inference tests...")
        for i in range(num_runs):
            img_path = test_images[i % len(test_images)]
            try:
                start_time = time.time()
                model(img_path, verbose=False, conf=self.confidence_threshold)
                end_time = time.time()
                times.append(end_time - start_time)
                successful_runs += 1
            except Exception as e:
                print(f"️Failed inference on {img_path}: {e}")
                continue

        if not times:
            return 0.0, 0.0

        avg_time = np.mean(times)
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"Completed {successful_runs}/{num_runs} successful inferences")
        return avg_time, fps

    def evaluate_model(self, model_name: str, validation_yaml: Optional[str] = None) -> ModelMetrics:
        model = self.models[model_name]

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        test_images = []

        for ext in image_extensions:
            test_images.extend(list(self.test_data_path.glob(ext)))
            test_images.extend(list(self.test_data_path.glob(ext.upper())))

        if not test_images:
            raise ValueError(f"No test images found in {self.test_data_path}")

        test_images = [str(img) for img in test_images[:100]]  # Limit for speed
        print(f"Found {len(test_images)} test images")

        print(f"\nEvaluating {model_name}...")

        metrics = ModelMetrics(model_name=model_name)

        try:
            if hasattr(model.model, 'model'):
                metrics.total_params = sum(p.numel() for p in model.model.parameters())
            else:
                total_params = 0
                for param in model.model.parameters():
                    total_params += param.numel()
                metrics.total_params = total_params
        except:
            metrics.total_params = 0

        print("⚡ Benchmarking inference speed...")
        avg_time, fps = self.benchmark_inference_speed(model, test_images)
        metrics.inference_time = avg_time
        metrics.fps = fps

        if validation_yaml and os.path.exists(validation_yaml):
            print("Running validation...")
            try:
                val_results = model.val(data=validation_yaml, verbose=False, plots=False)

                # Extract metrics with error handling
                if hasattr(val_results, 'box'):
                    metrics.map50 = float(getattr(val_results.box, 'map50', 0))
                    metrics.map50_95 = float(getattr(val_results.box, 'map', 0))
                    metrics.precision = float(getattr(val_results.box, 'mp', 0))
                    metrics.recall = float(getattr(val_results.box, 'mr', 0))

                    if metrics.precision + metrics.recall > 0:
                        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (
                                    metrics.precision + metrics.recall)

                print(f"✅ Validation completed - mAP@0.5: {metrics.map50:.3f}")

            except Exception as e:
                print(f"⚠️ Validation failed: {e}")

        print("Running inference on test set...")
        predictions = []
        confidence_scores = []

        sample_size = min(20, len(test_images))
        for i, img_path in enumerate(test_images[:sample_size]):
            try:
                results = model(img_path, verbose=False, conf=self.confidence_threshold)
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        confs = result.boxes.conf.cpu().numpy()
                        confidence_scores.extend(confs)
                        predictions.append({
                            'image': os.path.basename(img_path),
                            'detections': len(confs),
                            'avg_confidence': np.mean(confs) if len(confs) > 0 else 0
                        })
                    else:
                        predictions.append({
                            'image': os.path.basename(img_path),
                            'detections': 0,
                            'avg_confidence': 0
                        })

                if (i + 1) % 5 == 0:
                    print(f"  Processed {i + 1}/{sample_size} images...")

            except Exception as e:
                print(f"⚠️ Failed to process {os.path.basename(img_path)}: {e}")

        metrics.predictions = predictions
        print(f" Evaluation complete for {model_name}")
        print(f"   • Processed {len(predictions)} images")
        print(f"   • Average detections per image: {np.mean([p['detections'] for p in predictions]):.1f}")

        return metrics

    def run_ab_test(self, model_configs: Dict[str, Dict], use_validation: bool = True):

        print(" Starting YOLO A/B Test\n")
        print(f" Test images directory: {self.test_data_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f" Device: {self.device}")
        print()

        loaded_models = []
        for model_name, config in model_configs.items():
            if self.add_model(config['path'], model_name):
                loaded_models.append(model_name)
                if 'size_mb' not in config:
                    config['size_mb'] = self.get_model_size(config['path'])

        if not loaded_models:
            print(" No models loaded successfully!")
            return

        print(f"Successfully loaded {len(loaded_models)} models\n")

        validation_yaml = None
        if use_validation:
            validation_yaml = self.find_yaml_file()

        for model_name in loaded_models:
            try:
                self.results[model_name] = self.evaluate_model(model_name, validation_yaml)
                if model_name in model_configs:
                    self.results[model_name].model_size_mb = model_configs[model_name]['size_mb']
            except Exception as e:
                print(f" Failed to evaluate {model_name}: {e}")

        if self.results:
            print(f"\n A/B Test completed successfully!")
            print(f"{len(self.results)} models evaluated")
        else:
            print(f"\n A/B Test failed - no results generated")

    def statistical_analysis(self) -> Dict:
        if len(self.results) < 2:
            return {"error": "Need at least 2 models for statistical comparison"}

        analysis = {}
        model_names = list(self.results.keys())

        metrics_to_compare = ['map50', 'map50_95', 'precision', 'recall', 'f1_score', 'fps']

        for metric in metrics_to_compare:
            values_a = getattr(self.results[model_names[0]], metric, 0)
            values_b = getattr(self.results[model_names[1]], metric, 0)

            if values_a != 0 and values_b != 0:
                pct_diff = ((values_b - values_a) / values_a) * 100
                analysis[metric] = {
                    f'{model_names[0]}': values_a,
                    f'{model_names[1]}': values_b,
                    'percentage_difference': pct_diff,
                    'winner': model_names[1] if values_b > values_a else model_names[0]
                }

        return analysis

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive A/B test report"""
        if not self.results:
            return "No results available. Run the A/B test first."

        report = []
        report.append("=" * 70)
        report.append("YOLO MODEL A/B TEST REPORT")
        report.append("=" * 70)
        report.append(f" Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append(f"Confidence Threshold: {self.confidence_threshold}")
        report.append(f"Test Data Path: {self.test_data_path}")
        report.append("")

        # Model Overview
        report.append(" MODEL OVERVIEW")
        report.append("-" * 40)
        for name, metrics in self.results.items():
            report.append(f"• {name}")
            report.append(f"  Size: {metrics.model_size_mb:.1f} MB")
            report.append(f"  Parameters: {metrics.total_params:,}")
            report.append(f"  Predictions made: {len(metrics.predictions)}")
            report.append("")

        # Performance Comparison
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)

        # Create comparison table
        df_data = []
        for name, metrics in self.results.items():
            df_data.append({
                'Model': name,
                'mAP@0.5': f"{metrics.map50:.3f}" if metrics.map50 > 0 else "N/A",
                'mAP@0.5:0.95': f"{metrics.map50_95:.3f}" if metrics.map50_95 > 0 else "N/A",
                'Precision': f"{metrics.precision:.3f}" if metrics.precision > 0 else "N/A",
                'Recall': f"{metrics.recall:.3f}" if metrics.recall > 0 else "N/A",
                'F1-Score': f"{metrics.f1_score:.3f}" if metrics.f1_score > 0 else "N/A",
                'FPS': f"{metrics.fps:.1f}" if metrics.fps > 0 else "N/A",
                'Inference (ms)': f"{metrics.inference_time * 1000:.1f}" if metrics.inference_time > 0 else "N/A",
                'Size (MB)': f"{metrics.model_size_mb:.1f}"
            })

        df = pd.DataFrame(df_data)
        report.append(df.to_string(index=False))
        report.append("")

        report.append("DETECTION STATISTICS")
        report.append("-" * 40)
        for name, metrics in self.results.items():
            if metrics.predictions:
                total_detections = sum(p['detections'] for p in metrics.predictions)
                avg_detections = total_detections / len(metrics.predictions)
                avg_confidence = np.mean([p['avg_confidence'] for p in metrics.predictions if p['avg_confidence'] > 0])

                report.append(f"{name}:")
                report.append(f"  Total detections: {total_detections}")
                report.append(f"  Avg detections per image: {avg_detections:.1f}")
                report.append(f"  Avg confidence: {avg_confidence:.3f}")
                report.append("")

        if len(self.results) >= 2:
            report.append("📊 STATISTICAL ANALYSIS")
            report.append("-" * 40)
            analysis = self.statistical_analysis()

            for metric, data in analysis.items():
                if 'error' not in data:
                    report.append(f"{metric.upper().replace('_', ' ')}:")
                    model_keys = [k for k in data.keys() if k not in ['percentage_difference', 'winner']]
                    for key in model_keys:
                        report.append(f"  {key}: {data[key]:.3f}")
                    report.append(f"  Difference: {data['percentage_difference']:+.1f}%")
                    report.append(f"  Winner: {data['winner']}")
                    report.append("")

        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)

        if self.results:
            models_with_metrics = [(name, metrics) for name, metrics in self.results.items()]

            valid_accuracy = [(name, metrics) for name, metrics in models_with_metrics if metrics.map50 > 0]
            if valid_accuracy:
                best_accuracy = max(valid_accuracy, key=lambda x: x[1].map50)
                report.append(f"highest Accuracy: {best_accuracy[0]} (mAP@0.5: {best_accuracy[1].map50:.3f})")

            valid_speed = [(name, metrics) for name, metrics in models_with_metrics if metrics.fps > 0]
            if valid_speed:
                best_speed = max(valid_speed, key=lambda x: x[1].fps)
                report.append(f"Fastest Inference: {best_speed[0]} ({best_speed[1].fps:.1f} FPS)")

            smallest_model = min(models_with_metrics, key=lambda x: x[1].model_size_mb)
            report.append(f"Smallest Model: {smallest_model[0]} ({smallest_model[1].model_size_mb:.1f} MB)")
            report.append("")

            report.append("Use Case Recommendations:")
            if valid_speed:
                report.append(f"• Real-time applications: {best_speed[0]}")
            report.append(f"• Edge deployment: {smallest_model[0]}")
            if valid_accuracy:
                report.append(f"• Maximum accuracy: {best_accuracy[0]}")

        report.append("")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"📄 Report saved to: {save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save report: {e}")

        return report_text

    def create_visualizations(self, save_dir: Optional[str] = None):
        """Create comparison visualizations with improved error handling"""
        if not self.results:
            print("No results to visualize")
            return

        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO Model A/B Test Results', fontsize=16, fontweight='bold')

        model_names = list(self.results.keys())

        accuracy_metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        accuracy_data = []

        for metric in accuracy_metrics:
            for name, results in self.results.items():
                value = getattr(results, metric, 0)
                if value > 0:
                    accuracy_data.append({
                        'Model': name,
                        'Metric': metric.replace('_', ' ').replace('map', 'mAP').title(),
                        'Value': value
                    })

        if accuracy_data:
            df_acc = pd.DataFrame(accuracy_data)
            sns.barplot(data=df_acc, x='Metric', y='Value', hue='Model', ax=axes[0, 0])
            axes[0, 0].set_title('Accuracy Metrics Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[0, 0].text(0.5, 0.5, 'No accuracy data available',
                            ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Accuracy Metrics Comparison')

        speed_acc_data = []
        for name, results in self.results.items():
            if results.fps > 0 and results.map50 > 0:
                speed_acc_data.append({
                    'Model': name,
                    'FPS': results.fps,
                    'mAP@0.5': results.map50,
                    'Size': results.model_size_mb
                })

        if speed_acc_data:
            df_speed = pd.DataFrame(speed_acc_data)
            scatter = axes[0, 1].scatter(df_speed['FPS'], df_speed['mAP@0.5'],
                                         s=df_speed['Size'] * 20, alpha=0.7,
                                         c=range(len(df_speed)), cmap='viridis')
            axes[0, 1].set_xlabel('FPS (Higher is Better)')
            axes[0, 1].set_ylabel('mAP@0.5 (Higher is Better)')
            axes[0, 1].set_title('Speed vs Accuracy Trade-off\n(Bubble size = Model size)')

            for i, row in df_speed.iterrows():
                axes[0, 1].annotate(row['Model'], (row['FPS'], row['mAP@0.5']),
                                    xytext=(5, 5), textcoords='offset points', fontsize=10)
        else:
            axes[0, 1].text(0.5, 0.5, 'No speed/accuracy data available',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Speed vs Accuracy Trade-off')

        sizes = [results.model_size_mb for results in self.results.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = axes[1, 0].bar(model_names, sizes, color=colors, alpha=0.7)
        axes[1, 0].set_title('Model Size Comparison')
        axes[1, 0].set_ylabel('Size (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{size:.1f}MB', ha='center', va='bottom', fontsize=10)

        fps_values = [results.fps for results in self.results.values()]
        inf_times = [results.inference_time * 1000 for results in self.results.values()]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = axes[1, 1].bar(x - width / 2, fps_values, width, label='FPS', alpha=0.7, color='lightcoral')

        ax2 = axes[1, 1].twinx()
        bars2 = ax2.bar(x + width / 2, inf_times, width, label='Inference Time (ms)', alpha=0.7, color='lightblue')

        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('FPS', color='darkred')
        ax2.set_ylabel('Inference Time (ms)', color='darkblue')
        axes[1, 1].set_title('Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)

        for bar, fps in zip(bars1, fps_values):
            if fps > 0:
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                                f'{fps:.1f}', ha='center', va='bottom', fontsize=9)

        for bar, inf_time in zip(bars2, inf_times):
            if inf_time > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{inf_time:.1f}ms', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'yolo_ab_test_results.png')
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save visualization: {e}")

        try:
            plt.show()
        except:
            print("Could not display plot (possibly running in headless environment)")

    def export_results(self, filepath: str):
        if not self.results:
            print("No results to export")
            return

        exportable_results = {}

        for name, metrics in self.results.items():
            result_dict = asdict(metrics)
            result_dict['prediction_count'] = len(metrics.predictions)
            result_dict['avg_detections_per_image'] = np.mean(
                [p['detections'] for p in metrics.predictions]) if metrics.predictions else 0
            result_dict['avg_confidence'] = np.mean([p['avg_confidence'] for p in metrics.predictions if
                                                     p['avg_confidence'] > 0]) if metrics.predictions else 0
            del result_dict['predictions']

            exportable_results[name] = result_dict

        try:
            with open(filepath, 'w') as f:
                json.dump(exportable_results, f, indent=2)
            print(f"Results exported to: {filepath}")
        except Exception as e:
            print(f"Failed to export results: {e}")


if __name__ == "__main__":

    tester = YOLOABTester(
        dataset_path=r"C:\Users\elbek\mol-bozor-demo-alif\Mol-bozor--person-3",
        confidence_threshold=0.25
    )


    model_configs = {
        "model/1": {
            "path": r"C:\Users\elbek\mol-bozor-demo-alif\models\best.pt"
        },
        "model/2": {
            "path": r"C:\Users\elbek\mol-bozor-demo-alif\models\mol-bozor v2.pt"
        },
        "model/3": {
            "path": r"C:\Users\elbek\mol-bozor-demo-alif\models\mol-bozor-02.09.2025.pt"
        }

    }

    tester.run_ab_test(
        model_configs=model_configs,
        use_validation=True
    )

    if tester.results:
        print("\n" + "=" * 70)
        print("GENERATING RESULTS...")
        print("=" * 70)

        # Generate and save report
        report = tester.generate_report("yolo_ab_test_report.txt")



        from dataclasses import dataclass, field
        from typing import Dict


        @dataclass
        class ModelMetrics:
            model_name: str
            map50: float = 0.0
            map50_95: float = 0.0
            precision: float = 0.0
            recall: float = 0.0
            f1_score: float = 0.0
            inference_time: float = 0.0
            fps: float = 0.0
            model_size_mb: float = 0.0
            total_params: int = 0
            predictions: list = field(default_factory=list)
            per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # <--- Added


        if validation_yaml and os.path.exists(validation_yaml):
            print("\n Running validation...")
            try:
                val_results = model.val(data=validation_yaml, verbose=False, plots=False)

                if hasattr(val_results, 'box'):
                    metrics.map50 = float(getattr(val_results.box, 'map50', 0))
                    metrics.map50_95 = float(getattr(val_results.box, 'map', 0))
                    metrics.precision = float(getattr(val_results.box, 'mp', 0))
                    metrics.recall = float(getattr(val_results.box, 'mr', 0))
                    if metrics.precision + metrics.recall > 0:
                        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (
                                    metrics.precision + metrics.recall)

                if hasattr(val_results, 'names') and hasattr(val_results, 'box') and hasattr(val_results.box, 'maps'):
                    for cls_idx, cls_name in val_results.names.items():
                        metrics.per_class_metrics[cls_name] = {
                            "mAP@0.5": float(val_results.box.maps[cls_idx]) if val_results.box.maps else 0.0,
                            "Precision": float(
                                val_results.box.precision[cls_idx]) if val_results.box.precision is not None else 0.0,
                            "Recall": float(
                                val_results.box.recall[cls_idx]) if val_results.box.recall is not None else 0.0,
                        }

                print(f"Validation completed - mAP@0.5: {metrics.map50:.3f}")
            except Exception as e:
                print(f"Validation failed: {e}")

#report

        report.append("PER-CLASS METRICS")
        report.append("-" * 40)
        for name, metrics in self.results.items():
            report.append(f"{name}:")
            if metrics.per_class_metrics:
                df_perclass = pd.DataFrame.from_dict(metrics.per_class_metrics, orient='index')
                df_perclass.index.name = "Class"
                report.append(df_perclass.to_string())
            else:
                report.append("  No per-class metrics available.")
            report.append("")
