Animal Counting and Classification System
Project Overview
This repository showcases a computer vision system for detecting, tracking, and classifying animals (e.g., cows, sheep, goats, horses) in video footage, designed as a demo for a portfolio to reflect enterprise-level work. The system, named "Mol-Bozor" (Uzbek for "cattle market"), processes videos to count animals crossing a virtual line, classify them as adult or young based on size statistics, and evaluate model performance. It is a simplified version of larger-scale projects, focusing on core functionality without cloud integrations or real-time streaming.
The project leverages YOLOv11 for object detection, combined with custom logic for tracking, statistical analysis, and dataset preparation. It was developed to demonstrate expertise in computer vision, object tracking, and model evaluation for agricultural applications, such as livestock monitoring.
Technologies Used

Python 3.8+: Core programming language for flexibility and extensive library support.
Ultralytics YOLOv11: State-of-the-art object detection framework for accurate and efficient animal detection.
OpenCV (cv2): For video processing, image manipulation, and visualization of detection results.
Supervision (ByteTrack): Multi-object tracking to maintain consistent animal identities across frames.
NumPy, Pandas, Matplotlib, Seaborn, SciPy: For statistical analysis, data processing, and visualization of model performance and classification results.
PyTorch: Backend for YOLO models, enabling GPU-accelerated inference and training.
Roboflow: For dataset management and download, streamlining labeled data acquisition.
Pathlib: For cross-platform file path handling.

Why These Technologies?

YOLOv11: Chosen for its balance of speed and accuracy, suitable for real-time applications. The nano version (yolo11n.pt) ensures efficiency on resource-constrained devices, while custom fine-tuning adapts it to specific animal classes.
ByteTrack: Selected for robust multi-object tracking, critical for tracking animals across frames and detecting line crossings without losing identity.
OpenCV: Used for its comprehensive image and video processing capabilities, enabling line drawing, bounding box visualization, and screenshot capture.
Statistical Libraries (NumPy, Pandas, etc.): Essential for size-based classification (using percentiles like Q1 for young/adult thresholds) and A/B testing metrics (mAP, FPS, precision).
Roboflow: Simplifies dataset acquisition and versioning, crucial for managing large-scale labeled datasets (4000+ images in the enterprise version).
PyTorch: Provides a flexible deep learning framework, supporting YOLO's training and inference needs, with CUDA for GPU acceleration.

How It Works
The system comprises four main scripts, each addressing a specific aspect of the pipeline:

Dataset Download (downloadDataset.py):

Pulls labeled datasets from Roboflow (e.g., "mol-bozor-person" project, version 3).
Why: Ensures access to structured, annotated data for training and validation, mimicking enterprise workflows where datasets are centrally managed.
How: Uses Roboflow's API to download YOLO-compatible datasets with images and annotations.


Model Training (train.py):

Fine-tunes a YOLOv11 model (yolo11n.pt) on a custom dataset.
Why: Tailors the model to detect specific animal classes (cow, sheep, goat, horse) with high accuracy.
How: Configures hyperparameters (e.g., epochs=50, batch=16, image size=640) and trains using the dataset's data.yaml. Outputs trained weights to ./runs/detect/train/weights/best.pt.


Inference and Counting (inference.py):

Processes videos to detect animals, track them, and count crossings over a virtual line.
Why: Core functionality for livestock monitoring, enabling automated counting and age classification.
How: Uses YOLOv11 for detection, ByteTrack for tracking, and custom geometry to detect line crossings. Sizes are collected near a detection zone (300px from the line) and classified as adult or young using the 25th percentile (Q1) as a threshold. Screenshots are saved for each crossing, organized by class and category.


Screenshot Capture (screenshotTaker.py):

Extracts frames with detections from videos to expand training datasets.
Why: Supports iterative dataset improvement, a common enterprise need for model retraining.
How: Detects animals, filters by confidence (e.g., 0.6), and saves images every 30 frames per class to avoid redundancy, organized by animal type.


A/B Testing (a-bTesting.py):

Compares multiple YOLO models on metrics like mAP@0.5, FPS, and model size.
Why: Ensures optimal model selection for deployment, balancing accuracy and performance.
How: Evaluates models on a test dataset, calculates metrics, and generates visualizations (e.g., bar charts, scatter plots). Supports per-class metrics for detailed analysis.



Project Design

Modularity: Each script handles a distinct task (data prep, training, inference, evaluation), mirroring microservices in enterprise systems.
Scalability: Designed to handle large datasets (4000+ images in the full version) and multiple models, though the demo uses a smaller subset.
Extensibility: Easily adaptable to new animal classes or detection tasks by updating the dataset and retraining.
Robustness: Includes error handling for file paths, model loading, and video processing, ensuring reliability.

Limitations and Notes

Large datasets and trained models/videos are not committed due to GitHub size limits. They can be recreated using downloadDataset.py and train.py.
Paths in scripts are hardcoded (e.g., C:\Users\elbek\...) for demo purposes; production versions use environment variables or config files.
The demo focuses on offline video processing; enterprise versions included real-time streaming and cloud integration.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Built with Ultralytics YOLO and Roboflow.
Inspired by enterprise-level livestock monitoring systems, adapted for portfolio demonstration.
