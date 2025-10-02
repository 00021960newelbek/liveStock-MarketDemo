import cv2
import os
from ultralytics import YOLO
import time
from pathlib import Path


class AnimalDetectionCapture:
    def __init__(self, model_path, output_dir="captured_animals", confidence_threshold=0.5, capture_interval=30):
        """
        Initialize the animal detection and capture system

        Args:
            model_path: Path to your trained YOLOv11 model
            output_dir: Directory to save captured screenshots
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            capture_interval: Minimum frames between captures of same animal type
        """
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.capture_interval = capture_interval

        self.animal_classes = ['cow', 'sheep', 'goat', 'horse']

        self.setup_directories()

        self.last_capture = {animal: 0 for animal in self.animal_classes}

        self.capture_count = {animal: 0 for animal in self.animal_classes}

    def setup_directories(self):
        self.output_dir.mkdir(exist_ok=True)
        for animal in self.animal_classes:
            (self.output_dir / animal).mkdir(exist_ok=True)

    def should_capture(self, animal_type, frame_count):
        return frame_count - self.last_capture[animal_type] >= self.capture_interval

    def save_detection(self, frame, detections, frame_count, video_name):
        captured_animals = []

        for detection in detections:
            box = detection.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = detection.boxes.conf[0].cpu().numpy()
            class_id = int(detection.boxes.cls[0].cpu().numpy())

            if confidence < self.confidence_threshold:
                continue

            animal_name = self.model.names[class_id]

            if animal_name not in self.animal_classes:
                continue

            if not self.should_capture(animal_name, frame_count):
                continue

            timestamp = int(time.time())
            self.capture_count[animal_name] += 1
            filename = f"{video_name}_{animal_name}_{self.capture_count[animal_name]}_{timestamp}.jpg"
            filepath = self.output_dir / animal_name / filename

            cv2.imwrite(str(filepath), frame)

            self.last_capture[animal_name] = frame_count
            captured_animals.append((animal_name, confidence, filepath))

            print(f"Captured {animal_name} (conf: {confidence:.2f}) -> {filename}")

        return captured_animals

    def process_video(self, video_path, save_annotations=True):
        video_path = Path(video_path)
        video_name = video_path.stem

        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        frame_count = 0
        total_captures = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 5 == 0:
                results = self.model(frame, verbose=False)


                if len(results[0].boxes) > 0:
                    captured = self.save_detection(frame, results, frame_count, video_name)
                    total_captures += len(captured)

                    if save_annotations and captured:
                        self.save_yolo_annotations(frame, results[0], video_name, frame_count)

        cap.release()
        print(f"Video processing complete. Total captures: {total_captures}")

    def save_yolo_annotations(self, frame, result, video_name, frame_count):
        h, w = frame.shape[:2]
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        ann_filename = f"{video_name}_{frame_count}_{timestamp}.txt"
        ann_filepath = annotations_dir / ann_filename

        with open(ann_filepath, 'w') as f:
            for detection in result.boxes:
                if detection.conf[0] >= self.confidence_threshold:
                    class_id = int(detection.cls[0])
                    x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()

                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def process_multiple_videos(self, video_directory, video_extensions=None):
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        video_dir = Path(video_directory)
        video_files = []

        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))

        print(f"Found {len(video_files)} video files")

        for video_file in video_files:
            try:
                self.process_video(video_file)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    def get_capture_statistics(self):
        stats = {}
        for animal in self.animal_classes:
            animal_dir = self.output_dir / animal
            if animal_dir.exists():
                image_count = len(list(animal_dir.glob("*.jpg")))
                stats[animal] = image_count
            else:
                stats[animal] = 0

        return stats


def main():
    # Configuration
    MODEL_PATH = r"C:\Users\elbek\mol-bozor-demo-alif\models\mol-bozor-02.09.2025.pt"
    VIDEO_DIRECTORY = r"C:\Users\elbek\mol-bozor-demo-alif\models\videos\2025-09-07T05-27-07_to_2025-09-07T05-37-07.mp4"
    OUTPUT_DIRECTORY = "captured_training_data"
    CONFIDENCE_THRESHOLD = 0.6
    CAPTURE_INTERVAL = 30


    capture_system = AnimalDetectionCapture(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        capture_interval=CAPTURE_INTERVAL
    )

    if os.path.isfile(VIDEO_DIRECTORY):
        capture_system.process_video(VIDEO_DIRECTORY)
    else:
        capture_system.process_multiple_videos(VIDEO_DIRECTORY)

    stats = capture_system.get_capture_statistics()
    print("\n=== Capture Statistics ===")
    for animal, count in stats.items():
        print(f"{animal}: {count} images captured")

    print(f"\nImages saved to: {OUTPUT_DIRECTORY}")
    print("You can now use these images to retrain your model!")


if __name__ == "__main__":
    main()