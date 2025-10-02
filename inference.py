import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
import time
import numpy as np


class AnimalCounter:
    def __init__(self, model_path):

        self.model = YOLO(model_path)

        self.class_names = list(self.model.names.values())

        self.tracker = sv.ByteTrack()

        self.line_points = [(1107, 799), (1185, 1507)]

        self.track_history = defaultdict(list)
        self.tracked_states = {}

        self.crossings_data = []

        self.all_sizes_by_class = {cls: [] for cls in self.class_names}

        self.detection_zone_distance = 300  # Collect sizes only within 300px of line

        self.final_counts = {cls: {"adult": 0, "young": 0} for cls in self.class_names}

        self.screenshot_dir = "screenshots-new"
        os.makedirs(self.screenshot_dir, exist_ok=True)

    # ---------------- LINE GEOMETRY ----------------
    def is_above_line(self, x, y, p1, p2):
        """Check if (x,y) is on one side of line (p1→p2)."""
        (x1, y1), (x2, y2) = p1, p2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0

    def distance_to_line(self, x, y, p1, p2):
        """Calculate perpendicular distance from point (x,y) to line segment p1-p2."""
        x1, y1 = p1
        x2, y2 = p2

        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_len_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

    def has_crossed_line(self, cx, cy, track_id):
        """Detect if object crossed the line."""
        p1, p2 = self.line_points
        current_side = self.is_above_line(cx, cy, p1, p2)

        if track_id not in self.tracked_states:
            self.tracked_states[track_id] = {"last_side": current_side, "counted": False}
            return False

        last_side = self.tracked_states[track_id]["last_side"]
        counted = self.tracked_states[track_id]["counted"]

        if not counted and last_side != current_side:
            self.tracked_states[track_id]["counted"] = True
            return True

        self.tracked_states[track_id]["last_side"] = current_side
        return False

    def save_full_frame_screenshot(self, frame, class_name, category, track_id, timestamp):
        """Save full frame screenshot."""
        folder = os.path.join(self.screenshot_dir, class_name, category)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"fullframe_{class_name}_{category}_{track_id}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        return filename

    def process_frame(self, frame):
        results = self.model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)

        for xyxy, confidence, class_id, track_id in zip(
                tracked_detections.xyxy,
                tracked_detections.confidence,
                tracked_detections.class_id,
                tracked_detections.tracker_id,
        ):
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            class_name = self.class_names[int(class_id)]

            width = x2 - x1
            height = y2 - y1
            size = width * height

            dist_to_line = self.distance_to_line(cx, cy, self.line_points[0], self.line_points[1])

            if dist_to_line <= self.detection_zone_distance:
                self.all_sizes_by_class[class_name].append(size)
                box_color = (0, 255, 255)  # Yellow for objects in detection zone
            else:
                box_color = (0, 255, 0)  # Green for objects outside zone

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{class_name} {track_id} (s:{size})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            if self.has_crossed_line(cx, cy, track_id):
                timestamp = time.strftime("%Y%m%d-%H%M%S") + f"-{track_id}"

                # Store crossing data for later processing
                crossing_data = {
                    'class_name': class_name,
                    'size': size,
                    'track_id': track_id,
                    'frame_copy': frame.copy(),
                    'timestamp': timestamp,
                    'coordinates': (cx, cy),
                    'width': width,
                    'height': height
                }
                self.crossings_data.append(crossing_data)

                print(f"📝 Crossing detected: {class_name} (ID: {track_id}, Size: {size})")

        cv2.line(frame, self.line_points[0], self.line_points[1], (0, 0, 255), 3)

        angle = np.arctan2(self.line_points[1][1] - self.line_points[0][1],
                           self.line_points[1][0] - self.line_points[0][0])
        perpendicular = angle + np.pi / 2

        offset_x = int(self.detection_zone_distance * np.cos(perpendicular))
        offset_y = int(self.detection_zone_distance * np.sin(perpendicular))

        zone_line1_p1 = (self.line_points[0][0] + offset_x, self.line_points[0][1] + offset_y)
        zone_line1_p2 = (self.line_points[1][0] + offset_x, self.line_points[1][1] + offset_y)
        zone_line2_p1 = (self.line_points[0][0] - offset_x, self.line_points[0][1] - offset_y)
        zone_line2_p2 = (self.line_points[1][0] - offset_x, self.line_points[1][1] - offset_y)

        cv2.line(frame, zone_line1_p1, zone_line1_p2, (255, 255, 0), 1)
        cv2.line(frame, zone_line2_p1, zone_line2_p2, (255, 255, 0), 1)

        cv2.putText(frame, f"Crossings detected: {len(self.crossings_data)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Yellow box = in detection zone", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def process_crossings(self):
        """Process all crossings after video is complete."""
        print("\n" + "=" * 60)
        print("🔄 PHASE 2: Processing crossings with calculated statistics...")
        print("=" * 60)

        if not self.crossings_data:
            print("❌ No crossings detected!")
            return

        class_stats = {}
        for class_name, sizes in self.all_sizes_by_class.items():
            if len(sizes) >= 3:  # Need at least 3 samples
                sizes_array = np.array(sizes)
                median_size = np.median(sizes_array)
                mean_size = np.mean(sizes_array)
                q1 = np.percentile(sizes_array, 25)  # 25th percentile
                q3 = np.percentile(sizes_array, 75)  # 75th percentile

                class_stats[class_name] = {
                    'median': median_size,
                    'mean': mean_size,
                    'q1': q1,
                    'q3': q3,
                    'samples': len(sizes)
                }

                print(f"📊 {class_name} statistics (from {len(sizes)} samples near line):")
                print(f"   Median: {median_size:.0f}")
                print(f"   Mean: {mean_size:.0f}")
                print(f"   Q1 (25%): {q1:.0f}")
                print(f"   Q3 (75%): {q3:.0f}")

        print(f"\n🎯 Processing {len(self.crossings_data)} crossings...")
        print(f"📝 Classification: size < Q1 = young, size >= Q1 = adult")
        print()

        for i, crossing in enumerate(self.crossings_data):
            class_name = crossing['class_name']
            size = crossing['size']
            track_id = crossing['track_id']
            frame = crossing['frame_copy']
            timestamp = crossing['timestamp']

            if class_name in class_stats:
                stats = class_stats[class_name]
                threshold = stats['q1']  # Use Q1 (25th percentile) as threshold
                category = "young" if size < threshold else "adult"

                self.final_counts[class_name][category] += 1

                filename = self.save_full_frame_screenshot(frame, class_name, category, track_id, timestamp)

                percentage = (size / stats['median']) * 100
                print(
                    f"✅ {i + 1:2d}. {class_name} ID:{track_id} | Size:{size:5.0f} ({percentage:5.1f}% of median) → {category.upper()}")
                print(f"     Screenshot: {filename}")
            else:
                print(f"⚠️  {class_name} ID:{track_id} - Limited data, using heuristic classification")
                category = "adult"
                self.final_counts[class_name][category] += 1
                filename = self.save_full_frame_screenshot(frame, class_name, category, track_id, timestamp)
                print(f"     Screenshot: {filename}")

    def show_final_results(self):
        """Display final classification results."""
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)

        total_animals = 0
        total_adults = 0
        total_young = 0

        for class_name, counts in self.final_counts.items():
            class_total = counts['adult'] + counts['young']
            if class_total > 0:
                total_animals += class_total
                total_adults += counts['adult']
                total_young += counts['young']

                print(f"🐄 {class_name}:")
                print(f"   Adults: {counts['adult']}")
                print(f"   Young:  {counts['young']}")
                print(f"   Total:  {class_total}")
                print()

        print(f"🎯 GRAND TOTALS:")
        print(f"   Total Animals: {total_animals}")
        print(f"   Total Adults:  {total_adults}")
        print(f"   Total Young:   {total_young}")
        print()
        print(f"📁 Screenshots saved to: {self.screenshot_dir}")
        print("=" * 60)

    # ---------------- RUN VIDEO ----------------
    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video")
            return

        print("PHASE 1: Collecting crossing data...")
        print(f"Video: {video_path}")
        print(f"Detection zone: {self.detection_zone_distance}px around counting line")
        print("   (Only objects in this zone are used for size calibration)")
        print("Press 'q' to quit early\n")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            frame_count += 1

            display_frame = cv2.resize(frame, (1000, 700))
            cv2.imshow("Animal Counter - Data Collection Phase", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nPhase 1 complete! Processed {frame_count} frames")

        self.process_crossings()

        self.show_final_results()


if __name__ == "__main__":
    counter = AnimalCounter(r"C:\Users\elbek\mol-bozor-demo-alif\runs\detect\train5\weights\best.pt")
    counter.run_video(r"C:\Users\elbek\mol-bozor-demo-alif\models\videos\2025-09-07T05-27-07_to_2025-09-07T05-37-07.mp4q")