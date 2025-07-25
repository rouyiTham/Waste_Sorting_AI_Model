from ultralytics import YOLO
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import Counter
import os
from datetime import datetime

# Load models
yolo_model = YOLO("yolo_best.pt")
reclassifier = load_model("efficientnetb0_reclassifier.keras", compile=False)

# Open camera: 0 = integrated cam, 1 = external webcam
cap = cv2.VideoCapture(0)

# Thresholds
CONF_THRESHOLD = 0.5
RECLASSIFY_THRESHOLD = 0.6

# Tracking variables
frame_count = 0
class_counter = Counter()
reclassify_count = 0

# Window setup
WINDOW_NAME = "Waste Detection and Classification"

# Define colors for each class
COLOR_MAP = {
    "food_waste": (255, 51, 153),   # Purple
    "metal": (255, 51, 51),         # Blue
    "paper": (51, 255, 153),        # Green
    "plastic": (51, 255, 255),      # Yellow
}
RECLASSIFIED_COLOR = (51, 51, 255)   # Red for all reclassified detections


# Initialize video writer
save_output = True
out = None

# Prepare output directory and filename
if save_output:
    os.makedirs("output_runs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output_runs", f"detection_{timestamp}.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Initialize video writer once obtained the frame shape
    if save_output and out is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    results = yolo_model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = yolo_model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        reclassified = False

        # Reclassify if YOLO detects "food_waste" with low confidence
        if label == "food_waste" and conf < CONF_THRESHOLD:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                img = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
                input_array = np.expand_dims(img, axis=0)
                pred = reclassifier.predict(input_array, verbose=0)[0][0]
                new_label = "food_waste" if pred > RECLASSIFY_THRESHOLD else "not_food_waste"
                reclassify_count += 1
                label = new_label
                reclassified = True
                confidence_display = pred if new_label == "food_waste" else 1 - pred

        # Get bounding box color
        color = RECLASSIFIED_COLOR if reclassified else COLOR_MAP.get(label, (255, 255, 255))

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        conf_text = f"{confidence_display:.2f}" if reclassified else f"{conf:.2f}"
        cv2.putText(frame, f"{label} ({conf_text})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        class_counter[label] += 1

    # Show and optionally save frame
    cv2.imshow(WINDOW_NAME, frame)
    if save_output:
        out.write(frame)

    # Exit if 'q' is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Detection Summary
print("\n--------------- Detection Summary ---------------")

print(f"Total frames processed: {frame_count}")

print(f"Total detections: {sum(class_counter.values())}")

ALL_CLASSES = ["food_waste", "metal", "paper", "plastic", "not_food_waste"]
print("Detections by class:")
for cls in ALL_CLASSES:
    print(f"    {cls}: {class_counter.get(cls, 0)}")

print(f"Items reclassified by EfficientNetB0: {reclassify_count}")

print("-------------------------------------------------")
