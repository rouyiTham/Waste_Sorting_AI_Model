import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("best_8.pt")

# Input video
video_path = "C:/Users/User/Desktop/test_vid13.webm" # videos to extract cropped detections
cap = cv2.VideoCapture(video_path)

# Output dirs
output_fw = "reclassifier_raw/food_waste"
output_nfw = "reclassifier_raw/not_food_waste"
os.makedirs(output_fw, exist_ok=True)
os.makedirs(output_nfw, exist_ok=True)

# Class info
fw_class = "food_waste"
fw_id = [k for k, v in model.names.items() if v == fw_class][0]

frame_idx = 0
count_fw, count_nfw = 0, 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.3)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            if class_id == fw_id:
                path = f"{output_fw}/fw_frame{frame_idx}_{x1}_{y1}.jpg"
                count_fw += 1
            else:
                path = f"{output_nfw}/nfw_frame{frame_idx}_{x1}_{y1}.jpg"
                count_nfw += 1

            cv2.imwrite(path, crop)

    frame_idx += 1

cap.release()
print(f"Saved {count_fw} food_waste and {count_nfw} not_food_waste crops from video.")
