from ultralytics import YOLO
import cv2
import os
import glob

# Load trained YOLOv8 model
model = YOLO("best_tuned_yolo.pt")

# Paths
test_images_dir = "test/images" # images to extract cropped detections
output_dir_fw = "reclassifier_raw/food_waste"
output_dir_nfw = "reclassifier_raw/not_food_waste"
os.makedirs(output_dir_fw, exist_ok=True)
os.makedirs(output_dir_nfw, exist_ok=True)

# Identify class IDs
class_names = model.names
food_waste_class_name = "food_waste"
food_waste_id = [k for k, v in class_names.items() if v == food_waste_class_name][0]

# Process test images
count_fw, count_nfw = 0, 0

for img_path in glob.glob(f"{test_images_dir}/*.jpg"):
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model.predict(img, conf=0.3)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            if class_id == food_waste_id:
                save_path = f"{output_dir_fw}/fw_{base_name}_{x1}_{y1}.jpg"
                count_fw += 1
            else:
                save_path = f"{output_dir_nfw}/nfw_{base_name}_{x1}_{y1}.jpg"
                count_nfw += 1

            cv2.imwrite(save_path, crop)

print(f"Saved {count_fw} food_waste and {count_nfw} not_food_waste cropped detections.")
