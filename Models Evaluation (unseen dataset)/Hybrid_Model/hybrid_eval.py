import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
img_dir = "../Unseen_Validation_Dataset/test/images"
label_dir = "../Unseen_Validation_Dataset/test/labels"

# Output directory for boxed images
output_dir = "output_eval"
os.makedirs(output_dir, exist_ok=True)

# Class names
CLASS_NAMES = ["food_waste", "metal", "paper", "plastic", "not_food_waste", "background"]

# Exclude "not_food_waste" and "background" from classification report
EXCLUDED_NAMES_REPORT = ["not_food_waste", "background"]
filtered_labels_report = [i for i, name in enumerate(CLASS_NAMES) if name not in EXCLUDED_NAMES_REPORT]
filtered_class_names_report = [name for i, name in enumerate(CLASS_NAMES) if name not in EXCLUDED_NAMES_REPORT]

# Only exclude "not_food_waste" from confusion matrix
EXCLUDED_NAMES_CM = ["not_food_waste"]
filtered_labels_cm = [i for i, name in enumerate(CLASS_NAMES) if name not in EXCLUDED_NAMES_CM]
filtered_class_names_cm = [name for i, name in enumerate(CLASS_NAMES) if name not in EXCLUDED_NAMES_CM]

# Load models
yolo_model = YOLO("best_tuned_yolo.pt")
reclassifier = load_model("efficientnetb0_reclassifier.keras", compile=False)

CONF_THRESHOLD = 0.5
RECLASSIFY_THRESHOLD = 0.6
IOU_THRESHOLD = 0.4

# For ground truths and predictions
y_true = []
y_pred = []

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

for img_name in os.listdir(img_dir):
    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    frame = cv2.imread(img_path)
    h_img, w_img, _ = frame.shape

    # Load ground truth
    gt_boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id, x_center, y_center, width, height = map(float, parts[:5])
                    x1 = int((x_center - width / 2) * w_img)
                    y1 = int((y_center - height / 2) * h_img)
                    x2 = int((x_center + width / 2) * w_img)
                    y2 = int((y_center + height / 2) * h_img)
                    gt_boxes.append((int(cls_id), (x1, y1, x2, y2)))
                except ValueError:
                    print(f"Skipping invalid line in {label_path}: {line}")
                    continue
            else:
                print(f"Skipping short or malformed line in {label_path}: {line}")
                continue

    # YOLO + hybrid inference
    results = yolo_model(frame)[0]
    pred_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = yolo_model.names[cls_id]

        # Skip any detection below threshold (except food_waste, which is reclassified)
        if conf < CONF_THRESHOLD and label != "food_waste":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Reclassify low-confidence food_waste
        if label == "food_waste" and conf < CONF_THRESHOLD:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                img = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
                input_array = np.expand_dims(img, axis=0)
                pred = reclassifier.predict(input_array, verbose=0)[0][0]
                label = "food_waste" if pred > RECLASSIFY_THRESHOLD else "not_food_waste"
                cls_id = CLASS_NAMES.index(label)

        pred_boxes.append((cls_id, (x1, y1, x2, y2)))

    matched_gt = set()
    matched_pred = set()

    # Match predictions with ground truths
    for pred_idx, (pred_cls, pred_box) in enumerate(pred_boxes):
        best_iou = 0
        matched_cls = None
        matched_gt_idx = None
        for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou_score = iou(pred_box, gt_box)
            if iou_score >= IOU_THRESHOLD and iou_score > best_iou:
                best_iou = iou_score
                matched_cls = gt_cls
                matched_gt_idx = gt_idx
        if matched_cls is not None:
            y_true.append(matched_cls)
            y_pred.append(pred_cls)
            matched_gt.add(matched_gt_idx)
            matched_pred.add(pred_idx)
        else:
            # False positive: prediction didn't match any ground truth
            y_true.append(CLASS_NAMES.index("background"))
            y_pred.append(pred_cls)

    # Handle false negatives: ground truths that weren't matched by any prediction
    for gt_idx, (gt_cls, _) in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            y_true.append(gt_cls)
            y_pred.append(CLASS_NAMES.index("background"))


    # Draw bounding boxes
    for pred_cls, pred_box in pred_boxes:
        x1, y1, x2, y2 = pred_box
        label_name = CLASS_NAMES[pred_cls]

        color_map = {
            "food_waste": (255, 51, 153),
            "metal": (255, 51, 51),
            "paper": (51, 255, 153),
            "plastic": (51, 255, 255),
            "not_food_waste": (51, 51, 255)
        }
        color = color_map.get(label_name, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save image with boxes
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, frame)

###### METRICS ######

# Classification Report
print("\nClassification Report (hybrid):")
report = classification_report(
    y_true,
    y_pred,
    labels=filtered_labels_report,
    target_names=filtered_class_names_report,
    zero_division=0
)
print(report)

# Save report
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report (hybrid):\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=filtered_labels_cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=filtered_class_names_report, yticklabels=filtered_class_names_report, cmap="Blues")
plt.title("Confusion Matrix (hybrid)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save confusion matrix plot
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()
