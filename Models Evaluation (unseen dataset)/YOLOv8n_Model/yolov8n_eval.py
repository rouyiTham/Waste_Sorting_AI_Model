import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# PATHS  
img_dir = "../Unseen_Validation_Dataset/test/images"
label_dir = "../Unseen_Validation_Dataset/test/labels"
output_dir = "yolov8n_eval_output"
os.makedirs(output_dir, exist_ok=True)

CLASS_NAMES = ["food_waste", "metal", "paper", "plastic", "background"]
filtered_labels = [0, 1, 2, 3, 4]  

# Exclude 'background' (index 4) from classification report
report_excluded_label = CLASS_NAMES.index("background")
report_labels = [i for i in range(len(CLASS_NAMES)) if i != report_excluded_label]
report_class_names = [name for i, name in enumerate(CLASS_NAMES) if i != report_excluded_label]

model = YOLO("best_tuned_yolo.pt")
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

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

# Process each image
for img_name in os.listdir(img_dir):
    if not img_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    frame = cv2.imread(img_path)
    h_img, w_img, _ = frame.shape

    # Load ground truths
    gt_boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id, x_center, y_center, width, height = map(float, parts[:5])
                x1 = int((x_center - width / 2) * w_img)
                y1 = int((y_center - height / 2) * h_img)
                x2 = int((x_center + width / 2) * w_img)
                y2 = int((y_center + height / 2) * h_img)
                gt_boxes.append((int(cls_id), (x1, y1, x2, y2)))

    # Run YOLO inference
    results = model(frame)[0]
    pred_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pred_boxes.append((cls_id, (x1, y1, x2, y2)))

    matched_gt = set()
    matched_pred = set()

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
            y_true.append(CLASS_NAMES.index("background"))
            y_pred.append(pred_cls)

    for gt_idx, (gt_cls, _) in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            y_true.append(gt_cls)
            y_pred.append(CLASS_NAMES.index("background"))

    # Draw predicted bounding boxes
    for pred_cls, (x1, y1, x2, y2) in pred_boxes:
        label_name = CLASS_NAMES[pred_cls]
        color_map = {
            "food_waste": (255, 51, 153),
            "metal": (255, 51, 51),
            "paper": (51, 255, 153),
            "plastic": (51, 255, 255),
            "background": (200, 200, 200)
        }
        color = color_map.get(label_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save image with boxes
    output_img_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_img_path, frame)


### EVALUATION ###

# Classification Report
print("\nClassification Report (YOLO-only):")
report = classification_report(
    y_true,
    y_pred,
    labels=report_labels,
    target_names=report_class_names,
    zero_division=0
)
print(report)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report (YOLO-only):\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=filtered_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=report_class_names, yticklabels=report_class_names, cmap="Blues")
plt.title("Confusion Matrix (YOLO-only)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()
