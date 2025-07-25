from ultralytics import YOLO

model = YOLO("best_tuned_yolo.pt")
results = model.val(
    data="../YOLOv8n_training_dataset/data.yaml",
    split="test",  
    imgsz=640,
    conf=0.1,      # Confidence threshold
    iou=0.6,       # IoU threshold
    plots=True,     # Generate plots
    save_json=True  # Save metrics to JSON
)

