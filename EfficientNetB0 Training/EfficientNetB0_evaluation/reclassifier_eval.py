import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
test_data_dir = "../EfficientNetB0_training_dataset/reclassifier_split/test"
model_path = "efficientnetb0_reclassifier.keras"
output_dir = "effnet_test_results"
os.makedirs(output_dir, exist_ok=True)

# Parameters
img_size = (224, 224)
batch_size = 32
class_names = ['food_waste', 'not_food_waste']

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# oad model
model = load_model(model_path, compile=False)

# Predict
pred_probs = model.predict(test_generator, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

# True labels
true_labels = test_generator.classes

# Classification report
report = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
print("\nClassification Report:")
print(report)

# Save report to .txt
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report (EfficientNetB0):\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Split')
plt.tight_layout()

# Save confusion matrix as image
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"\nResults saved to folder: {output_dir}")
