
# Final Evaluation 

import os
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm.notebook import tqdm

IMG_ROOT = "/home/stud/fwag/bhome/ele670_project/data/split_random_filtered"
RESULTS_DIR = "/home/stud/fwag/bhome/ele670_project/results"
SAVE_PATH = os.path.join(RESULTS_DIR, "resnet50_kvasir_finetuned_ver2.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, GPUs visible: {torch.cuda.device_count()}")

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dir = os.path.join(IMG_ROOT, "test")
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

num_classes = len(test_data.classes)
print(f"Testing on {len(test_data)} images across {num_classes} classes.")


model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Handle DataParallel prefix
state_dict = torch.load(SAVE_PATH, map_location=device)
clean_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
model.load_state_dict(clean_state_dict)
model = model.to(device)
model.eval()


all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.title("Confusion Matrix  Test Set (Filtered Split)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


print("\nClassification Report:\n")
report = classification_report(all_labels, all_preds,
                               target_names=test_data.classes,
                               digits=3, output_dict=True)

print(classification_report(all_labels, all_preds,
                            target_names=test_data.classes, digits=3))


class_acc = cm.diagonal() / cm.sum(axis=1)
sorted_idx = np.argsort(class_acc)

plt.figure(figsize=(10,6))
sns.barplot(x=class_acc[sorted_idx], y=np.array(test_data.classes)[sorted_idx], palette="crest")
plt.title("Per-Class Accuracy – Test Set")
plt.xlabel("Accuracy")
plt.ylabel("Class")
plt.xlim(0, 1)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

overall_acc = np.trace(cm) / np.sum(cm)
macro_f1 = np.mean([report[c]["f1-score"] for c in report if c in test_data.classes])
print(f"Overall Test Accuracy: {overall_acc*100:.2f}%")
print(f"Macro-average F1-score: {macro_f1:.3f}")
