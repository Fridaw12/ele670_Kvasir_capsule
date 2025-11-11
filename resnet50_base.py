

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


IMG_ROOT = "/home/stud/fwag/bhome/ele670_project/data/split_random_filtered"
RESULTS_DIR = "/home/stud/fwag/bhome/ele670_project/results"
SAVE_PATH = os.path.join(RESULTS_DIR, "resnet50_kvasir_finetuned_ver2.pt")

BATCH_SIZE = 64
NUM_EPOCHS = 25
BASE_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8

os.makedirs(RESULTS_DIR, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}, GPUs visible: {torch.cuda.device_count()}")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = os.path.join(IMG_ROOT, "train")
val_dir   = os.path.join(IMG_ROOT, "val")

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

num_classes = len(train_data.classes)
print(f"Classes ({num_classes}): {train_data.classes}")


model = models.resnet50(weights="IMAGENET1K_V1")


for name, param in model.named_parameters():
    if not name.startswith("layer4") and not name.startswith("fc"):
        param.requires_grad = False

# Replace classifier 
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)


# compute gradients 
criterion = nn.CrossEntropyLoss()
# how model is updated based on gradients
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=BASE_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)
# adjust learning speed 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


best_acc = 0.0
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []

for epoch in range(NUM_EPOCHS):
    print(f"\n Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0


    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc  = 100 * correct / total

   
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc  = 100 * val_correct / val_total

    scheduler.step()
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f" TrainLoss {train_loss:.4f} TrainAcc {train_acc:.2f}% "
          f" ValLoss {val_loss:.4f} ValAcc {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved best model {SAVE_PATH} (ValAcc: {best_acc:.2f}%)")

print("\nTraining finished.")
print(f"Best validation accuracy: {best_acc:.2f}%")


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
