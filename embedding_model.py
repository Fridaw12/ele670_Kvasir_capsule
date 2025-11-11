
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # "0" for one GPU, "0,1" for two GPUs

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, copy
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet50(weights="IMAGENET1K_V1")


base_model.fc = torch.nn.Linear(base_model.fc.in_features, 11)  


state_dict = torch.load(
    "/home/stud/fwag/bhome/ele670_project/results/resnet50_kvasir_finetuned_ver2.pt",
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
base_model.load_state_dict(new_state_dict, strict=True)


# embedding model
embedding_model = copy.deepcopy(base_model)
embedding_model.fc = torch.nn.Identity()      # remove classification head
embedding_model.eval().to(device)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

data_dir = "/home/stud/fwag/bhome/ele670_project/data/split/val"
dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

class_names = dataset.classes
print(f" Loaded {len(dataset)} images from {data_dir}")

from tqdm import tqdm
import numpy as np

all_embeddings, all_labels = [], []

with torch.no_grad():
    for imgs, lbls in tqdm(loader, desc="Extracting embeddings"):
        imgs = imgs.to(device)
        feats = embedding_model(imgs)        
        all_embeddings.append(feats.cpu())
        all_labels.append(lbls)

embeddings = torch.cat(all_embeddings).numpy()
labels = torch.cat(all_labels).numpy()

print("Embeddings shape:", embeddings.shape)  # (N, 2048)

import os, numpy as np
out_dir = "/home/stud/fwag/bhome/ele670_project/results"
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(out_dir, "labels.npy"), labels)
print(" Saved embeddings and labels to:", out_dir)
