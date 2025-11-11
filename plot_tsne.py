import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

out_dir = "/home/stud/fwag/bhome/ele670_project/results"
embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
labels = np.load(os.path.join(out_dir, "labels.npy"))

print("Loaded data")
print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)


class_names = None  # Set to None if not available

# Run t-SNE
print(" Running t-SNE")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=2000,
    init="pca",
    random_state=42
)
X_2d = tsne.fit_transform(embeddings)

unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
indexed_labels = np.array([label_to_index[lbl] for lbl in labels])

num_classes = len(unique_labels)
cmap = cm.get_cmap('tab20', num_classes)

markers = ['o', 's', 'v', '^', '<', '>', 'D', 'P', '*', 'X', 'H', '+']  

plt.figure(figsize=(10, 8))

for label, idx in label_to_index.items():
    mask = labels == label
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        c=[cmap(idx)],
        label=class_names[idx] if class_names else str(label),
        marker=markers[idx % len(markers)],
        s=30,
        alpha=0.8,
        edgecolors='k',
        linewidth=0.3
    )

plt.title("t-SNE ResNet-50 Kvasir Embeddings")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
