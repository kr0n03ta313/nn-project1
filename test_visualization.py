"""Generate Part A/B visualizations: confusion matrices, MLP weights, CNN kernels."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------- Load test data --------
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
test_imgs = test_imgs.astype(np.float64) / 255.0

# ============================================================
# 1. Confusion Matrix — MLP
# ============================================================
print("Generating confusion matrices...")
mlp = nn.models.Model_MLP()
mlp.load_model(r'.\best_models_mlp\best_model.pickle')
mlp_preds = np.argmax(mlp(test_imgs), axis=1)

cnn = nn.models.Model_CNN()
cnn.load_model(r'.\best_models_cnn\best_model.pickle')
cnn_preds = np.argmax(cnn(test_imgs), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, preds, title, acc in [
    (axes[0], mlp_preds, 'MLP', (mlp_preds == test_labs).mean()),
    (axes[1], cnn_preds, 'CNN', (cnn_preds == test_labs).mean())
]:
    cm = np.zeros((10, 10), dtype=int)
    for t in range(10):
        mask = test_labs == t
        for p in range(10):
            cm[t, p] = ((preds == p) & mask).sum()

    # Log-scale colors: log(1+value) so small values get distinct tints
    cm_log = np.log1p(cm)
    norm = LogNorm(vmin=1, vmax=cm.max())
    im = ax.matshow(cm, cmap='YlOrRd', norm=norm)

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            # White text on dark cells (high values), black on light cells
            text_color = 'white' if val > cm.max() * 0.3 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=7 if val < 100 else 10,
                    fontweight='bold' if i == j else 'normal',
                    color=text_color)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{title} Confusion Matrix (acc={acc:.4f})', fontsize=14)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Count (log scale)', fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.close()
print("  Saved confusion_matrices.png")

# ============================================================
# 2. MLP Weight Visualization (first layer W: 784 x 600)
# ============================================================
print("Generating MLP weight visualization...")
W1 = mlp.layers[0].params['W']  # [784, 600] — each column is a neuron's weights

# Show 10x10 = 100 neurons' weights reshaped to 28x28
fig, axes = plt.subplots(10, 10, figsize=(14, 14))
fig.suptitle('MLP First Layer Weights (100 of 600 hidden neurons)', fontsize=16)

# Pick 100 diverse neurons (sort by weight variance to show most interesting ones)
variances = W1.var(axis=0)
top_idx = np.argsort(variances)[-50:]  # top 50 by variance
# Random 50 others
np.random.seed(42)
rand_idx = np.random.choice(600, 50, replace=False)
show_idx = np.concatenate([top_idx, rand_idx])

for i, ax in enumerate(axes.flat):
    if i < 100:
        w = W1[:, show_idx[i]].reshape(28, 28)
        ax.matshow(w, cmap='RdBu_r')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('mlp_weights.png', dpi=150)
plt.close()
print("  Saved mlp_weights.png")

# ============================================================
# 3. CNN Convolution Kernel Visualization
# ============================================================
print("Generating CNN kernel visualization...")

# conv1: [8, 1, 3, 3] — 8 filters, each 1-channel 3x3
conv1_W = cnn.conv1.W
fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
fig.suptitle('CNN Conv1 Kernels (8 filters, 3x3, input=1 channel)', fontsize=14)
for i in range(8):
    kernel = conv1_W[i, 0]  # [3, 3]
    axes[i].matshow(kernel, cmap='RdBu_r')
    axes[i].set_title(f'F{i}', fontsize=10)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.savefig('cnn_conv1_kernels.png', dpi=150)
plt.close()
print("  Saved cnn_conv1_kernels.png")

# conv2: [16, 8, 3, 3] — 16 filters, each 8-channel 3x3
# Visualize as 16x8 grid
conv2_W = cnn.conv2.W
fig, axes = plt.subplots(16, 8, figsize=(12, 24))
fig.suptitle('CNN Conv2 Kernels (16 filters, 3x3, input=8 channels)', fontsize=14)
for out_c in range(16):
    for in_c in range(8):
        kernel = conv2_W[out_c, in_c]
        axes[out_c, in_c].matshow(kernel, cmap='RdBu_r')
        axes[out_c, in_c].set_xticks([])
        axes[out_c, in_c].set_yticks([])
        if in_c == 0:
            axes[out_c, in_c].set_ylabel(f'F{out_c}', fontsize=9)
        if out_c == 0:
            axes[out_c, in_c].set_title(f'C{in_c}', fontsize=8)
plt.tight_layout()
plt.savefig('cnn_conv2_kernels.png', dpi=150)
plt.close()
print("  Saved cnn_conv2_kernels.png")

# Zoomed: show each conv2 filter's 8-channel kernel as combined heatmap
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle('CNN Conv2 Filters (16 filters, each 3x3x8 → reshaped as 6x12)', fontsize=14)
for i, ax in enumerate(axes.flat):
    # Reshape [8, 3, 3] → [6, 12] by stacking channels
    kernel_all = conv2_W[i]  # [8, 3, 3]
    # Reshape to 4x2 grid of 3x3 kernels = 12x6
    display = np.zeros((12, 6))
    for c in range(8):
        row = c // 2
        col = c % 2
        display[row*3:(row+1)*3, col*3:(col+1)*3] = kernel_all[c]
    ax.matshow(display, cmap='RdBu_r')
    ax.set_title(f'Filter {i}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig('cnn_conv2_filters.png', dpi=150)
plt.close()
print("  Saved cnn_conv2_filters.png")

print("\nAll visualizations saved!")
