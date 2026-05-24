import mynn as nn
from draw_tools.plot import plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from struct import unpack
import gzip
import pickle
import os

# Try importing wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, using console logging only")

# fixed seed for experiment
np.random.seed(309)

# -------- Data Loading --------
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# Use saved split if exists, else create new split
idx_path = 'idx.pickle'
if os.path.exists(idx_path):
    with open(idx_path, 'rb') as f:
        idx = pickle.load(f)
else:
    idx = np.random.permutation(np.arange(num))
    with open(idx_path, 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs.astype(np.float64) / 255.0
valid_imgs = valid_imgs.astype(np.float64) / 255.0

# -------- Common Hyperparameters --------
batch_size = 64
num_epochs = 5
init_lr = 0.06
milestones = [800, 2400, 4000]
lr_gamma = 0.5

# ================================================
# Train MLP Baseline
# ================================================
print("=" * 50)
print("Training MLP Baseline")
print("=" * 50)

if WANDB_AVAILABLE:
    wandb.init(project="nn-project1", name="MLP-Baseline", reinit=True,
               config={"model": "MLP", "hidden": 600, "lr": init_lr,
                       "batch_size": batch_size, "epochs": num_epochs})

mlp_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
mlp_optimizer = nn.optimizer.SGD(init_lr=init_lr, model=mlp_model)
mlp_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=mlp_optimizer, milestones=milestones, gamma=lr_gamma)
mlp_loss_fn = nn.op.MultiCrossEntropyLoss(model=mlp_model, max_classes=10)

mlp_runner = nn.runner.RunnerM(mlp_model, mlp_optimizer, nn.metric.accuracy, mlp_loss_fn,
                                batch_size=batch_size, scheduler=mlp_scheduler,
                                use_wandb=WANDB_AVAILABLE)

mlp_runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                 num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_mlp')

if WANDB_AVAILABLE:
    wandb.finish()

# ================================================
# Train CNN
# ================================================
print("\n" + "=" * 50)
print("Training CNN")
print("=" * 50)

if WANDB_AVAILABLE:
    wandb.init(project="nn-project1", name="CNN", reinit=True,
               config={"model": "CNN", "lr": init_lr,
                       "batch_size": batch_size, "epochs": num_epochs})

cnn_model = nn.models.Model_CNN()
cnn_optimizer = nn.optimizer.SGD(init_lr=init_lr, model=cnn_model)
cnn_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=cnn_optimizer, milestones=milestones, gamma=lr_gamma)
cnn_loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

cnn_runner = nn.runner.RunnerM(cnn_model, cnn_optimizer, nn.metric.accuracy, cnn_loss_fn,
                                batch_size=batch_size, scheduler=cnn_scheduler,
                                use_wandb=WANDB_AVAILABLE)

cnn_runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                 num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_cnn')

if WANDB_AVAILABLE:
    wandb.finish()

# ================================================
# Plot Learning Curves
# ================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot(mlp_runner, [axes[0][0], axes[0][1]])
axes[0][0].set_title("MLP Loss")
axes[0][1].set_title("MLP Accuracy")

plot(cnn_runner, [axes[1][0], axes[1][1]])
axes[1][0].set_title("CNN Loss")
axes[1][1].set_title("CNN Accuracy")

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
plt.close()
print("Learning curves saved to learning_curves.png")

# ================================================
# Summary
# ================================================
print("\n" + "=" * 50)
print("Training Summary")
print("=" * 50)
print(f"MLP - Best Dev Accuracy: {mlp_runner.best_score:.4f}")
print(f"CNN - Best Dev Accuracy: {cnn_runner.best_score:.4f}")
