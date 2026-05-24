"""Compare Round 1 (uniform) vs Round 2 (Hard Negative Mining) OvA CNNs,
and test best calibration on Round 2 models."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle

np.random.seed(309)

# -------- Load test/valid data --------
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
test_imgs = test_imgs.astype(np.float64) / 255.0

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
with gzip.open(train_images_path, 'rb') as f:
    magic, tnum, rows, cols = unpack('>4I', f.read(16))
    all_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(tnum, 28*28)
with gzip.open(train_labels_path, 'rb') as f:
    magic, tnum = unpack('>2I', f.read(8))
    all_labs = np.frombuffer(f.read(), dtype=np.uint8)
with open('idx_ova.pickle', 'rb') as f:
    idx = pickle.load(f)
all_imgs = all_imgs[idx]
all_labs = all_labs[idx]
valid_imgs = all_imgs[:10000].astype(np.float64) / 255.0
valid_labs = all_labs[:10000]
train_imgs_cal = all_imgs[10000:].astype(np.float64) / 255.0
train_labs_cal = all_labs[10000:]

eval_bs = 256

def load_models(model_dir):
    models = []
    for d in range(10):
        m = nn.models.Model_BinaryCNN()
        m.load_model(f'{model_dir}/digit_{d}/best_model.pickle')
        models.append(m)
    return models

def collect_logits(models, images):
    n = images.shape[0]
    logits = np.zeros((n, 10))
    for start in range(0, n, eval_bs):
        end = min(start + eval_bs, n)
        batch = images[start:end]
        for d in range(10):
            logits[start:end, d:d+1] = models[d](batch)
    return logits

def pairwise_augment(X):
    parts = [X]
    for i in range(10):
        for j in range(i+1, 10):
            parts.append(X[:, i:i+1] - X[:, j:j+1])
    return np.concatenate(parts, axis=1)

# -------- Load both rounds --------
print("Loading Round 1 models (uniform sampling)...")
models_r1 = load_models('./ova_models')
print("Loading Round 2 models (Hard Negative Mining)...")
models_r2 = load_models('./ova_models_v2')

# -------- Evaluate Round 1 --------
print("\nEvaluating Round 1...")
logits_r1_test = collect_logits(models_r1, test_imgs)
acc_r1 = (np.argmax(logits_r1_test, axis=1) == test_labs).mean()

# Per-digit recall for Round 1
print("Round 1 per-digit test recall:")
for d in range(10):
    mask = test_labs == d
    pred = np.argmax(logits_r1_test[mask], axis=1)
    recall = (pred == d).mean()
    print(f"  digit {d}: {recall:.4f}")

# -------- Evaluate Round 2 --------
print("\nEvaluating Round 2...")
logits_r2_test = collect_logits(models_r2, test_imgs)
acc_r2 = (np.argmax(logits_r2_test, axis=1) == test_labs).mean()

print("Round 2 per-digit test recall:")
for d in range(10):
    mask = test_labs == d
    pred = np.argmax(logits_r2_test[mask], axis=1)
    recall = (pred == d).mean()
    print(f"  digit {d}: {recall:.4f}")

# -------- Best calibration on Round 2 --------
print("\nTraining best calibration (Pairwise Deep MLP) on Round 2 logits...")
logits_r2_valid = collect_logits(models_r2, valid_imgs)
logits_r2_train = collect_logits(models_r2, train_imgs_cal)

# Pairwise Deep MLP
class MLPCalibration(nn.op.Layer):
    def __init__(self, hidden_dims, input_dim):
        super().__init__()
        self.optimizable = False
        self.fc1 = nn.op.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.op.ReLU()
        self.layers = [self.fc1, self.relu1]
        prev_dim = hidden_dims[0]
        self.fcs = []
        self.relus = []
        for hd in hidden_dims[1:]:
            fc = nn.op.Linear(prev_dim, hd)
            relu = nn.op.ReLU()
            self.fcs.append(fc)
            self.relus.append(relu)
            self.layers.extend([fc, relu])
            prev_dim = hd
        self.fc_out = nn.op.Linear(prev_dim, 10)
        self.layers.append(self.fc_out)

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        x = self.fc1(X); x = self.relu1(x)
        for fc, relu in zip(self.fcs, self.relus):
            x = fc(x); x = relu(x)
        return self.fc_out(x)

    def backward(self, g):
        g = self.fc_out.backward(g)
        for fc, relu in zip(reversed(self.fcs), reversed(self.relus)):
            g = relu.backward(g); g = fc.backward(g)
        g = self.relu1.backward(g); g = self.fc1.backward(g)
        return g

class PairwiseCal(nn.op.Layer):
    def __init__(self, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.optimizable = False
        self.mlp = MLPCalibration(hidden_dims, input_dim=55)
        self.layers = self.mlp.layers
    def __call__(self, X): return self.forward(X)
    def forward(self, X): return self.mlp(pairwise_augment(X))
    def backward(self, g): return self.mlp.backward(g)

# Train on 50K training logits (raw, PairwiseCal applies augmentation internally)
cal_X_raw_raw = logits_r2_train  # [50000, 10] raw logits
cal_y = train_labs_cal
cal_model = PairwiseCal([128, 64, 32])

optimizer = nn.optimizer.SGD(init_lr=0.005, model=cal_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500, 1500, 3000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cal_model, max_classes=10)

for epoch in range(30):
    idx = np.random.permutation(cal_X_raw.shape[0])
    X_shuf = cal_X_raw[idx]
    y_shuf = cal_y[idx]
    for b in range(0, X_shuf.shape[0], 256):
        if b >= X_shuf.shape[0]: break
        batch_X = X_shuf[b:b+256]
        batch_y = y_shuf[b:b+256]
        _ = loss_fn(cal_model(batch_X), batch_y)
        loss_fn.backward()
        optimizer.step()
        scheduler.step()

acc_r2_cal = (np.argmax(cal_model(logits_r2_test), axis=1) == test_labs).mean()

# -------- Final Summary --------
print(f"\n{'='*60}")
print(f"HARD NEGATIVE MINING — FINAL RESULTS")
print(f"{'='*60}")
print(f"Original multi-class CNN:                       0.9800")
print(f"")
print(f"Round 1 OvA (uniform sampling, no cal):         {acc_r1:.4f}")
print(f"Round 1 OvA + best calibration:                 0.9586  (from earlier experiment)")
print(f"")
print(f"Round 2 OvA (HNM, no cal):                      {acc_r2:.4f}")
print(f"Round 2 OvA (HNM) + Pairwise Deep MLP:          {acc_r2_cal:.4f}")

# Per-digit comparison
print(f"\nPer-digit recall comparison (Round 1 vs Round 2):")
print(f"{'Digit':<8} {'R1 Recall':<12} {'R2 Recall':<12} {'Change':<10}")
print(f"{'-'*42}")
for d in range(10):
    mask1 = test_labs == d
    r1_recall = (np.argmax(logits_r1_test[mask1], axis=1) == d).mean()
    mask2 = test_labs == d
    r2_recall = (np.argmax(logits_r2_test[mask2], axis=1) == d).mean()
    print(f"{d:<8} {r1_recall:<12.4f} {r2_recall:<12.4f} {r2_recall-r1_recall:+.4f}")
