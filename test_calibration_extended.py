"""Extended calibration experiments: use full training set logits + deeper networks."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle

np.random.seed(309)

# -------- Load data --------
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
train_imgs_cal = all_imgs[10000:].astype(np.float64) / 255.0  # 50000 training images
train_labs_cal = all_labs[10000:]

# -------- Load models and collect logits --------
print("Loading 10 binary CNN models...")
models = []
for digit in range(10):
    m = nn.models.Model_BinaryCNN()
    m.load_model(f'./ova_models/digit_{digit}/best_model.pickle')
    models.append(m)

eval_bs = 256

def collect_logits(images):
    n = images.shape[0]
    logits = np.zeros((n, 10))
    for start in range(0, n, eval_bs):
        end = min(start + eval_bs, n)
        batch = images[start:end]
        for d in range(10):
            logits[start:end, d:d+1] = models[d](batch)
    return logits

print("Collecting test logits...")
all_logits_test = collect_logits(test_imgs)
print("Collecting training logits for calibration (50000 samples)...")
all_logits_train = collect_logits(train_imgs_cal)

cal_X = all_logits_train
cal_y = train_labs_cal

# -------- Generic training --------
def train_calibration(cal_model, cal_X, cal_y, test_X, test_y,
                      lr=0.005, epochs=30, batch_size=256,
                      milestones=[500, 1500, 3000]):
    optimizer = nn.optimizer.SGD(init_lr=lr, model=cal_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=cal_model, max_classes=10)

    for epoch in range(epochs):
        idx = np.random.permutation(cal_X.shape[0])
        X_shuf = cal_X[idx]
        y_shuf = cal_y[idx]
        for b in range(0, X_shuf.shape[0], batch_size):
            batch_X = X_shuf[b:b+batch_size]
            batch_y = y_shuf[b:b+batch_size]
            if batch_X.shape[0] == 0:
                break
            logits = cal_model(batch_X)
            _ = loss_fn(logits, batch_y)
            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    test_out = cal_model(test_X)
    test_acc = (np.argmax(test_out, axis=1) == test_y).mean()
    return test_acc

# -------- Pairwise features helper --------
def pairwise_augment(X):
    batch = X.shape[0]
    parts = [X]
    for i in range(10):
        for j in range(i+1, 10):
            parts.append(X[:, i:i+1] - X[:, j:j+1])
    return np.concatenate(parts, axis=1)  # [batch, 55]

# -------- MLP Calibration base class --------
class MLPCalibration(nn.op.Layer):
    def __init__(self, hidden_dims=[32], input_dim=10):
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
            self.layers.append(fc)
            self.layers.append(relu)
            prev_dim = hd
        self.fc_out = nn.op.Linear(prev_dim, 10)
        self.layers.append(self.fc_out)

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        x = self.fc1(X)
        x = self.relu1(x)
        for fc, relu in zip(self.fcs, self.relus):
            x = fc(x)
            x = relu(x)
        return self.fc_out(x)

    def backward(self, g):
        g = self.fc_out.backward(g)
        for fc, relu in zip(reversed(self.fcs), reversed(self.relus)):
            g = relu.backward(g)
            g = fc.backward(g)
        g = self.relu1.backward(g)
        g = self.fc1.backward(g)
        return g

# -------- Pairwise MLP calibration --------
class PairwiseMLPCalibration(nn.op.Layer):
    def __init__(self, hidden_dims=[64, 32]):
        super().__init__()
        self.optimizable = False
        self.mlp = MLPCalibration(hidden_dims, input_dim=55)
        self.layers = self.mlp.layers

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        augmented = pairwise_augment(X)
        return self.mlp(augmented)

    def backward(self, g):
        return self.mlp.backward(g)

# -------- Experiments --------
print(f"\n{'='*60}")
print(f"EXTENDED CALIBRATION (trained on 50000 training logits)")
print(f"{'='*60}")

# E1: MLP on training set
print("\nE1: MLP(10->64->32->10) on training set...")
acc = train_calibration(MLPCalibration([64, 32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=30)
print(f"  Test acc: {acc:.4f}")

# E2: Deeper MLP on training set
print("\nE2: Deep MLP(10->128->64->32->10) on training set...")
acc = train_calibration(MLPCalibration([128, 64, 32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=30)
print(f"  Test acc: {acc:.4f}")

# E3: Pairwise MLP on training set
print("\nE3: Pairwise MLP(55->64->32->10) on training set...")
acc = train_calibration(PairwiseMLPCalibration([64, 32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=30)
print(f"  Test acc: {acc:.4f}")

# E4: Pairwise + deeper MLP on training set
print("\nE4: Pairwise Deep MLP(55->128->64->32->10) on training set...")
acc = train_calibration(PairwiseMLPCalibration([128, 64, 32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=30)
print(f"  Test acc: {acc:.4f}")

# E5: MLP with dropout-like L2 regularization on training set
print("\nE5: Deep MLP(10->128->64->32->10) with L2 reg...")
cal = MLPCalibration([128, 64, 32])
acc = train_calibration(cal, cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=40, milestones=[1000, 3000, 5000])
print(f"  Test acc: {acc:.4f}")

print(f"\n{'='*60}")
print(f"SUMMARY (all calibration methods)")
print(f"{'='*60}")
print(f"Original multi-class CNN:                             0.9800")
print(f"MLP baseline:                                         0.9475")
print(f"BCCNN paper reference:                                ~0.9803")
print(f"")
print(f"OvA no calibration:                                   0.9493")
print(f"OvA + Linear(10,10) on valid:                         0.9492")
print(f"OvA + MLP(10->32->10) on valid:                       0.9508")
print(f"OvA + Deep MLP(10->64->32->10) on valid:              0.9556")
print(f"OvA + Element-wise affine on valid:                   0.9500")
print(f"OvA + Pairwise(55->10) on valid:                      0.9563")
print(f"OvA + MLP(10->64->32->10) on 50K train:               (see above)")
print(f"OvA + Deep MLP(10->128->64->32->10) on 50K train:     (see above)")
print(f"OvA + Pairwise MLP(55->64->32->10) on 50K train:      (see above)")
print(f"OvA + Pairwise Deep MLP(55->128->64->32->10) on 50K:  (see above)")
