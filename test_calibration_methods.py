"""Test multiple calibration strategies for OvA binary CNN ensemble."""
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
    magic, num = unpack('>2I', f.read(8))
    _, num = unpack('>2I', f.read(8))
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

# -------- Load models and collect logits --------
print("Loading 10 binary CNN models and collecting logits...")
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

print("  Collecting test set logits...")
all_logits_test = collect_logits(test_imgs)
print("  Collecting validation set logits...")
all_logits_valid = collect_logits(valid_imgs)

cal_X = all_logits_valid
cal_y = valid_labs

# -------- Baseline: No calibration --------
preds_no_cal = np.argmax(all_logits_test, axis=1)
acc_no_cal = (preds_no_cal == test_labs).mean()
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"1. No calibration (raw logit argmax):           {acc_no_cal:.4f}")

# -------- Generic calibration training loop --------
def train_calibration(cal_model, cal_X, cal_y, test_X, test_y,
                      lr=0.01, epochs=20, batch_size=128,
                      milestones=[300, 800, 1500]):
    optimizer = nn.optimizer.SGD(init_lr=lr, model=cal_model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=cal_model, max_classes=10)

    for epoch in range(epochs):
        idx = np.random.permutation(cal_X.shape[0])
        X_shuf = cal_X[idx]
        y_shuf = cal_y[idx]
        step = 0
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
            step += 1

    test_out = cal_model(test_X)
    test_acc = (np.argmax(test_out, axis=1) == test_y).mean()
    return test_acc


# -------- Calibration Method 1: Linear(10,10) --------
class LinearCalibration(nn.op.Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.linear = nn.op.Linear(10, 10)
        self.layers = [self.linear]
    def __call__(self, X): return self.forward(X)
    def forward(self, X): return self.linear(X)
    def backward(self, g): return self.linear.backward(g)

print("Training Linear calibration...")
acc_linear = train_calibration(LinearCalibration(), cal_X, cal_y, all_logits_test, test_labs, lr=0.01, epochs=20)
print(f"2. Linear(10,10) calibration:                   {acc_linear:.4f}")


# -------- Calibration Method 2: MLP (10->32->10, ReLU) --------
class MLPCalibration(nn.op.Layer):
    def __init__(self, hidden_dims=[32]):
        super().__init__()
        self.optimizable = False
        self.fc1 = nn.op.Linear(10, hidden_dims[0])
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

print("Training MLP calibration (10->32->10)...")
acc_mlp = train_calibration(MLPCalibration([32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.01, epochs=30)
print(f"3. MLP(10->32->10, ReLU) calibration:           {acc_mlp:.4f}")

# -------- Calibration Method 3: Deep MLP (10->64->32->10) --------
print("Training Deep MLP calibration (10->64->32->10)...")
acc_deep_mlp = train_calibration(MLPCalibration([64, 32]), cal_X, cal_y, all_logits_test, test_labs, lr=0.01, epochs=40)
print(f"4. Deep MLP(10->64->32->10, ReLU) calibration:  {acc_deep_mlp:.4f}")

# -------- Calibration Method 4: Element-wise affine (Platt-like) --------
class ElementwiseAffine(nn.op.Layer):
    """Learn per-class scale + shift: output_i = gamma_i * input_i + beta_i"""
    def __init__(self):
        super().__init__()
        self.optimizable = True
        self.weight_decay = False
        self.gamma = np.ones((1, 10)) * 0.5
        self.beta = np.zeros((1, 10))
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {'gamma': None, 'beta': None}
        self._input = None
        self.layers = [self]

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        self._input = X
        return X * self.gamma + self.beta

    def backward(self, g):
        self.grads['gamma'] = np.sum(g * self._input, axis=0, keepdims=True)
        self.grads['beta'] = np.sum(g, axis=0, keepdims=True)
        return g * self.gamma

print("Training element-wise affine calibration...")
acc_affine = train_calibration(ElementwiseAffine(), cal_X, cal_y, all_logits_test, test_labs, lr=0.005, epochs=30)
print(f"5. Element-wise affine (per-class scale+shift):  {acc_affine:.4f}")

# -------- Calibration Method 5: Full quadratic pairwise --------
class PairwiseCalibration(nn.op.Layer):
    """Take all 45 pairwise differences, learn to combine them"""
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.fc = nn.op.Linear(10, 10)
        self.layers = [self.fc]
    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        # Augment with pairwise features
        batch = X.shape[0]
        pairwise = []
        for i in range(10):
            for j in range(i+1, 10):
                pairwise.append((X[:, i:i+1] - X[:, j:j+1]))
        pairwise = np.concatenate(pairwise, axis=1)  # [batch, 45]
        augmented = np.concatenate([X, pairwise], axis=1)  # [batch, 55]
        self.aug_input = augmented
        return self.fc(X)  # Just learn from the 10 logits for now

    def backward(self, g):
        return self.fc.backward(g)

# Better pairwise: use a Linear(55, 10) on augmented features
class PairwiseAugmented(nn.op.Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.fc = nn.op.Linear(55, 10)
        self.layers = [self.fc]
        self._input = None
    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        self._input = X
        batch = X.shape[0]
        pairwise = []
        for i in range(10):
            for j in range(i+1, 10):
                pairwise.append(X[:, i:i+1] - X[:, j:j+1])
        augmented = np.concatenate([X] + pairwise, axis=1)
        return self.fc(augmented)
    def backward(self, g): return self.fc.backward(g)

print("Training pairwise-augmented calibration (10+45=55 dims)...")
acc_pairwise = train_calibration(PairwiseAugmented(), cal_X, cal_y, all_logits_test, test_labs, lr=0.01, epochs=20)
print(f"6. Pairwise-augmented (55->10) calibration:      {acc_pairwise:.4f}")

# -------- Final Summary --------
print(f"\n{'='*60}")
print(f"FINAL COMPARISON")
print(f"{'='*60}")
print(f"Reference: Original multi-class CNN:             0.9800")
print(f"Reference: Original MLP:                         0.9475")
print(f"Reference: BCCNN paper (softmax collection):     ~0.9803")
print(f"")
print(f"1. OvA, no calibration:                         {acc_no_cal:.4f}")
print(f"2. OvA + Linear(10,10):                          {acc_linear:.4f}")
print(f"3. OvA + MLP(10->32->10, ReLU):                  {acc_mlp:.4f}")
print(f"4. OvA + Deep MLP(10->64->32->10, ReLU):         {acc_deep_mlp:.4f}")
print(f"5. OvA + Element-wise affine (gamma*x + beta):   {acc_affine:.4f}")
print(f"6. OvA + Pairwise-augmented (55->10):            {acc_pairwise:.4f}")
