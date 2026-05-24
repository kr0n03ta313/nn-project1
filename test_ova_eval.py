"""Evaluate One-vs-All binary CNNs with and without calibration layer."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import os

np.random.seed(309)

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

# Also load validation data for calibration training
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

# -------- Load 10 binary CNN models --------
print("Loading 10 binary CNN models...")
models = []
for digit in range(10):
    m = nn.models.Model_BinaryCNN()
    m.load_model(f'./ova_models/digit_{digit}/best_model.pickle')
    models.append(m)
    print(f"  Digit {digit}: loaded")

# -------- Evaluate without calibration --------
print("\n" + "=" * 50)
print("EVALUATION WITHOUT CALIBRATION (raw logit argmax)")
print("=" * 50)

# Batch evaluation to avoid memory issues
eval_bs = 256
all_logits_test = np.zeros((test_imgs.shape[0], 10))
for start in range(0, test_imgs.shape[0], eval_bs):
    end = min(start + eval_bs, test_imgs.shape[0])
    batch = test_imgs[start:end]
    for digit in range(10):
        all_logits_test[start:end, digit:digit+1] = models[digit](batch)

# argmax over raw logits
pred_labels = np.argmax(all_logits_test, axis=1)
acc_no_cal = (pred_labels == test_labs).mean()
print(f"Test Accuracy (no calibration): {acc_no_cal:.4f}")

# Also evaluate on validation set for calibration training
all_logits_valid = np.zeros((valid_imgs.shape[0], 10))
for start in range(0, valid_imgs.shape[0], eval_bs):
    end = min(start + eval_bs, valid_imgs.shape[0])
    batch = valid_imgs[start:end]
    for digit in range(10):
        all_logits_valid[start:end, digit:digit+1] = models[digit](batch)
pred_valid = np.argmax(all_logits_valid, axis=1)
acc_valid_no_cal = (pred_valid == valid_labs).mean()
print(f"Valid Accuracy (no calibration): {acc_valid_no_cal:.4f}")

# Per-digit score statistics
print("\nPer-digit raw logit statistics on test set:")
for digit in range(10):
    mask = test_labs == digit
    digit_logits = all_logits_test[mask, digit]
    other_logits = all_logits_test[~mask, digit]
    print(f"  Digit {digit}: pos_mean={digit_logits.mean():.4f} pos_std={digit_logits.std():.4f}  "
          f"neg_mean={other_logits.mean():.4f} neg_std={other_logits.std():.4f}")

# -------- Calibration Layer Training --------
print("\n" + "=" * 50)
print("TRAINING CALIBRATION LAYER")
print("=" * 50)

# Calibration: Linear(10, 10) with softmax
cal_layer = nn.op.Linear(10, 10)

# A wrapper model for calibration training
class CalibrationModel(nn.op.Layer):
    def __init__(self, cal_layer):
        super().__init__()
        self.optimizable = False
        self.cal_layer = cal_layer
        self.layers = [cal_layer]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return self.cal_layer(X)

    def backward(self, loss_grad):
        return self.cal_layer.backward(loss_grad)

cal_model = CalibrationModel(cal_layer)
cal_loss_fn = nn.op.MultiCrossEntropyLoss(model=cal_model, max_classes=10)
cal_optimizer = nn.optimizer.SGD(init_lr=0.01, model=cal_model)
cal_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=cal_optimizer, milestones=[200, 500], gamma=0.5)

batch_size = 128
n_epochs = 10
best_cal_acc = 0
iteration = 0

for epoch in range(n_epochs):
    idx = np.random.permutation(valid_imgs.shape[0])
    X_cal = all_logits_valid[idx]
    y_cal = valid_labs[idx]

    for b in range(0, X_cal.shape[0], batch_size):
        batch_X = X_cal[b:b+batch_size]
        batch_y = y_cal[b:b+batch_size]
        if batch_X.shape[0] == 0:
            break

        logits = cal_model(batch_X)
        loss = cal_loss_fn(logits, batch_y)
        cal_loss_fn.backward()
        cal_optimizer.step()
        if cal_scheduler is not None:
            cal_scheduler.step()

        if iteration % 50 == 0:
            # Evaluate on full validation
            val_out = cal_model(all_logits_valid)
            val_acc = (np.argmax(val_out, axis=1) == valid_labs).mean()

            test_out = cal_model(all_logits_test)
            test_acc = (np.argmax(test_out, axis=1) == test_labs).mean()
            print(f"cal epoch={epoch} iter={iteration} lr={cal_optimizer.init_lr:.4f}  "
                  f"valid acc={val_acc:.4f}  test acc={test_acc:.4f}")

            if val_acc > best_cal_acc:
                best_cal_acc = val_acc

        iteration += 1

# -------- Final Evaluation --------
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

test_out_cal = cal_model(all_logits_test)
acc_with_cal = (np.argmax(test_out_cal, axis=1) == test_labs).mean()

print(f"OVA (10 binary CNNs, no calibration):  {acc_no_cal:.4f}")
print(f"OVA (10 binary CNNs, with calibration): {acc_with_cal:.4f}")
print(f"Original multi-class CNN:                0.9800")
print(f"Original MLP:                            0.9475")

# Save calibration layer
cal_data = {'W': cal_layer.W, 'b': cal_layer.b}
with open('./ova_models/cal_layer.pickle', 'wb') as f:
    pickle.dump(cal_data, f)
print("\nCalibration layer saved.")
