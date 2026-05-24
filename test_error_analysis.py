"""Error analysis: for each binary CNN, which digits are false positives/negatives."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle

np.random.seed(309)

# -------- Load training data (to analyze what the CNNs learned) --------
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

with open('idx_ova.pickle', 'rb') as f:
    idx = pickle.load(f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# Use validation set (10000) for quick analysis
valid_imgs = train_imgs[:10000].astype(np.float64) / 255.0
valid_labs = train_labs[:10000]

# -------- Load models --------
print("Loading 10 binary CNN models...")
models = []
for digit in range(10):
    m = nn.models.Model_BinaryCNN()
    m.load_model(f'./ova_models/digit_{digit}/best_model.pickle')
    models.append(m)

# -------- Evaluate each CNN on validation set, record all logits --------
eval_bs = 256
all_logits = np.zeros((valid_imgs.shape[0], 10))
for start in range(0, valid_imgs.shape[0], eval_bs):
    end = min(start + eval_bs, valid_imgs.shape[0])
    batch = valid_imgs[start:end]
    for d in range(10):
        all_logits[start:end, d:d+1] = models[d](batch)

# Apply sigmoid to get probabilities
all_probs = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -50, 50)))

# -------- Per-digit error analysis --------
print("\n" + "=" * 70)
print("PER-DIGIT BINARY CNN ERROR ANALYSIS (on 10K validation set)")
print("=" * 70)

for digit in range(10):
    print(f"\n--- Binary CNN for digit {digit} ---")
    print(f"  Threshold: prob > 0.5 => positive (is digit {digit})")
    print(f"  Optimal threshold determined per-class")

    # Find best threshold for this digit
    best_acc = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = all_probs[:, digit] > t
        true = (valid_labs == digit)
        acc = (preds == true).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    preds = all_probs[:, digit] > best_thresh
    true = (valid_labs == digit)

    total = valid_labs.shape[0]
    n_pos = true.sum()
    n_neg = total - n_pos

    tp = (preds & true).sum()
    tn = (~preds & ~true).sum()
    fp = (preds & ~true).sum()
    fn = (~preds & true).sum()

    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Recall: {tp/n_pos:.4f}  Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.4f}")
    print(f"  FPR: {fp/n_neg:.4f}  FNR: {fn/n_pos:.4f}")

    # False positives: predicted digit D but actually NOT D — break down by true digit
    if fp > 0:
        fp_mask = preds & ~true
        fp_labels = valid_labs[fp_mask]
        print(f"\n  False Positives (predicted {digit}, actually not {digit}): {fp} total")
        fp_counts = {}
        for d in range(10):
            if d == digit:
                continue
            cnt = (fp_labels == d).sum()
            if cnt > 0:
                fp_counts[d] = cnt
        # Sort by count
        sorted_fp = sorted(fp_counts.items(), key=lambda x: -x[1])
        for d, cnt in sorted_fp:
            bar = '█' * max(1, cnt * 50 // fp)
            print(f"    actually {d}: {cnt:5d} ({cnt/fp*100:5.1f}%) {bar}")

    # False negatives: actually digit D but predicted not D
    if fn > 0:
        fn_count = fn  # count of missed positives (doesn't break down by "predicted as what")
        print(f"\n  False Negatives (actually {digit}, predicted not {digit}): {fn} total ({fn/n_pos*100:.1f}% of all {digit}s)")
        # Show which other digits' CNNs scored these samples highly
        fn_mask = ~preds & true
        fn_logits = all_logits[fn_mask]  # logits from all 10 CNNs for FN samples

        # For these FN samples, which other CNNs give high scores?
        fn_scores = 1.0 / (1.0 + np.exp(-np.clip(fn_logits, -50, 50)))
        rival_counts = {}
        for d in range(10):
            if d == digit:
                continue
            rival_preds = fn_scores[:, d] > 0.5
            cnt = rival_preds.sum()
            if cnt > 0:
                rival_counts[d] = cnt
        if rival_counts:
            print(f"  For these {fn} FN samples, which other CNNs fire (>0.5):")
            sorted_rivals = sorted(rival_counts.items(), key=lambda x: -x[1])
            for d, cnt in sorted_rivals:
                bar = '█' * max(1, cnt * 50 // fn)
                print(f"    CNN-{d} fires: {cnt:5d} ({cnt/fn*100:5.1f}%) {bar}")

# -------- Summary matrix --------
print(f"\n{'='*70}")
print(f"CONFUSION MATRIX OF OvA ENSEMBLE (argmax)")
print(f"{'='*70}")

pred_labels = np.argmax(all_logits, axis=1)
confusion = np.zeros((10, 10), dtype=int)
for t in range(10):
    mask = valid_labs == t
    for p in range(10):
        confusion[t, p] = ((pred_labels == p) & mask).sum()

# Print formatted confusion matrix
print(f"\n       Predicted")
print(f"       ", end="")
for p in range(10):
    print(f"  {p:4d}", end="")
print(f"   (recall)")

for t in range(10):
    print(f"True {t}:", end="")
    for p in range(10):
        print(f"  {confusion[t,p]:4d}", end="")
    recall = confusion[t,t] / confusion[t].sum() * 100
    print(f"   {recall:5.1f}%")

# Most confused pairs
print(f"\nMost confused digit pairs (argmax misclassification):")
mistakes = []
for t in range(10):
    for p in range(10):
        if t != p and confusion[t, p] > 0:
            mistakes.append((t, p, confusion[t, p]))
mistakes.sort(key=lambda x: -x[2])
for t, p, cnt in mistakes[:15]:
    print(f"  True {t} -> predicted {p}: {cnt} times")
