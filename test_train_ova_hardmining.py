"""Retrain 10 binary CNNs with Hard Negative Mining based on Round 1 error analysis.

For each binary CNN (digit k), we weight negative samples by how often they were
confused with digit k in Round 1. Digits that look similar to k get higher sampling weight.
"""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

np.random.seed(309)

# -------- Confusion data from Round 1 error analysis --------
# Each entry: {confused_digit: false_positive_count}
# Only digits with FP count > 0 are listed
ROUND1_FP = {
    0: {5: 8, 6: 7, 2: 3, 3: 3, 8: 3, 4: 1, 7: 1},
    1: {5: 8, 2: 6, 6: 6, 8: 6, 4: 2, 9: 2},
    2: {7: 15, 3: 11, 1: 6, 8: 6, 4: 5, 5: 4, 9: 1},
    3: {5: 46, 2: 12, 8: 11, 0: 5, 9: 4, 1: 3, 4: 1, 7: 1},
    4: {2: 21, 6: 15, 9: 11, 5: 6, 7: 6, 0: 3, 8: 3, 1: 2},
    5: {3: 17, 6: 16, 8: 8, 9: 6, 0: 4, 1: 2, 7: 2, 4: 1},
    6: {5: 23, 2: 16, 0: 12, 8: 6, 4: 5, 3: 1, 7: 1},
    7: {9: 23, 2: 17, 3: 11, 5: 4, 0: 2, 8: 2, 1: 1, 4: 1},
    8: {2: 21, 5: 17, 1: 11, 3: 9, 0: 3, 4: 3, 9: 3, 6: 2, 7: 1},
    9: {4: 33, 7: 21, 8: 6, 3: 5, 5: 5, 1: 1, 2: 1},
}

def build_neg_weights(digit, alpha=3.0):
    """Build per-digit negative sampling weights based on Round 1 FP data.
    Returns array weight[j] for j=0..9 (weight for digit itself is 0).
    alpha controls how aggressively we upweight hard negatives."""
    fp_data = ROUND1_FP[digit]
    total_fp = sum(fp_data.values())
    weights = np.ones(10)
    for confused_digit, count in fp_data.items():
        # Weight proportional to confusion rate, scaled by alpha
        confusion_rate = count / total_fp
        weights[confused_digit] = 1.0 + alpha * confusion_rate
    weights[digit] = 0  # Don't sample positives as negatives
    return weights / weights.sum()  # Normalize to probability distribution


# -------- Load data --------
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
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs.astype(np.float64) / 255.0
valid_imgs = valid_imgs.astype(np.float64) / 255.0

# -------- Hyperparameters --------
batch_size = 64
num_epochs = 5
init_lr = 0.06
milestones = [800, 2400, 4000]
lr_gamma = 0.5
log_iters = 100
alpha = 3.0  # Hard negative upweight factor

save_root = './ova_models_v2'
if not os.path.exists(save_root):
    os.mkdir(save_root)

# -------- Train each binary CNN with hard negative mining --------
for digit in range(10):
    print("\n" + "=" * 50)
    print(f"Training Binary CNN for digit {digit} (Hard Negative Mining)")
    print("=" * 50)

    if WANDB_AVAILABLE:
        wandb.init(project="nn-project1", name=f"OvA-HNM-digit-{digit}", reinit=True)

    train_bin = (train_labs == digit).astype(np.float64).reshape(-1, 1)
    valid_bin = (valid_labs == digit).astype(np.float64).reshape(-1, 1)

    pos_idx = np.where(train_bin.flatten() == 1)[0]
    neg_idx_all = np.where(train_bin.flatten() == 0)[0]
    neg_labels_all = train_labs[neg_idx_all]

    # Build hard negative weights
    neg_weights = build_neg_weights(digit, alpha=alpha)
    sample_weights = neg_weights[neg_labels_all]  # Weight for each negative sample
    sample_probs = sample_weights / sample_weights.sum()

    model = nn.models.Model_BinaryCNN()
    optimizer = nn.optimizer.SGD(init_lr=init_lr, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lr_gamma)
    loss_fn = nn.op.BinaryCrossEntropyLoss(model=model)

    half_batch = batch_size // 2
    best_score = 0
    iteration = 0
    save_dir = os.path.join(save_root, f'digit_{digit}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Print sampling weights for info
    print(f"  Hard negative weights (alpha={alpha}):")
    for d in range(10):
        if d != digit and neg_weights[d] > 1.01 / 9:  # Only show elevated weights
            print(f"    digit {d}: {neg_weights[d]:.4f} (vs uniform {1/9:.4f})")

    for epoch in range(num_epochs):
        np.random.shuffle(pos_idx)

        # Weighted sampling of negatives
        n_neg_needed = len(pos_idx)  # equal number
        hard_neg_idx = np.random.choice(neg_idx_all, size=n_neg_needed, replace=True, p=sample_probs)

        pos_ptr = 0
        neg_ptr = 0
        n_batches = min(len(pos_idx), len(hard_neg_idx)) // half_batch

        for _ in range(n_batches):
            if pos_ptr + half_batch > len(pos_idx):
                np.random.shuffle(pos_idx)
                pos_ptr = 0
                hard_neg_idx = np.random.choice(neg_idx_all, size=n_neg_needed, replace=True, p=sample_probs)
                neg_ptr = 0

            batch_pos = pos_idx[pos_ptr:pos_ptr + half_batch]
            batch_neg = hard_neg_idx[neg_ptr:neg_ptr + half_batch]
            pos_ptr += half_batch
            neg_ptr += half_batch

            batch_idx = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch_idx)

            train_X = train_imgs[batch_idx]
            train_y = train_bin[batch_idx]

            logits = model(train_X)
            trn_loss = loss_fn(logits, train_y)

            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            preds = (probs > 0.5).astype(np.float64)
            trn_acc = (preds == train_y).mean()

            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if iteration % log_iters == 0:
                valid_logits = model(valid_imgs)
                valid_loss = loss_fn(valid_logits, valid_bin)
                valid_probs = 1.0 / (1.0 + np.exp(-np.clip(valid_logits, -50, 50)))
                valid_preds = (valid_probs > 0.5).astype(np.float64)
                valid_acc = (valid_preds == valid_bin).mean()

                print(f"d={digit} epoch={epoch} iter={iteration} lr={optimizer.init_lr:.4f}")
                print(f"  train loss={trn_loss:.4f} acc={trn_acc:.4f}  valid loss={valid_loss:.4f} acc={valid_acc:.4f}")

                if valid_acc > best_score:
                    save_path = os.path.join(save_dir, 'best_model.pickle')
                    model.save_model(save_path)
                    print(f"  >> best acc: {best_score:.5f} -> {valid_acc:.5f}")
                    best_score = valid_acc

                if WANDB_AVAILABLE:
                    wandb.log({
                        "train/loss": trn_loss, "train/acc": trn_acc,
                        "valid/loss": valid_loss, "valid/acc": valid_acc,
                        "train/lr": optimizer.init_lr, "iteration": iteration
                    })

            iteration += 1

    print(f"Digit {digit} final best valid accuracy: {best_score:.4f}")

    if WANDB_AVAILABLE:
        wandb.finish()

print("\nAll 10 binary CNNs retrained with Hard Negative Mining!")
