"""Train 10 binary CNN classifiers for One-vs-All MNIST classification."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

# Use saved split if exists
idx_path = 'idx_ova.pickle'
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

train_imgs = train_imgs.astype(np.float64) / 255.0
valid_imgs = valid_imgs.astype(np.float64) / 255.0

# -------- Hyperparameters --------
batch_size = 64
num_epochs = 5
init_lr = 0.06
milestones = [800, 2400, 4000]
lr_gamma = 0.5
log_iters = 100

save_root = './ova_models'
if not os.path.exists(save_root):
    os.mkdir(save_root)

# -------- Train one binary CNN per digit --------
for digit in range(10):
    print("\n" + "=" * 50)
    print(f"Training Binary CNN for digit {digit}")
    print("=" * 50)

    if WANDB_AVAILABLE:
        wandb.init(project="nn-project1", name=f"OvA-digit-{digit}", reinit=True)

    # Create binary labels
    train_bin = (train_labs == digit).astype(np.float64).reshape(-1, 1)
    valid_bin = (valid_labs == digit).astype(np.float64).reshape(-1, 1)

    # Separate positive and negative indices for balanced batching
    pos_idx = np.where(train_bin.flatten() == 1)[0]
    neg_idx = np.where(train_bin.flatten() == 0)[0]

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

    for epoch in range(num_epochs):
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        pos_ptr = 0
        neg_ptr = 0

        n_batches = min(len(pos_idx), len(neg_idx)) // half_batch

        for _ in range(n_batches):
            if pos_ptr + half_batch > len(pos_idx):
                pos_ptr = 0
                np.random.shuffle(pos_idx)
            if neg_ptr + half_batch > len(neg_idx):
                neg_ptr = 0
                np.random.shuffle(neg_idx)

            batch_pos = pos_idx[pos_ptr:pos_ptr + half_batch]
            batch_neg = neg_idx[neg_ptr:neg_ptr + half_batch]
            pos_ptr += half_batch
            neg_ptr += half_batch

            batch_idx = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch_idx)

            train_X = train_imgs[batch_idx]
            train_y = train_bin[batch_idx]

            logits = model(train_X)
            trn_loss = loss_fn(logits, train_y)

            # Accuracy on this balanced batch
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            preds = (probs > 0.5).astype(np.float64)
            trn_acc = (preds == train_y).mean()

            loss_fn.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if iteration % log_iters == 0:
                # Validation (unbalanced — evaluate accuracy as binary classifier)
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
                    print(f"  >> best acc updated: {best_score:.5f} -> {valid_acc:.5f}")
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

print("\nAll 10 binary CNNs trained!")
