"""Part C: Direction 1 (Momentum SGD) + Direction 2 (Dropout Regularization)."""
import mynn as nn
import numpy as np
from struct import unpack
import gzip, pickle, os

np.random.seed(309)

# -------- Load data --------
with gzip.open(r'.\dataset\MNIST\train-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(r'.\dataset\MNIST\train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

with open('idx.pickle', 'rb') as f: idx = pickle.load(f)
train_imgs = train_imgs[idx]; train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]; valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]; train_labs = train_labs[10000:]
train_imgs = train_imgs.astype(np.float64) / 255.0
valid_imgs = valid_imgs.astype(np.float64) / 255.0

# -------- Test data --------
with gzip.open(r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz', 'rb') as f:
    magic, tnum, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(tnum, 28*28)
with gzip.open(r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz', 'rb') as f:
    magic, tnum = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
test_imgs = test_imgs.astype(np.float64) / 255.0

# -------- Common hyperparams --------
batch_size = 64; num_epochs = 5; init_lr = 0.06
milestones = [800, 2400, 4000]; lr_gamma = 0.5

results = {}

def run_exp(name, model, optimizer, scheduler):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                batch_size=batch_size, scheduler=scheduler)
    save_dir = f'./partc_{name.replace(" ","_")}'
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                 num_epochs=num_epochs, log_iters=100, save_dir=save_dir)

    # Reload best and test
    model2 = type(model)()
    if hasattr(model, 'dropout_p'):
        model2 = type(model)(p=model.dropout_p)
    model2.load_model(os.path.join(save_dir, 'best_model.pickle'))
    model2.dropout.eval() if hasattr(model2, 'dropout') else None
    test_acc = (np.argmax(model2(test_imgs), axis=1) == test_labs).mean()

    print(f"{name}  best dev: {runner.best_score:.4f}  test: {test_acc:.4f}")
    results[name] = {'dev': runner.best_score, 'test': test_acc}
    return runner

# -------- Baseline: Vanilla SGD --------
print("\n## BASELINE: Vanilla SGD")
m0 = nn.models.Model_CNN()
opt0 = nn.optimizer.SGD(init_lr=init_lr, model=m0)
sch0 = nn.lr_scheduler.MultiStepLR(optimizer=opt0, milestones=milestones, gamma=lr_gamma)
run_exp("Vanilla SGD", m0, opt0, sch0)

# -------- Exp 1: Momentum SGD (mu=0.9) --------
print("\n## Direction 1: Momentum SGD")
m1 = nn.models.Model_CNN()
opt1 = nn.optimizer.MomentGD(init_lr=init_lr, model=m1, mu=0.9)
sch1 = nn.lr_scheduler.MultiStepLR(optimizer=opt1, milestones=milestones, gamma=lr_gamma)
run_exp("Momentum SGD (mu=0.9)", m1, opt1, sch1)

# -------- Exp 2: Dropout p=0.3 --------
print("\n## Direction 2a: Dropout p=0.3")
m2 = nn.models.Model_CNN_Dropout(p=0.3)
opt2 = nn.optimizer.SGD(init_lr=init_lr, model=m2)
sch2 = nn.lr_scheduler.MultiStepLR(optimizer=opt2, milestones=milestones, gamma=lr_gamma)
run_exp("Dropout p=0.3", m2, opt2, sch2)

# -------- Exp 3: Dropout p=0.5 --------
print("\n## Direction 2b: Dropout p=0.5")
m3 = nn.models.Model_CNN_Dropout(p=0.5)
opt3 = nn.optimizer.SGD(init_lr=init_lr, model=m3)
sch3 = nn.lr_scheduler.MultiStepLR(optimizer=opt3, milestones=milestones, gamma=lr_gamma)
run_exp("Dropout p=0.5", m3, opt3, sch3)

# -------- Summary --------
print(f"\n{'='*60}")
print("PART C — FINAL RESULTS")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name:<30s}  dev={r['dev']:.4f}  test={r['test']:.4f}")
