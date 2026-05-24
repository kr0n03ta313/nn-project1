import mynn as nn
import numpy as np
from struct import unpack
import gzip

# -------- Choose model type --------
MODEL_TYPE = 'CNN'  # Change to 'MLP' or 'CNN'

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

# -------- Load model and evaluate --------
if MODEL_TYPE == 'MLP':
    model = nn.models.Model_MLP()
    model.load_model(r'.\best_models_mlp\best_model.pickle')
else:
    model = nn.models.Model_CNN()
    model.load_model(r'.\best_models_cnn\best_model.pickle')

logits = model(test_imgs)
acc = nn.metric.accuracy(logits, test_labs)
print(f"{MODEL_TYPE} Test Accuracy: {acc:.4f}")
