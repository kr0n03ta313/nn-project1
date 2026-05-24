from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = initialize_method(size=(in_dim, out_dim)) * scale
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        k = kernel_size
        scale = np.sqrt(2.0 / (in_channels * k * k))
        self.W = initialize_method(size=(out_channels, in_channels, k, k)) * scale
        self.b = np.zeros((1, out_channels))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.input = None
        self.col = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, kernel_size, kernel_size]
        Implements 2D convolution using im2col.
        """
        self.input = X
        batch_size, in_c, H, W_d = X.shape
        out_c = self.out_channels
        k = self.kernel_size
        s = self.stride
        p = self.padding

        new_H = (H + 2 * p - k) // s + 1
        new_W = (W_d + 2 * p - k) // s + 1

        X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        col = np.zeros((batch_size, in_c, k, k, new_H, new_W))
        for y in range(k):
            y_max = y + s * new_H
            for x in range(k):
                x_max = x + s * new_W
                col[:, :, y, x, :, :] = X_pad[:, :, y:y_max:s, x:x_max:s]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * new_H * new_W, -1)
        self.col = col

        W_col = self.W.reshape(out_c, -1)
        out = col @ W_col.T
        out = out.reshape(batch_size, new_H, new_W, out_c).transpose(0, 3, 1, 2)
        out += self.b.reshape(1, out_c, 1, 1)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, out_c, new_H, new_W = grads.shape
        in_c = self.in_channels
        k = self.kernel_size
        s = self.stride
        p = self.padding

        grad_flat = grads.transpose(0, 2, 3, 1).reshape(-1, out_c)
        W_col = self.W.reshape(out_c, -1)

        dW_col = grad_flat.T @ self.col
        self.grads['W'] = dW_col.reshape(self.W.shape)

        self.grads['b'] = np.sum(grad_flat, axis=0, keepdims=True)

        dcol = grad_flat @ W_col
        dcol = dcol.reshape(batch_size, new_H, new_W, in_c, k, k)
        dcol = dcol.transpose(0, 3, 4, 5, 1, 2)

        X_pad = np.pad(self.input, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        dX_pad = np.zeros_like(X_pad)

        for y in range(k):
            y_max = y + s * new_H
            for x in range(k):
                x_max = x + s * new_W
                np.add.at(dX_pad, (slice(None), slice(None), slice(y, y_max, s), slice(x, x_max, s)),
                          dcol[:, :, y, x, :, :])

        if p > 0:
            return dX_pad[:, :, p:-p, p:-p]
        return dX_pad

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Dropout(Layer):
    """
    Dropout regularization layer.
    p: probability of dropping a neuron (setting to zero).
    During training, each neuron is dropped with probability p and surviving
    neurons are scaled by 1/(1-p). During evaluation, the layer is a no-op.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.optimizable = False
        self.training = True
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, grads):
        if self.training:
            return grads * self.mask
        return grads


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.preds = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        loss = -np.log(probs[np.arange(batch_size), labels] + 1e-12)
        return np.mean(loss)

    def backward(self):
        batch_size = self.preds.shape[0]

        if self.has_softmax:
            probs = softmax(self.preds)
        else:
            probs = self.preds

        self.grads = probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass

class Sigmoid(Layer):
    """
    Sigmoid activation layer for binary classification.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.output = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        self.output = 1.0 / (1.0 + np.exp(-np.clip(X, -50, 50)))
        return self.output

    def backward(self, grads):
        return grads * self.output * (1.0 - self.output)

class BinaryCrossEntropyLoss(Layer):
    """
    Binary Cross-Entropy Loss with sigmoid. For One-vs-All training.
    predicts: [batch_size, 1]
    labels: [batch_size, 1] — 0 or 1
    """
    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model
        self.optimizable = False
        self.preds = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.preds = predicts
        self.labels = labels
        batch_size = predicts.shape[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(predicts, -50, 50)))
        loss = -(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
        return np.mean(loss)

    def backward(self):
        batch_size = self.preds.shape[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(self.preds, -50, 50)))
        self.grads = (probs - self.labels) / batch_size
        self.model.backward(self.grads)
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition