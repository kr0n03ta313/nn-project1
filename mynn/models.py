from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A model with conv2D layers.
    Architecture: Conv(1,8,3,p=1) -> ReLU -> Conv(8,16,3,s=2,p=1) -> ReLU -> Flatten -> Linear(16*14*14, 10)
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 8, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(8, 16, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.fc = Linear(16 * 14 * 14, 10)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        batch_size = X.shape[0]
        x = X.reshape(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

    def backward(self, loss_grad):
        batch_size = loss_grad.shape[0]
        grad = self.fc.backward(loss_grad)
        grad = grad.reshape(batch_size, 16, 14, 14)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad.reshape(batch_size, -1)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            data = pickle.load(f)
        self.conv1.W = data[0]['W']
        self.conv1.b = data[0]['b']
        self.conv1.params['W'] = self.conv1.W
        self.conv1.params['b'] = self.conv1.b
        self.conv2.W = data[1]['W']
        self.conv2.b = data[1]['b']
        self.conv2.params['W'] = self.conv2.W
        self.conv2.params['b'] = self.conv2.b
        self.fc.W = data[2]['W']
        self.fc.b = data[2]['b']
        self.fc.params['W'] = self.fc.W
        self.fc.params['b'] = self.fc.b

    def save_model(self, save_path):
        data = [
            {'W': self.conv1.params['W'], 'b': self.conv1.params['b']},
            {'W': self.conv2.params['W'], 'b': self.conv2.params['b']},
            {'W': self.fc.params['W'], 'b': self.fc.params['b']},
        ]
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


class Model_BinaryCNN(Layer):
    """
    Binary CNN for One-vs-All classification.
    Same conv architecture as Model_CNN but outputs a single sigmoid logit.
    Architecture: Conv(1,8,3,p=1) -> ReLU -> Conv(8,16,3,s=2,p=1) -> ReLU -> Flatten -> Linear(3136, 1)
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 8, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(8, 16, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.fc = Linear(16 * 14 * 14, 1)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        batch_size = X.shape[0]
        x = X.reshape(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

    def backward(self, loss_grad):
        batch_size = loss_grad.shape[0]
        grad = self.fc.backward(loss_grad)
        grad = grad.reshape(batch_size, 16, 14, 14)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad.reshape(batch_size, -1)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            data = pickle.load(f)
        self.conv1.W = data[0]['W']
        self.conv1.b = data[0]['b']
        self.conv1.params['W'] = self.conv1.W
        self.conv1.params['b'] = self.conv1.b
        self.conv2.W = data[1]['W']
        self.conv2.b = data[1]['b']
        self.conv2.params['W'] = self.conv2.W
        self.conv2.params['b'] = self.conv2.b
        self.fc.W = data[2]['W']
        self.fc.b = data[2]['b']
        self.fc.params['W'] = self.fc.W
        self.fc.params['b'] = self.fc.b

    def save_model(self, save_path):
        data = [
            {'W': self.conv1.params['W'], 'b': self.conv1.params['b']},
            {'W': self.conv2.params['W'], 'b': self.conv2.params['b']},
            {'W': self.fc.params['W'], 'b': self.fc.params['b']},
        ]
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


# ============ Reduced Binary CNNs ============

class SmallBinaryCNN_A(Layer):
    """
    Plan A (~1,905 params): Conv(1->4,3,p=1)->ReLU->Conv(4->8,3,s=2,p=1)->ReLU->Flatten(1568)->Linear(1568,1)
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 4, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(4, 8, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.fc = Linear(8 * 14 * 14, 1)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        bs = X.shape[0]
        x = X.reshape(bs, 1, 28, 28)
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        return self.fc(x.reshape(bs, -1))
    def backward(self, g):
        bs = g.shape[0]
        g = self.fc.backward(g); g = g.reshape(bs, 8, 14, 14)
        g = self.relu2.backward(g); g = self.conv2.backward(g)
        g = self.relu1.backward(g); g = self.conv1.backward(g)
        return g.reshape(bs, -1)

    def load_model(self, path):
        with open(path, 'rb') as f: data = pickle.load(f)
        for i, layer in enumerate([self.conv1, self.conv2, self.fc]):
            layer.W = data[i]['W']; layer.b = data[i]['b']
            layer.params['W'] = layer.W; layer.params['b'] = layer.b

    def save_model(self, path):
        data = [{'W': l.params['W'], 'b': l.params['b']} for l in [self.conv1, self.conv2, self.fc]]
        with open(path, 'wb') as f: pickle.dump(data, f)


class SmallBinaryCNN_B(Layer):
    """
    Plan B (~881 params): Conv(1->2,3,p=1)->ReLU->Conv(2->4,3,s=2,p=1)->ReLU->Flatten(784)->Linear(784,1)
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 2, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(2, 4, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.fc = Linear(4 * 14 * 14, 1)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        bs = X.shape[0]
        x = X.reshape(bs, 1, 28, 28)
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        return self.fc(x.reshape(bs, -1))
    def backward(self, g):
        bs = g.shape[0]
        g = self.fc.backward(g); g = g.reshape(bs, 4, 14, 14)
        g = self.relu2.backward(g); g = self.conv2.backward(g)
        g = self.relu1.backward(g); g = self.conv1.backward(g)
        return g.reshape(bs, -1)

    def load_model(self, path):
        with open(path, 'rb') as f: data = pickle.load(f)
        for i, layer in enumerate([self.conv1, self.conv2, self.fc]):
            layer.W = data[i]['W']; layer.b = data[i]['b']
            layer.params['W'] = layer.W; layer.params['b'] = layer.b

    def save_model(self, path):
        data = [{'W': l.params['W'], 'b': l.params['b']} for l in [self.conv1, self.conv2, self.fc]]
        with open(path, 'wb') as f: pickle.dump(data, f)


class Model_CNN_Small(Layer):
    """
    Reduced multi-class CNN (~16,026 params). Same conv as Plan A but 10-class output.
    Conv(1->4,3,p=1)->ReLU->Conv(4->8,3,s=2,p=1)->ReLU->Flatten(1568)->Linear(1568,10)
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 4, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(4, 8, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.fc = Linear(8 * 14 * 14, 10)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        bs = X.shape[0]
        x = X.reshape(bs, 1, 28, 28)
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        return self.fc(x.reshape(bs, -1))

    def backward(self, g):
        bs = g.shape[0]
        g = self.fc.backward(g); g = g.reshape(bs, 8, 14, 14)
        g = self.relu2.backward(g); g = self.conv2.backward(g)
        g = self.relu1.backward(g); g = self.conv1.backward(g)
        return g.reshape(bs, -1)

    def load_model(self, path):
        with open(path, 'rb') as f: data = pickle.load(f)
        for i, layer in enumerate([self.conv1, self.conv2, self.fc]):
            layer.W = data[i]['W']; layer.b = data[i]['b']
            layer.params['W'] = layer.W; layer.params['b'] = layer.b

    def save_model(self, path):
        data = [{'W': l.params['W'], 'b': l.params['b']} for l in [self.conv1, self.conv2, self.fc]]
        with open(path, 'wb') as f: pickle.dump(data, f)


class Model_CNN_Dropout(Layer):
    """
    CNN with Dropout before FC layer.
    Conv(1,8)->ReLU->Conv(8,16,s=2)->ReLU->Flatten->Dropout(p)->Linear(3136,10)
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.optimizable = False
        self.conv1 = conv2D(1, 8, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = conv2D(8, 16, 3, stride=2, padding=1)
        self.relu2 = ReLU()
        self.dropout = Dropout(p=p)
        self.fc = Linear(16 * 14 * 14, 10)
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.dropout, self.fc]

    def __call__(self, X): return self.forward(X)
    def forward(self, X):
        bs = X.shape[0]
        x = X.reshape(bs, 1, 28, 28)
        x = self.conv1(x); x = self.relu1(x)
        x = self.conv2(x); x = self.relu2(x)
        x = x.reshape(bs, -1)
        x = self.dropout(x)
        return self.fc(x)

    def backward(self, g):
        bs = g.shape[0]
        g = self.fc.backward(g)
        g = self.dropout.backward(g)
        g = g.reshape(bs, 16, 14, 14)
        g = self.relu2.backward(g); g = self.conv2.backward(g)
        g = self.relu1.backward(g); g = self.conv1.backward(g)
        return g.reshape(bs, -1)

    def load_model(self, path):
        with open(path, 'rb') as f: data = pickle.load(f)
        for i, layer in enumerate([self.conv1, self.conv2, self.fc]):
            layer.W = data[i]['W']; layer.b = data[i]['b']
            layer.params['W'] = layer.W; layer.params['b'] = layer.b

    def save_model(self, path):
        data = [{'W': l.params['W'], 'b': l.params['b']} for l in [self.conv1, self.conv2, self.fc]]
        with open(path, 'wb') as f: pickle.dump(data, f)
