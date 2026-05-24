from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}

    def step(self):
        for lid, layer in enumerate(self.model.layers):
            if layer.optimizable:
                if lid not in self.velocity:
                    self.velocity[lid] = {}
                for key in layer.params.keys():
                    if key not in self.velocity[lid]:
                        self.velocity[lid][key] = np.zeros_like(layer.params[key])
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocity[lid][key] = self.mu * self.velocity[lid][key] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.velocity[lid][key]