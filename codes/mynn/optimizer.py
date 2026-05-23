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
                    grad = layer.grads[key]
                    if grad is None:
                        continue
                    if layer.weight_decay:
                        # in-place: keep layer.W / layer.params['W'] as the same array
                        layer.params[key] *= 1 - self.init_lr * layer.weight_decay_lambda
                    layer.params[key] -= self.init_lr * grad


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        pass
    
    def step(self):
        pass