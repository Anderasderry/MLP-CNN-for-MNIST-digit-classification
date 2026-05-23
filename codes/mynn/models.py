from .op import *
import pickle
import numpy as np


class Flatten(Layer):
    """[N, C, H, W] -> [N, C*H*W] for Linear; backward reshapes gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self._in_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self._in_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self._in_shape)


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == "Logistic":
                    raise NotImplementedError
                elif act_func == "ReLU":
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert (
            self.size_list is not None and self.act_func is not None
        ), "Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered."
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
        with open(param_list, "rb") as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(
                in_dim=self.size_list[i], out_dim=self.size_list[i + 1]
            )
            layer.W = np.asarray(param_list[i + 2]["W"], dtype=np.float64)
            layer.b = np.asarray(param_list[i + 2]["b"], dtype=np.float64)
            layer.params["W"] = layer.W
            layer.params["b"] = layer.b
            layer.weight_decay = param_list[i + 2]["weight_decay"]
            layer.weight_decay_lambda = param_list[i + 2]["lambda"]
            if self.act_func == "Logistic":
                raise NotImplementedError
            elif self.act_func == "ReLU":
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append(
                    {
                        "W": layer.params["W"],
                        "b": layer.params["b"],
                        "weight_decay": layer.weight_decay,
                        "lambda": layer.weight_decay_lambda,
                    }
                )

        with open(save_path, "wb") as f:
            pickle.dump(param_list, f)


_CNN_TAG = "Model_CNN_v1"


class Model_CNN(Layer):
    """
    Simple MNIST CNN: conv -> ReLU -> conv -> ReLU -> Flatten -> Linear.
    Expects flattened images [N, 784] or tensors [N, C, H, W] (C matches in_channels).
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=10,
        lambda_list=None,
        act_func="ReLU",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.act_func = act_func
        self.layers = []
        self._build_layers(lambda_list)

    def _build_layers(self, lambda_list):
        self.layers = []
        # 28x28 -> 28x28
        c1 = conv2D(self.in_channels, 16, kernel_size=5, stride=1, padding=2)
        # 28x28 -> 14x14
        c2 = conv2D(16, 32, kernel_size=5, stride=2, padding=2)
        flat_dim = 32 * 14 * 14
        lin = Linear(flat_dim, self.num_classes)

        if lambda_list is not None:
            c1.weight_decay = True
            c1.weight_decay_lambda = lambda_list[0]
            c2.weight_decay = True
            c2.weight_decay_lambda = lambda_list[1]
            lin.weight_decay = True
            lin.weight_decay_lambda = lambda_list[2]

        self.layers.append(c1)
        if self.act_func == "ReLU":
            self.layers.append(ReLU())
        elif self.act_func == "Logistic":
            raise NotImplementedError
        self.layers.append(c2)
        if self.act_func == "ReLU":
            self.layers.append(ReLU())
        self.layers.append(Flatten())
        self.layers.append(lin)

    def _to_nchw(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            n = X.shape[0]
            d = X.shape[1]
            side = int(np.sqrt(d))
            if side * side != d:
                raise ValueError(
                    f"Expected square spatial dim, got feature dim {d}"
                )
            return X.reshape(n, self.in_channels, side, side)
        if X.ndim == 4:
            return X
        raise ValueError(f"Expected 2D or 4D input, got shape {X.shape}")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert len(self.layers) > 0, "Model not initialized."
        out = self._to_nchw(X)
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def _layer_to_state(self, layer):
        if isinstance(layer, conv2D):
            return {
                "type": "conv2d",
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
                "W": np.asarray(layer.W),
                "b": np.asarray(layer.b),
                "weight_decay": layer.weight_decay,
                "lambda": layer.weight_decay_lambda,
            }
        if isinstance(layer, Linear):
            return {
                "type": "linear",
                "in_dim": layer.W.shape[0],
                "out_dim": layer.W.shape[1],
                "W": np.asarray(layer.W),
                "b": np.asarray(layer.b),
                "weight_decay": layer.weight_decay,
                "lambda": layer.weight_decay_lambda,
            }
        if isinstance(layer, ReLU):
            return {"type": "relu"}
        if isinstance(layer, Flatten):
            return {"type": "flatten"}
        raise TypeError(f"Unknown layer type: {type(layer)}")

    def _layer_from_state(self, state):
        t = state["type"]
        if t == "conv2d":
            L = conv2D(
                state["in_channels"],
                state["out_channels"],
                kernel_size=state["kernel_size"],
                stride=state["stride"],
                padding=state["padding"],
            )
            L.W = np.asarray(state["W"], dtype=np.float64)
            L.b = np.asarray(state["b"], dtype=np.float64)
            L.params["W"] = L.W
            L.params["b"] = L.b
            L.weight_decay = state["weight_decay"]
            L.weight_decay_lambda = state["lambda"]
            return L
        if t == "linear":
            L = Linear(state["in_dim"], state["out_dim"])
            L.W = np.asarray(state["W"], dtype=np.float64)
            L.b = np.asarray(state["b"], dtype=np.float64)
            L.params["W"] = L.W
            L.params["b"] = L.b
            L.weight_decay = state["weight_decay"]
            L.weight_decay_lambda = state["lambda"]
            return L
        if t == "relu":
            return ReLU()
        if t == "flatten":
            return Flatten()
        raise ValueError(f"Unknown layer type tag: {t}")

    def save_model(self, save_path):
        payload = [
            _CNN_TAG,
            self.in_channels,
            self.num_classes,
            self.act_func,
        ]
        for layer in self.layers:
            payload.append(self._layer_to_state(layer))
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self, param_list):
        with open(param_list, "rb") as f:
            payload = pickle.load(f)
        if payload[0] != _CNN_TAG:
            raise ValueError(
                f"Expected {_CNN_TAG!r} checkpoint, got {payload[0]!r}"
            )
        self.in_channels = int(payload[1])
        self.num_classes = int(payload[2])
        self.act_func = payload[3]
        self.layers = []
        for state in payload[4:]:
            self.layers.append(self._layer_from_state(state))
