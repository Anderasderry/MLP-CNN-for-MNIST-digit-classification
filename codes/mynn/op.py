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
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            scale = np.sqrt(2.0 / in_dim)
            self.W = np.random.randn(in_dim, out_dim) * scale
            self.b = np.zeros((1, out_dim), dtype=np.float64)
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim), dtype=np.float64)
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
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = int(kernel_size[0]), int(kernel_size[1])
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = np.sqrt(2.0 / (in_channels * kh * kw))
        self.W = initialize_method(size=(out_channels, in_channels, kh, kw)) * scale
        self.b = np.zeros((1, out_channels, 1, 1), dtype=np.float64)
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def _pad_input(self, X):
        p = self.padding
        if p > 0:
            return np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant", constant_values=0)
        return X

    def _im2col(self, Xp):
        """Xp: [N,C,H,W] -> cols [N, C*kh*kw, L], with L = out_h*out_w."""
        kh, kw = self.kernel_size
        s = self.stride
        N, C, Hp, Wp = Xp.shape
        out_h = (Hp - kh) // s + 1
        out_w = (Wp - kw) // s + 1
        L = out_h * out_w
        cols = np.zeros((N, C * kh * kw, L), dtype=np.float64)
        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = Xp[:, :, i * s : i * s + kh, j * s : j * s + kw]
                cols[:, :, col_idx] = patch.reshape(N, -1)
                col_idx += 1
        return cols, out_h, out_w

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, kh, kw] (cross-correlation, same as common CNN conv)
        """
        self.input = np.asarray(X, dtype=np.float64)
        kh, kw = self.kernel_size
        Xp = self._pad_input(self.input)
        cols, out_h, out_w = self._im2col(Xp)
        self._cols = cols
        self._out_hw = (out_h, out_w)
        self._Xp_shape = Xp.shape

        W_mat = self.W.reshape(self.out_channels, -1)
        N, _, L = cols.shape
        out = np.zeros((N, self.out_channels, out_h, out_w), dtype=np.float64)
        out_flat = np.einsum("oi,nil->nol", W_mat, cols)
        out = out_flat.reshape(N, self.out_channels, out_h, out_w)
        out += self.b.reshape(1, self.out_channels, 1, 1)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        grads = np.asarray(grads, dtype=np.float64)
        N, C_in, H, W = self.input.shape
        kh, kw = self.kernel_size
        s = self.stride
        p = self.padding
        out_h, out_w = self._out_hw
        L = out_h * out_w

        Xp = self._pad_input(self.input)
        cols = self._cols
        grad_flat = grads.reshape(N, self.out_channels, L)
        W_mat = self.W.reshape(self.out_channels, -1)

        dW = np.einsum("nol,nil->oi", grad_flat, cols).reshape(self.W.shape)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        grad_cols = np.einsum("oi,nol->nil", W_mat, grad_flat)
        dXp = np.zeros_like(Xp)
        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch_grad = grad_cols[:, :, col_idx].reshape(N, C_in, kh, kw)
                dXp[:, :, i * s : i * s + kh, j * s : j * s + kw] += patch_grad
                col_idx += 1

        if p > 0:
            dX = dXp[:, :, p : p + H, p : p + W]
        else:
            dX = dXp

        self.grads["W"] = dW
        self.grads["b"] = db
        return dX
    
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

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.optimizable = False
        self.has_softmax = True
        self.grads = None
        self._probs = None
        self._predicts = None
        self._labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self._predicts = np.asarray(predicts, dtype=np.float64)
        self._labels = np.asarray(labels).reshape(-1)
        batch_size = self._predicts.shape[0]
        if batch_size == 0:
            return 0.0

        if self.has_softmax:
            self._probs = softmax(self._predicts)
        else:
            p = np.clip(self._predicts, 1e-12, 1.0)
            self._probs = p / np.sum(p, axis=1, keepdims=True)

        idx = np.arange(batch_size)
        log_p = np.log(self._probs[idx, self._labels] + 1e-12)
        loss = -np.mean(log_p)
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # Then send the grads to model for back propagation
        batch_size = self._predicts.shape[0]
        if batch_size == 0:
            return
        if self.has_softmax:
            one_hot = np.zeros_like(self._predicts)
            one_hot[np.arange(batch_size), self._labels] = 1.0
            self.grads = (self._probs - one_hot) / batch_size
        else:
            one_hot = np.zeros_like(self._predicts)
            one_hot[np.arange(batch_size), self._labels] = 1.0
            self.grads = -one_hot / (batch_size * (self._probs + 1e-12))
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition