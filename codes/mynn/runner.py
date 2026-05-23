import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        dev_max_samples = kwargs.get("dev_max_samples", None)
        eval_every = kwargs.get("eval_every", 1)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        best_score = 0
        dev_score = 0.0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            num_batches = int(np.ceil(X.shape[0] / self.batch_size))
            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if iteration % eval_every == 0:
                    dev_score, dev_loss = self.evaluate(dev_set, max_samples=dev_max_samples)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)

                    if iteration % log_iters == 0:
                        print(f"epoch: {epoch}, iteration: {iteration}")
                        print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                        print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            dev_score, dev_loss = self.evaluate(dev_set, max_samples=dev_max_samples)
            print(f"epoch {epoch} end: dev_score={dev_score:.5f}, dev_loss={dev_loss:.5f}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, "best_model.pickle")
                self.save_model(save_path)
                print(f"best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set, max_samples=None):
        X, y = data_set
        if max_samples is not None and X.shape[0] > max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)