import numpy as np
import os

class RunnerM():
    """
    Trainer for neural network models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.use_wandb = use_wandb

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.dev_iters = []

    def _set_dropout_mode(self, training):
        for layer in self.model.layers:
            if hasattr(layer, 'training'):
                if training:
                    layer.train()
                else:
                    layer.eval()

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        iteration = 0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            n_batches = int(X.shape[0] / self.batch_size) + 1

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, X.shape[0])
                if start >= X.shape[0]:
                    break

                train_X = X[start:end]
                train_y = y[start:end]

                self._set_dropout_mode(training=True)
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if iteration % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.dev_iters.append(iteration)

                    lr = self.optimizer.init_lr
                    print(f"epoch: {epoch}, iter: {iteration}, lr: {lr:.4f}")
                    print(f"  [Train] loss: {trn_loss:.4f}, acc: {trn_score:.4f}")
                    print(f"  [Dev]   loss: {dev_loss:.4f}, acc: {dev_score:.4f}")

                    if self.use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                "train/loss": trn_loss, "train/acc": trn_score,
                                "dev/loss": dev_loss, "dev/acc": dev_score,
                                "train/lr": lr, "iteration": iteration, "epoch": epoch
                            })
                        except ImportError:
                            pass

                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"  >> best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score

                iteration += 1

        self.best_score = best_score

    def evaluate(self, data_set):
        self._set_dropout_mode(training=False)
        X, y = data_set
        eval_bs = 256
        total_loss = 0.0
        total_correct = 0
        n_samples = X.shape[0]

        for start in range(0, n_samples, eval_bs):
            end = min(start + eval_bs, n_samples)
            batch_X = X[start:end]
            batch_y = y[start:end]
            logits = self.model(batch_X)
            total_loss += self.loss_fn(logits, batch_y) * (end - start)
            pred_label = np.argmax(logits, axis=-1)
            total_correct += (pred_label == batch_y).sum()

        return total_correct / n_samples, total_loss / n_samples

    def save_model(self, save_path):
        self.model.save_model(save_path)
