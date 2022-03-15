import os
import abc
import sys
from itertools import chain

import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Sequence
from torch.utils.tensorboard import SummaryWriter
from .train_results import FitResult, BatchResult, EpochResult

from .classifier import Classifier


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
            self, model: nn.Module, device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(
            self,
            dl_train: DataLoader,
            dl_val: DataLoader,
            dl_test: DataLoader,
            num_epochs: int,
            checkpoints: str = None,
            early_stopping: int = None,
            print_every: int = 1,
            **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()

        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        best_acc = None
        best_loss = None
        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if print_every > 0 and (
                    epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            actual_num_epochs += 1
            train_losses, train_accuracy,train_accuracy_top_k, train_avg_losses = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_acc.append(train_accuracy)
            batch_train_loss = sum(train_losses) / len(train_losses)
            train_loss.append(batch_train_loss)

            val_result = self.val_epoch(dl_val, verbose=verbose, **kw)
            val_accuracy = val_result.accuracy
            val_accuracy_top_k = val_result.accuracy_top_k
            val_losses = val_result.losses
            val_avg_losses = val_result.avg_losses
            val_acc.append(val_accuracy)
            batch_val_loss = sum(val_losses) / len(val_losses)
            val_loss.append(batch_val_loss)

            # Invoke scheduler at end of epoch
            self.scheduler.step(batch_val_loss)

            # + operator is used to perform task of concatenation
            train_avg_losses = {'train_' + str(key): val for key, val in train_avg_losses.items()}
            val_avg_losses = {'val_' + str(key): val for key, val in val_avg_losses.items()}
            avg_losses = dict(chain(train_avg_losses.items(), val_avg_losses.items()))
            # ...log the running loss
            writer.add_scalars(f'./Loss/',
                               {
                                   'Training Total Loss': batch_train_loss,
                                   'Val Total Loss': batch_val_loss
                               },
                               epoch)
            writer.add_scalars(f'./AVG_LOSSES/',
                               avg_losses,
                               epoch)
            # ...log the running loss
            writer.add_scalars(f'./Accuracy/',
                               {
                                   'Training Accuracy': train_accuracy,
                                   'Val Accuracy': val_accuracy
                               },
                               epoch)
            writer.add_scalars(f'./Accuracy_Top_K/',
                               {
                                   'Training Accuracy': train_accuracy_top_k,
                                   'Val Accuracy': val_accuracy_top_k
                               },
                               epoch)

            if best_loss is None or batch_val_loss < best_loss:

                # There is improvement
                epochs_without_improvement = 0
                if checkpoints:
                    self.save_checkpoint(checkpoints)
                best_loss = batch_val_loss

            else:
                # No improvement
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement is early_stopping:
                    print("invoking early stop")
                    break

        # Test model on test data
        test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
        test_accuracy = test_result.accuracy
        test_accuracy_top_k = test_result.accuracy_top_k
        test_losses = test_result.losses
        batch_test_loss = sum(test_losses) / len(test_losses)
        writer.add_scalars(f'./Final Test Results/',
                           {
                               'Loss': batch_test_loss,
                               'Accuracy': test_accuracy,
                               'Accuracy Top K': test_accuracy_top_k,
                           },
                           epoch)

        return FitResult(actual_num_epochs, train_loss, train_acc, val_loss, val_acc)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def val_epoch(self, dl_val: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_val, self.val_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def val_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        total_losses = []
        losses_dicts = []
        num_correct = 0
        num_top_k = 0
        num_samples = 0
        num_batches = len(dl.batch_sampler)
        if max_batches is not None:
            if max_batches <= num_batches:
                num_batches = max_batches
                #num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses_dicts.append(batch_res.losses)
                total_losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                num_top_k += batch_res.num_top_k
                num_samples += batch_res.n_samples
            avg_loss = sum(total_losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            accuracy_top_k = 100.0 * num_top_k / num_samples
            # Calc avg loss for each component
            avg_losses = {}
            loss_fcs_names = losses_dicts[0].keys()
            losses_tensor = torch.empty(num_batches)
            for loss_fcn_name in loss_fcs_names:
                for i, losses_dict in enumerate(losses_dicts):
                    losses_tensor[i] = losses_dict[loss_fcn_name]
                avg_losses[loss_fcn_name] = torch.mean(losses_tensor)
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.3f}, "
                f"Accuracy Top K {accuracy_top_k:.3f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=total_losses, accuracy=accuracy,accuracy_top_k=accuracy_top_k, avg_losses=avg_losses)


class ClassifierTrainer(Trainer):
    """
    Trainer for our Classifier-based models.
    """

    def __init__(
            self,
            model: Classifier,
            features_loss_fns: Sequence[nn.Module],
            features_loss_weights: Sequence[nn.Module],
            label_loss_fns: Sequence[nn.Module],
            label_loss_weights: Sequence[nn.Module],
            optimizer: Optimizer,
            scheduler: ReduceLROnPlateau,
            device: Optional[torch.device] = None,
            train_nn_space=None,
            val_nn_space=None,
            test_nn_space=None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fns: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        assert len(features_loss_fns) == len(features_loss_weights)
        assert len(label_loss_fns) == len(label_loss_weights)
        super().__init__(model, device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.features_loss_fns = features_loss_fns
        self.features_loss_weights = features_loss_weights
        self.label_loss_fns = label_loss_fns
        self.label_loss_weights = label_loss_weights
        self.train_nn_space = {'labels': train_nn_space['labels'].to(device),
                               'features': train_nn_space['features'].to(device)}
        self.val_nn_space = {'labels': val_nn_space['labels'].to(device),
                             'features': val_nn_space['features'].to(device)}
        self.test_nn_space = {'labels': test_nn_space['labels'].to(device),
                              'features': test_nn_space['features'].to(device)}

    def calc_loss(self, input, ground_truth, loss_fcs, loss_weights):
        batch_loss = None
        losses = {}
        for loss_fn, loss_weight in zip(loss_fcs, loss_weights):
            if loss_weight == 0:
                continue
            loss = loss_fn(input, ground_truth)  # Y is GT of embedding vector
            if type(loss_fn) is nn.CosineSimilarity:
                loss = torch.mean(1 - loss)
            weighted_loss = loss_weight * loss
            batch_loss = batch_loss + weighted_loss if batch_loss is not None else weighted_loss
            losses[type(loss_fn).__name__] = weighted_loss.item()
        return batch_loss, losses

    def distance_matrix(self, x, y=None, p=2):  # pairwise distance of vectors

        y = x if type(y) == type(None) else y


        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        torch.cosine_similarity(x,y)
        #dist_mse = torch.pow(x - y, p).sum(2)
        dist_cosine = -(torch.cosine_similarity(x, y, dim=2))
        return dist_cosine

    def forward_pass(self, X1, X2, y, nn_space,calc_acc=True):

        # Forward Pass
        k = 5
        transformed_features, y_hat = self.model(X1)

        if calc_acc:
            n_batches = 4
            #top_k = []
            #y_hat = []

            #n_features_in_batch = X1.shape[0] // n_batches
            dist = self.distance_matrix(transformed_features, nn_space['features'], 2)
            knn = dist.topk(k, largest=False, sorted=True)
            top_k = nn_space['labels'][knn.indices]
            y_hat = top_k[:, 0]
            # for i in range(0, n_batches):
            #     x1_batch = X1[i * n_features_in_batch:(i + 1) * n_features_in_batch].to('cpu')
            #     dist = self.distance_matrix(x1_batch, nn_features, 2) ** (1 / 2)
            #     knn = dist.topk(k, largest=False, sorted=True)
            #     top_k += [nn_space['labels'][knn.indices].to(self.device)]
            #     y_hat += [top_k[0]]
            # top_k = torch.vstack(top_k)
            # y_hat = torch.vstack(y_hat)
        else:
            y_hat = y_hat.to(self.device)
            top_k = torch.zeros(size = y.shape).to(self.device) -1
        feature_loss, features_losses = self.calc_loss(input=transformed_features, ground_truth=X2,
                                                       loss_fcs=self.features_loss_fns,
                                                       loss_weights=self.features_loss_weights)
        label_loss, label_losses = self.calc_loss(input=y_hat, ground_truth=y, loss_fcs=self.label_loss_fns,
                                                  loss_weights=self.label_loss_weights)
        # Merge losses
        features_losses = {'feature_' + str(key): val for key, val in features_losses.items()}
        label_losses = {'label_' + str(key): val for key, val in label_losses.items()}
        losses = dict(chain(features_losses.items(), label_losses.items()))

        batch_loss = feature_loss
        if label_loss is not None:
            batch_loss += + label_loss
        if not calc_acc:
            _, y_indices = torch.max(y_hat, dim=1)
            num_correct = (y_indices == y).sum()
        else:
            num_correct = (y_hat == y).sum()

        # Calc top k
        num_top_k = torch.any(top_k == torch.unsqueeze(y,dim=1),dim=1).sum()
        return batch_loss, num_correct,num_top_k, losses

    def train_batch(self, batch) -> BatchResult:
        X1, X2, y = batch

        if self.device:
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        # Forward Pass

        batch_loss, num_correct,num_top_k, losses = self.forward_pass(X1, X2, y, self.train_nn_space, calc_acc = False)

        # Backward-pass + Update parameters

        if self.optimizer:
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        return BatchResult(batch_loss.item(), num_correct.item(),num_top_k.item(), losses,y.numel())

    def val_batch(self, batch) -> BatchResult:
        X1, X2, y = batch
        if self.device:
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        with torch.no_grad():
            # Forward Pass
            batch_loss, num_correct,num_top_k, losses = self.forward_pass(X1, X2, y, self.val_nn_space)

        return BatchResult(batch_loss.item(), num_correct.item(),num_top_k.item(), losses,y.numel())

    def test_batch(self, batch) -> BatchResult:
        X1, X2, y = batch
        if self.device:
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            y = y.to(self.device)

        self.model: Classifier
        batch_loss: float
        num_correct: int

        with torch.no_grad():
            # Forward Pass
            batch_loss, num_correct,num_top_k, losses = self.forward_pass(X1, X2, y, self.test_nn_space)

        return BatchResult(batch_loss.item(), num_correct.item(),num_top_k.item(), losses,y.numel())