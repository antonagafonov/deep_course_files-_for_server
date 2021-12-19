import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
from abc import ABC, abstractmethod
from torch import Tensor, nn
from typing import Optional
from sklearn.metrics import roc_curve


class Classifier(nn.Module, ABC):
    """
    Wraps a model which produces raw class scores, and provides methods to compute
    class labels and probabilities.
    """

    def __init__(self, model: nn.Module):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        returning (N,C) tensors where C is the number of classes.
        """
        super().__init__()
        self.model = model

        # TODO: Add any additional initializations here, if you need them.
        # ====== YOUR CODE: ======
        self.probability_score = None
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C class scores for each of N samples
        """
        z: Tensor = None

        # TODO: Implement the forward pass, returning raw scores from the wrapped model.
        # ====== YOUR CODE: ======
        z = self.model(x)
        # ========================
        assert z.shape[0] == x.shape[0] and z.ndim == 2, "raw scores should be (N, C)"
        return z

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        # TODO: Calcualtes class scores for each sample.
        # ====== YOUR CODE: ======
        z = self.forward(x)
        # ========================
        return self.predict_proba_scores(z)

    def predict_proba_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        # TODO: Calculate class probabilities for the input.
        # ====== YOUR CODE: ======
        self.softmax = nn.Softmax(dim=1)
        self.probability_score = self.softmax(z)
        return self.probability_score 
        # ========================

    def classify(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        # Calculate the class probabilities
        y_proba = self.predict_proba(x)
        # Use implementation-specific helper to assign a class based on the
        # probabilities.
        return self._classify(y_proba)

    def classify_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        y_proba = self.predict_proba_scores(z)
        return self._classify(y_proba)

    @abstractmethod
    def _classify(self, y_proba: Tensor) -> Tensor:
        pass


class ArgMaxClassifier(Classifier):
    """
    Multiclass classifier that chooses the maximal-probability class.
    """

    def _classify(self, y_proba: Tensor):
        # TODO:
        #  Classify each sample to one of C classes based on the highest score.
        #  Output should be a (N,) integer tensor.
        # ====== YOUR CODE: ======
        N = y_proba.shape[0]
        # print(f'{N=}')
        # print(f'{y_proba.shape=}')
        idx = torch.argmax(y_proba,dim=1)
        # print(f'{idx=}')
        idx = torch.reshape(idx, (N,))
        # print(f'{idx.shape=}')
        return idx
        # ========================


class BinaryClassifier(Classifier):
    """
    Binary classifier which classifies based on thresholding the probability of the
    positive class.
    """

    def __init__(
        self, model: nn.Module, positive_class: int = 1, threshold: float = 0.5
    ):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        returning (N,C) tensors where C is the number of classes.
        :param positive_class: The index of the 'positive' class (the one that's
            thresholded to produce the class label '1').
        :param threshold: The classification threshold for the positive class.
        """
        super().__init__(model)
        assert positive_class in (0, 1)
        assert 0 < threshold < 1
        self.threshold = threshold
        self.positive_class = positive_class

    def _classify(self, y_proba: Tensor):
        # TODO:
        #  Classify each sample class 1 if the probability of the positive class is
        #  greater or equal to the threshold.
        #  Output should be a (N,) integer tensor.
        # ====== YOUR CODE: ======
        # pass
        y = y_proba
        # print(f'{y.dtype=}')
        y[y >= self.threshold] = self.positive_class
        # print(f'{y=}')
        y[y < self.threshold] = 0
        # print(f'{y=}')
        return (y[:, 1]).int()
        # ========================


def plot_decision_boundary_2d(
    classifier: Classifier,
    x: Tensor,
    y: Tensor,
    dx: float = 0.1,
    ax: Optional[plt.Axes] = None,
    cmap=plt.cm.get_cmap("coolwarm"), device: Optional[torch.device] = None
):
    """
    Plots a decision boundary of a classifier based on two input features.

    :param classifier: The classifier to use.
    :param x: The (N, 2) feature tensor.
    :param y: The (N,) labels tensor.
    :param dx: Step size for creating an evaluation grid.
    :param ax: Optional Axes to plot on. If None, a new figure with one Axes will be
        created.
    :param cmap: Colormap to use.
    :return: A (figure, axes) tuple.
    """
    assert x.ndim == 2 and y.ndim == 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig, ax = ax.get_figure(), ax

    # Plot the data
    ax.scatter(
        x[:, 0].numpy(),
        x[:, 1].numpy(),
        c=y.numpy(),
        s=20,
        alpha=0.8,
        edgecolor="k",
        cmap=cmap,
    )

    # TODO:
    #  Construct the decision boundary.
    #  Use torch.meshgrid() to create the grid (x1_grid, x2_grid) with step dx on which
    #  you evaluate the classifier.
    #  The classifier predictions (y_hat) will be treated as values for which we'll
    #  plot a contour map.
    x1_grid, x2_grid, y_hat = None, None, None
    # ====== YOUR CODE: ======
    
    # credit to:
    # https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
    # https://github.com/prudvinit/MyML/blob/master/lib/neural%20networks/pytorch%20moons.py
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    # Generate a grid of points with distance h between them
    x1_grid,x2_grid=torch.meshgrid(torch.arange(x_min, x_max, dx), torch.arange(y_min, y_max, dx))
    # Predict the function value for the whole grid
    # print(f'{x1_grid.ravel().shape=}')
    # print(f'{x2_grid.ravel().shape=}')
    torch.unsqueeze(x1_grid.ravel(), 1)
    X = torch.hstack((torch.unsqueeze(x1_grid.ravel(), 1), torch.unsqueeze(x2_grid.ravel(), 1)))
    X = X.to(device)
    # print(f'{X.shape=}')
    y_hat = classifier.classify(X)
    y_hat = y_hat.reshape(x1_grid.shape)
    # ========================
    ax.contourf(x1_grid.numpy(), x2_grid.numpy(), y_hat.numpy(), alpha=0.3, cmap=cmap)
    # Plot the decision boundary as a filled contour
    
    # for cuda training
    # if not device.type == 'cuda':
    #     ax.contourf(x1_grid.numpy(), x2_grid.numpy(), y_hat.numpy(), alpha=0.3, cmap=cmap)
    # else:
    #     x1_grid_cpu = x1_grid.cpu()
    #     x2_grid_cpu = x2_grid.cpu()
    #     y_hat_cpu = y_hat.cpu()
    #     ax.contourf(x1_grid_cpu.detach().numpy(), x2_grid_cpu.detach().numpy(),
    #                 y_hat_cpu.detach().numpy(), alpha=0.3, cmap=cmap)
        
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    return fig, ax


def select_roc_thresh(
    classifier: Classifier, x: Tensor, y: Tensor, plot: bool = False, device: Optional[torch.device] = None 
):
    """
    Calculates (and optionally plot) a classification threshold of a binary
    classifier, based on ROC analysis.

    :param classifier: The BINARY classifier to use.
    :param x: The (N, D) feature tensor.
    :param y: The (N,) labels tensor.
    :param plot: Whether to also create the ROC plot.
    :param ax: If plotting, the ax to plot on. If not provided a new figure will be
        created.
    """

    # TODO:
    #  Calculate the optimal classification threshold using ROC analysis.
    #  You can use sklearn's roc_curve() which returns the (fpr, tpr, thresh) values.
    #  Calculate the index of the optimal threshold as optimal_thresh_idx.
    #  Calculate the optimal threshold as optimal_thresh.
    x = x.to(device)
    y_cpu = y
    y = y.to(device)
    fpr, tpr, thresh = None, None, None
    optimal_thresh_idx, optimal_thresh = None, None
    y_hat = classifier.predict_proba(x).detach().numpy()
    
    # training with cuda
    # if not device.type == 'cuda':
    #     y_hat = classifier.predict_proba(x).detach().numpy()
    # else:
    #     temp = classifier.predict_proba(x).cpu()
    #     y_hat = temp.detach().numpy()
        
    fpr, tpr, thresh = roc_curve(y_cpu, y_hat[:,1])
    # print(f'{len(thresh)=}')
    if (len(thresh) <= 50):
        warnings.warn('Warning: something wrong with threshold roc calculation')       
    # print(f'{y=},{y_hat[:,1]=}')
    # print(f'{fpr.shape=},{tpr.shape=},{thresh.shape=}')
    # credit to:
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    optimal_thresh_idx = np.argmax(gmeans)
    optimal_thresh = thresh[optimal_thresh_idx]
    if (optimal_thresh>=1.0):
        optimal_thresh = 0.5
    # print(f'{optimal_thresh=:.3f}')
    # ========================

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(fpr, tpr, color="C0")
        ax.scatter(
            fpr[optimal_thresh_idx], tpr[optimal_thresh_idx], color="C1", marker="o"
        )
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR=1-FNR")
        ax.legend(["ROC", f"Threshold={optimal_thresh:.2f}"])

    return optimal_thresh
