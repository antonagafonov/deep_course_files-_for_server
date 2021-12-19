r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4  # number of layers (not including output)
    hidden_dims = 8  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "logsoftmax"  # activation function to apply at the output layer
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.NLLLoss  # One of the torch.nn losses
    lr, weight_decay, momentum = 5e-4, 0.01, 0.99  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**

From tutorial :


How do we mitigate these errors in the practice of machine learning?

    Optimization error: Mini batches; GD variants like stochastic gradient, momentum, Adam, etc (we'll see later in the course).

    Generalization error: Get more data; get data which better represents D

    ; train-test splits; cross validation; early stopping; regularization.

    Approximation error: Use a powerful hypothesis class (e.g. DNN); more parameters; tailoring to the domain (e.g. CNN for images).



1.Generalization error - This error is measured on test set. How model we trained on train set is performing on test set. If train accuracy and test accuracy are in the same values so the model has good generalization , if test accuracy much worse than train accuracy - the model has High generalezation error.

In our case we trained fo 20 epochs with parameters written above and got after 20 epochs:
train accuracy = 93 %
test acuracy = 88.8 %
The Generalization error is pretty low, because performance on test set pretty same like on train test.

2.Aproximation error - This error is low when expressivness of model is High ,meaning very large amount of parameters relatively to $len(X)$.
In our case we have small amount of parameters (258) , and pretty complicated dataset, so thee approximation error is relativly high. Need to choose model with larger amount of parameters.

3.Optimization error - This error is comes from otimization problem and can be improved by following parameters and techniques:
* Mini batches
* GD variants like stochastic gradient
* Momentum
* Adam


"""

part1_q2 = r"""
For validation set the TP is going to be lower and FP higher ,so the FNR is going to be higher.
!!!!!!!!!!!!!!!EXplain why!!!!!!!!!!!!!!!!!!!!

tn=0.47,fp=0.0281,fn=0.0338,tp=0.468
FPR=0.0564,FNR=0.0673
tn=0.49,fp=0.0127,fn=0.109,tp=0.388
FPR=0.0253,FNR=0.219

"""

part1_q3 = r"""
The goal is to detect the desiase at the lowest possible cost and loss of life  **before any symptoms appear** 


1.A person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis and can then be treated.

In this case we still will use the optimal ROC value for threshold because the simptoms are not lethal and we are optimazing TPR vs FPR to get as much TPR .

2.A person with the disease shows no clear symptoms and may die with high probability if not diagnosed early enough, either by your model or by the expensive test.
In this case there are no clear early symptoms,so we want to choose threshold , such will yield **lowest FNR** , meaning highest TPR rate, that will yield higher FPR ,meaning more patients with no desies will be classified as positive to desies and needed to be send to further testing.Here we are optimizing for patients survaval.


"""


part1_q4 = r"""

1.Explain the decision boundaries and model performance you obtained for the columns (fixed depth, width varies).

Answer:
For fixed depth model has pieswise linear classification ,as width grows the model have more pieses of classification . We know from Aproxiamation theary that we can describe every function as number of neurons in layer goes to $N$ $\rightarrow$ $\infty$.
With growing width model classification perfoms better.

2.Explain the decision boundaries and model performance you obtained for the rows (fixed width, depth varies).

For fixed width the behavior of the model is such:
With small number of width parameters ,model is overfitting and not generalizing well,but when the amout of width parameters is high (128) , so the performance getting better with depth growing.
With growing depth model classification perfoms better , only when width is high enoght,otherwise the model overfits.


3.Compare and explain the results for the following pairs of configurations, which have the same number of total parameters:

Answer:

    for depth=1, width=32 ,validation accuracy 78.9%,test accuracy is 88.1% 
    for depth=4, width=8  ,validation accuracy 85.0%,test accuracy is 80.5%
Here shallow model performs better ,than deep model with same amount of parameters,because deeper model overfits the training data.

!!!!!!!!!!!!!!!!!! Explain why  !!!!!!!!!!!!!!!
    
    for depth=1, width=128 ,validation accuracy 91.4%,test accuracy is 89.8% 
    for depth=4, width=32  ,validation accuracy 67.8%,test accuracy is 81.8%
Shallow model performs optimally, and deep model is underfits the training data,probably needs regularization.

!!!!!!!!!!!!!!!!!! Explain why  !!!!!!!!!!!!!!!

4.Explain the effect of threshold selection on the validation set: did it improve the results on the test set? why?

Answer:

Threshold is hyper-parameters, and HP are optimized on validation set after the model already trained on training set.Threshold selection allows the model classify better and inprove TPR . 

"""
# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss  # One of the torch.nn losses
    lr, weight_decay, momentum = 5e-4, 0.01, 0.99  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""
Answer:

1.Number of parameters. Calculate the exact numbers for these two examples.

With no bias term in bottleneck block architecture there are:
1x1x256x64 + 3x3x64 + 1x1x256 = 17216 parameters

With no bias term in regular block architecture there are:
3x3x256x256 + 3x3x256*256 = 1179648 parameters

**Number of parameters in regular block 68 times bigger than in bottleneck block.**

2.Number of floating point operations required to compute an output (qualitative assessment).

Lets calculate FLOP for regular block , assumming input is 5x5 image and padding=0 ,stride=1:

For first convolution layer:

Filter is 3x3 -> 9 multiplications and 9 summations for each filter location = 18

9 change location for 5x5 image with 3x3 kernel = 9

256x256 in out channels = 256x256

Same for second convolution layer:

Filter is 3x3 -> 9 multiplications and 9 summations for each filter location = 18

1 change location for 3x3 image with 3x3 kernel = 1

256x256 in out channels = 256x256

Total we get: 18x9x256x256 + 18x1x256x256 = 1.18e+07


Lets calculate FLOP for bottleneck block , assumming input is 5x5 image and padding=0 ,stride=1:

For first convolution layer:

Filter is 1x1 -> 1 multiplications and 0 summations for each filter location = 1

9 change location for 5x5 image with 3x3 kernel = 9

256x64 in out channels = 256x64

Same for second convolution layer:

Filter is 3x3 -> 9 multiplications and 9 summations for each filter location = 18

1 change location for 3x3 image with 3x3 kernel = 1

64x64 in out channels = 64x64

For third convolution layer:

Filter is 1x1 -> 1 multiplications and 0 summations for each filter location = 1

1 change location for 3x3 image with 3x3 kernel = 1

64x256 in out channels = 64x256

Total we get:  1x9x256x64 + 18x1x64x64 + 1x1x64x256 = 2.38e+05

**Number of FLOP in bottleneck block is 1e+02 times smaller than regular block**

3.Ability to combine the input: (1) spatially (within feature maps); (2) across feature maps.

Regular block using 3x3 kernel twise , so receptive field  on output is 5x5 

Bottle neck using 1x1 3x3 and 1x1 kernels ,so receptive field on output is 3x3

So regular block has better ability ithin feature map.

Both has ability to combine across the feature maps.
"""

# ==============

# ==============
# Part 3 answers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
