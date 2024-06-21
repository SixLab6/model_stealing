# model_stealing
## About
This is the source code that uses large models to steal the data distribution of (small) victim models.

There are five tasks: MNIST, CIFAR10, SKIN_CANCER, IMDB.

## Attack Intuition
We compared the distribution scatter plot of the raw training data with that of the generated data. Taking CIFAR10 task as an example, the figure is as follow. It is our attack intuition since the distributions of both are similar.
<p align="center">
    <img src="./img/compare.png" alt="compare" width="800" height="210">
</p>
