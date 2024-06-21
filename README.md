# model_stealing
## About
This is the source code that uses large models to steal the data distribution of (small) victim models.

There are five tasks: MNIST, CIFAR10, SKIN_CANCER, IMDB.

## Attack Intuition
We compared the distribution scatter plot of the raw training data with that of the generated data. The figure indicated that the distribution of both is similar, which is our attack intuition.
