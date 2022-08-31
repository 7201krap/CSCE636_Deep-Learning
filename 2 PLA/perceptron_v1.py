import numpy as np
# Thanks to https://www.youtube.com/watch?v=t2ym2a3pb_Y


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters  # n_iters is the same as 'number of epochs'
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # do the following to make sure that y is either 0 or 1
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights = self.weights + update * x_i
                self.bias    = self.bias    + update * 1

    def predict(self, X):
        # given input data X, do prediction.
        # this function/method returns 0 or 1
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)  # return 1 when x >=0. otherwise return 0
