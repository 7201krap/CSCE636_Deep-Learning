import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""


class logistic_regression(object):

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        # define W
        # shape of _x: [n_features,]
        # shape of  W: [n_features,]
        self.W = np.zeros([1, n_features])
        for _ in range(self.max_iter):
            full_gradient = np.mean([self._gradient(x, y) for x, y in zip(X, y)], axis=0)
            self.W = self.W + self.learning_rate * (-full_gradient)

        ### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape

        self.W = np.zeros([1, n_features])
        for _ in range(self.max_iter):
            for global_idx in range(0, n_samples, batch_size):
                if global_idx + batch_size > n_samples:
                    samples_size = n_samples - global_idx
                else:
                    samples_size = batch_size
                    
                batch_gradient = np.mean(
                    [self._gradient(x, y) for x, y in zip(X[global_idx:global_idx + samples_size], y[global_idx:global_idx + samples_size])], axis=0)
                self.W = self.W + self.learning_rate * (-batch_gradient)
        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape

        self.W = np.zeros([1, n_features])
        for _ in range(self.max_iter):
            for j in range(n_samples):
                one_sample_gradient = self._gradient(X[j], y[j])
                self.W = self.W + self.learning_rate * (-one_sample_gradient)
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        # Equation from T04-LogReg.pdf p.28
        # np.dot == element-wise multiplication and summation
        _x = np.array(_x)
        _y = np.array(_y)
        _g = -np.divide((_y * _x), (1 + np.exp(_y * np.dot(self.W, _x))))
        return _g
        ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE
        prediction = list()
        prediction_the_other = list()

        for x_single_sample in X:
            prob = 1 / (1 + np.exp(-np.dot(self.W, x_single_sample)))   # scalar value
            prediction.append(prob)                 # row vector
            prediction_the_other.append(1-prob)     # row vector

        # Vertically stack two row vectors and transpose it.
        # Then it becomes column vector with two columns.
        # This satisfies the above condition. Shape of preds_proba == [n_samples, 2]
        preds_proba = np.vstack([prediction, prediction_the_other]).transpose()

        return preds_proba
        ### END YOUR CODE

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        prediction = list()
        for x_single_sample in X:
            if np.dot(self.W, x_single_sample) >= 0:
                prediction.append(1)
            else:
                prediction.append(-1)

        # print("shape of predictions", np.array(prediction).shape)
        return prediction
        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        prediction = self.predict(X)
        score = np.divide(np.sum(y == prediction), X.shape[0]) * 100
        return score
        ### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self
