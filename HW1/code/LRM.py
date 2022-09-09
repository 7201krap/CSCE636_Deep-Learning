#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):

    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        n_classes = self.k
        
        y = np.zeros([n_samples, n_classes])
        
        for idx, label in enumerate(labels):
            # Currently the type of 'label' is float. Type of 'label' should be converted to int
            label = int(label)
            y[idx, label] = 1
        
        # We should have weights for each class.
        self.W = np.zeros([n_classes, n_features])
        all_gradients = list()

        for _ in range(self.max_iter):
            for global_idx in range(0, n_samples, batch_size):
                # reset gradient for every batch
                gradient_acc = list()

                # global_idx: index in terms of all samples
                if global_idx + batch_size > n_samples:
                    samples_size = n_samples - global_idx
                else:
                    samples_size = batch_size 
                
                # local_idx: index in terms of samples in the batch
                for local_idx in range(global_idx, global_idx + samples_size):
                    gradient_acc.append(self._gradient(X[local_idx], y[local_idx]))
                batch_gradient = np.mean(gradient_acc, axis=0)
                
                self.W = self.W + self.learning_rate * (-batch_gradient)

                all_gradients.append(batch_gradient[0])

        print("All gradients in Multi:", all_gradients)
        ### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        p = self.softmax(self.W @ _x)
        cross_entropy_derivative = p - _y

        _x = _x.reshape(-1, 1)
        cross_entropy_derivative = cross_entropy_derivative.reshape(-1, 1)

        _g = cross_entropy_derivative @ _x.transpose()
        return _g
        ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE
        return np.divide(np.exp(x), np.sum(np.exp(x)))
        ### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE
        prediction_probabilities = list()
        for idx in range(X.shape[0]):
            prediction_probability = self.softmax(self.W @ X[idx])
            prediction_probabilities.append(prediction_probability)
        
        answer = np.argmax(prediction_probabilities, axis=1)
        return answer
        ### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE
        prediction = self.predict(X)
        score = np.divide(np.sum(labels == prediction), X.shape[0]) * 100
        return score

        ### END YOUR CODE

