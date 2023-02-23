import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os
import time


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)   # original
    # model.train(A, A, 64, 3000)   # for the last question 4-(h)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w


if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    # ## YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    ps = [32, 64, 128]

    f_error, f_error_t = list(), list()

    for p in ps:
        # PCA
        G = test_pca(A, p)
        time.sleep(5)

        # Autoencoder
        final_w = test_ae(A, p)
        time.sleep(5)

        f_error.append(frobeniu_norm_error(G, final_w))
        time.sleep(5)

        f_error_t.append(frobeniu_norm_error(np.dot(G.T, G), np.dot(final_w.T, final_w)))
        time.sleep(5)

    print("frobeniu_norm_error: G and W", f_error)
    print("frobeniu_norm_error: G.TG and W.TW", f_error_t)

    final_w = test_ae(A, 64)
    ### END YOUR CODE 
