from calendar import c
import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='green', marker='o', label='class 1', alpha=0.5)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red',   marker='s', label='class 2 (or -1)', alpha=0.5)
    plt.title('Data')
    plt.xlim([-1, 0.3])
    plt.ylim([-1, 0.3])
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig("../images/train_features.jpg")
    plt.clf()
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    xs = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    ys = -W[0, 0]/W[0, 2] - W[0, 1]/W[0, 2]*xs

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='green', marker='o', label='class 1', alpha=0.5)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red',   marker='s', label='class 2 (or -1)', alpha=0.5)
    plt.plot(xs, ys, c='orange', label='Decision Boundary', linestyle='-.')
    plt.title('Binary Classification with Logistic Regression')
    plt.xlim([-1.1, 0.3])
    plt.ylim([-1.1, 0.3])
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig("../images/train_result_sigmoid.jpg")
    plt.clf()
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    xs = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    ys0 = -(W[0, 0]-W[1, 0])/(W[0, 2]-W[1, 2]) - (W[0, 1]-W[1, 1])/(W[0, 2]-W[1, 2])*xs
    ys1 = -(W[0, 0]-W[2, 0])/(W[0, 2]-W[2, 2]) - (W[0, 1]-W[2, 1])/(W[0, 2]-W[2, 2])*xs
    ys2 = -(W[1, 0]-W[2, 0])/(W[1, 2]-W[2, 2]) - (W[1, 1]-W[2, 1])/(W[1, 2]-W[2, 2])*xs

    ys0_ys1 = np.maximum.reduce([ys0, ys1])
    ys0_ys2 = np.minimum.reduce([ys0, ys2])

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='green', marker='o', label='class 1', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red',   marker='s', label='class 2', alpha=0.5)
    plt.scatter(X[y==2, 0], X[y==2, 1], c='blue',  marker='^', label='class 3', alpha=0.5)

    # # original decision boundary
    # plt.plot(xs, ys0, c='yellow', label='DB between class 1 and 2', alpha=0.5)
    # plt.plot(xs, ys1, c='black',  label='DB between class 1 and 3', alpha=0.5)
    # plt.plot(xs, ys2, c='purple', label='DB between class 2 and 3', alpha=0.5)

    # new decision boundary
    # combination of 'decision boundary between class 1 and 2' + 'decision boundary between class 1 and 3'
    plt.plot(xs, ys0_ys1, c='black', alpha=0.5)
    # combination of 'decision boundary between class 1 and 2' + 'decision boundary between class 2 and 3'
    plt.plot(xs, ys0_ys2, c='black', alpha=0.5)

    plt.title('Multiclass Classification')
    plt.xlim([-1.1, 0.3])
    plt.ylim([-1.1, 0.3])
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig("../images/train_result_softmax.jpg")
    plt.clf()
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape = train_y.shape[0]

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    # logisticR_classifier = logistic_regression(learning_rate=1e-2, max_iter=1000)
    # logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    # print(logisticR_classifier.get_params())
    # print(logisticR_classifier.score(train_X, train_y))

    # This one is the best one.
    print("----------------- BEGIN: Best LR parameters, train, and valid accuracy -----------------")
    best_logisticR = logistic_regression(learning_rate=1e-2, max_iter=1000)
    best_logisticR.fit_miniBGD(train_X, train_y, 5)
    print(best_logisticR.get_params())
    print(best_logisticR.score(train_X, train_y))
    print(best_logisticR.score(valid_X, valid_y))
    print("----------------- END: Best LR parameters, train, and valid accuracy -----------------")

    # logisticR_classifier = logistic_regression(learning_rate=1e-2, max_iter=1000)
    # logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    # print(logisticR_classifier.get_params())
    # print(logisticR_classifier.score(train_X, train_y))
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    print("----------------- BEGIN: Best LR test accuracy -----------------")
    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_data)
    test_y_all, test_idx = prepare_y(test_labels)

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]

    test_y[np.where(test_y == 2)] = -1

    print("Accuracy on test data:", best_logisticR.score(test_X, test_y))
    print("----------------- END: Best LR test accuracy -----------------")
    print("***************** !Logistic Regression Sigmoid Case DONE! *****************")
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100, k=3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    print("----------------- BEGIN: Best Multi LR parameters, train, and valid accuracy -----------------")
    best_logistic_multi_R = logistic_regression_multiclass(learning_rate=1e-2, max_iter=1000, k=3)
    best_logistic_multi_R.fit_miniBGD(train_X, train_y, 5)
    print(best_logistic_multi_R.get_params())
    print(best_logistic_multi_R.score(train_X, train_y))
    print(best_logistic_multi_R.score(valid_X, valid_y))
    print("----------------- END: Best Multi LR parameters, train, and valid accuracy -----------------")
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    # visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    print("----------------- BEGIN: Best Multi LR test accuracy -----------------")
    visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())

    test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_data)
    test_y_all, _ = prepare_y(test_labels)

    print("Accuracy on test data:", best_logistic_multi_R.score(test_X_all, test_y_all))
    print("----------------- END: Best Multi LR test accuracy -----------------")
    print("***************** !Logistic Regression Multiple-class case DONE! *****************")
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2'

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0

    ###### First, fit softmax classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    print("------------BEGIN: *Binary* Connection between sigmoid and softmax------------")

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0

    compare_logistic_multi_R = logistic_regression_multiclass(learning_rate=1e-2, max_iter=10000, k=2)
    compare_logistic_multi_R.fit_miniBGD(train_X, train_y, 5)

    print("parameters -->", compare_logistic_multi_R.get_params())
    print("train accuracy -->", compare_logistic_multi_R.score(train_X, train_y))
    print("valid accuracy -->", compare_logistic_multi_R.score(valid_X, valid_y))
    print("test accuracy -->", compare_logistic_multi_R.score(test_X, test_y))
    print("------------END: *Binary* Connection between sigmoid and softmax------------")
    ### END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifier until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    print("------------START: *Multi* Connection between sigmoid and softmax------------")

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1

    compare_logisticR = logistic_regression(learning_rate=1e-2, max_iter=10000)
    compare_logisticR.fit_miniBGD(train_X, train_y, 5)

    print("parameters -->", compare_logisticR.get_params())
    print("train accuracy -->", compare_logisticR.score(train_X, train_y))
    print("valid accuracy -->", compare_logisticR.score(valid_X, valid_y))
    print("test accuracy -->", compare_logisticR.score(test_X, test_y))
    print("------------END: *Multi* Connection between sigmoid and softmax------------")
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy

    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''

    ### YOUR CODE HERE
    print("------------ START: sigmoid vs softmax weights and gradients ------------")
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]

    epoch = 1

    # Sigmoid
    train_y[np.where(train_y == 2)] = -1
    binary_classifier = logistic_regression(learning_rate=2 * 1e-2, max_iter=epoch)
    binary_classifier.fit_miniBGD(train_X, train_y, 5)
    print("weights of binary classifier:", binary_classifier.get_params())

    # Softmax with 2 classes
    train_y[np.where(train_y == -1)] = 0    # we have turn train_y to '-1' when the above sigmoid classification was done
                                            # as we set labels to 0,1 for softmax classifier, we should turn train_y to '0'

    multi_classifier = logistic_regression_multiclass(learning_rate=1e-2, max_iter=epoch, k=2)
    multi_classifier.fit_miniBGD(train_X, train_y, 5)
    print("weights of multi (k=2) classifier:", multi_classifier.get_params())
    print("------------ END: sigmoid vs softmax weights and gradients ------------")
    ### END YOUR CODE

    # ------------End------------

    

if __name__ == '__main__':
    main()
    
    
