import numpy as np

def linear_regression(X,Y,lam=0):
    """
    Train linear regression model, using ridge regularization

    Params:
        X (np.array): A numpy array with the shape (N, d) where N is the number of data points and d is dimension
        Y (np.array): A numpy array with the shape (N,), where N is the number of data points
        lam (int): The regularization coefficient where default value is 0
            
    Retursn:
        beta (np.array): A numpy array with the shape (d+1, 1) that represents the linear regression weight vector
    """
    assert X.ndim==2
    N = X.shape[0]
    d = X.shape[1]
    assert Y.size == N

    # Prepend a column of ones to X, to give us a Y intercept
    ones = np.ones((N, 1))
    x = np.hstack((ones, X))

    # Reshape Y to be a 2D matrix, not a vector (just for numpy calculations to work)
    y = Y.reshape(-1,1)

    # Without ridge regularization:
    #   Solve for B in the following equation:
    #     (X^T * X)B = (X^T * Y)
    #
    #     B = (X^T * X)^-1 * (X^T * Y) 
    # xtxi = np.linalg.pinv(x.T @ x) # xtxi means (X^T * X)^-1
    # beta = xtxi @ (x.T @y)
    # return beta


    # With ridge regularization
    #   Solve for B in the following equation:
    #       [(X^T * X)/N + (lam * I)]B = (X^T * Y)/N
    #
    #       B = [(X^T * X)/N + (lam * I)]^-1 * (X^T * Y)/N
    xtxn = x.T @ x / N
    xtyn = x.T @ y / N
    lami = lam * np.eye(d+1)
    beta = np.linalg.pinv(xtxn + lami) @ xtyn
    return beta

if __name__ == '__main__':
    # Performing sanity checks on your implementation
    print('Check 1')
    some_X = (np.arange(35).reshape(7,5) ** 13) % 20
    some_Y = np.sum(some_X, axis=1)
    some_beta = linear_regression(some_X, some_Y, lam=0)
    assert np.array_equal(some_beta.round(3), np.array([[ 0.],
                                                        [ 1.],
                                                        [ 1.],
                                                        [ 1.],
                                                        [ 1.],
                                                        [ 1.]]))

    print('Check 2')
    some_beta_2 = linear_regression(some_X, some_Y, lam=1)
    assert np.array_equal(some_beta_2.round(3), np.array([[0.032],
                                                          [0.887],
                                                          [1.08 ],
                                                          [1.035],
                                                          [0.86 ],
                                                          [1.021]]))

    print('Check 3')
    another_X = some_X.T
    another_Y = np.sum(another_X, axis=1)
    another_beta = linear_regression(another_X, another_Y, lam=0)
    assert np.array_equal(another_beta.round(3), np.array([[-0.01 ],
                                                           [ 0.995],
                                                           [ 1.096],
                                                           [ 0.993],
                                                           [ 0.996],
                                                           [ 0.995],
                                                           [ 0.946],
                                                           [ 0.966]]))

    print('All checks passed!')
