import numpy as np

def linear_predict(X,beta):
    """
    Predict with linear regression model

        Parameters:
                X (np.array): A numpy array with the shape (N, d) where N is the number of data points and d is dimension
                beta (np.array): A numpy array of the shape (d+1,1) where d is the data dimension
                
        Returns:
                y_hat (np.array): A numpy array with the shape (N, )
    """
    assert X.ndim==2
    N = X.shape[0]
    d = X.shape[1]
    assert beta.shape == (d+1,1)

    # Prepend a column of ones to X, to give us a Y intercept
    ones = np.ones((N, 1))
    x = np.hstack((ones, X))

    y = x @ beta
    return y.reshape(-1)
    
if __name__ == '__main__':
    # Performing sanity checks on your implementation
    some_X = (np.arange(35).reshape(7,5) ** 13) % 20
    some_beta = 2.**(-np.arange(6).reshape(-1,1))
    some_yhat = linear_predict(some_X, some_beta)
    assert np.array_equal(some_yhat.round(3), np.array([ 3.062,  9.156,  6.188, 15.719,  3.062,  9.281,  7.062]))

    print('All checks passed')
